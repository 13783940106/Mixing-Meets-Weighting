import os
# import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
import re


def _parse_slice_key(slice_name):
    """Parse patientXXX_frameYY_slice_Z -> (patient_id, frame_id, slice_num)."""
    m = re.match(r"(patient\d+)_(frame\d+)_slice_(\d+)", slice_name)
    if m is None:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _make_slice_pattern(patient, frame, slice_num):
    """Build slice filename for a given triplet position."""
    return f"{patient}_{frame}_slice_{slice_num}"


class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
            list_file=None,
            input_mode="2d",
            context_slices=1,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.input_mode = input_mode
        self.context_slices = context_slices
        self._slice_index = {}   # (patient, frame) -> set of slice numbers

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if list_file is not None:
            with open(list_file, "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.strip() for item in self.sample_list if item.strip()]
            print("{} total {} samples from {}".format(self.split, len(self.sample_list), list_file))
        elif self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print("Train total {} samples".format(len(self.sample_list)))

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print("Val total {} samples".format(len(self.sample_list)))
        elif self.split == "test":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print("Val total {} samples".format(len(self.sample_list)))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]

        # Build slice lookup for 2.5D triplet construction (train only)
        if self.input_mode in ("25d_3ch", "25d_neighbor_relation") and self.split == "train":
            self._build_slice_index()

    def _build_slice_index(self):
        """Build (patient, frame) -> sorted slice_num list for triplet lookup."""
        self._slice_index = {}
        for case in self.sample_list:
            parsed = _parse_slice_key(case)
            if parsed is None:
                continue
            patient, frame, sn = parsed
            key = (patient, frame)
            if key not in self._slice_index:
                self._slice_index[key] = set()
            self._slice_index[key].add(sn)
        # Convert sets to sorted lists
        for key in self._slice_index:
            self._slice_index[key] = sorted(self._slice_index[key])
        print("Slice index built: {} (patient,frame) groups".format(len(self._slice_index)))

    def _get_neighbors(self, patient, frame, center_sn):
        """Return (prev_sn, next_sn) with edge-repeat policy."""
        key = (patient, frame)
        slices = self._slice_index.get(key, [])
        if len(slices) == 0:
            return center_sn, center_sn
        idx = slices.index(center_sn) if center_sn in slices else -1
        if idx < 0:
            return center_sn, center_sn
        prev_sn = slices[idx - 1] if idx > 0 else center_sn   # edge repeat
        next_sn = slices[idx + 1] if idx < len(slices) - 1 else center_sn  # edge repeat
        return prev_sn, next_sn

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            center_path = self._base_dir + "/data/slices/{}.h5".format(case)
            if self.input_mode in ("25d_3ch", "25d_neighbor_relation"):
                parsed = _parse_slice_key(case)
                if parsed is not None:
                    patient, frame, center_sn = parsed
                    prev_sn, next_sn = self._get_neighbors(patient, frame, center_sn)
                    prev_case = _make_slice_pattern(patient, frame, prev_sn)
                    next_case = _make_slice_pattern(patient, frame, next_sn)
                    prev_path = self._base_dir + "/data/slices/{}.h5".format(prev_case)
                    next_path = self._base_dir + "/data/slices/{}.h5".format(next_case)
                else:
                    prev_path = center_path
                    next_path = center_path
                    center_sn = -1
                    prev_sn, next_sn = center_sn, center_sn

                with h5py.File(prev_path, "r") as pf:
                    prev_img = pf["image"][:]
                with h5py.File(center_path, "r") as cf:
                    center_img = cf["image"][:]
                    label = cf["label"][:]
                with h5py.File(next_path, "r") as nf:
                    next_img = nf["image"][:]

                image = np.stack([prev_img, center_img, next_img], axis=0)  # [3, H, W]
                prev_valid = 0.0 if prev_sn == center_sn else 1.0
                next_valid = 0.0 if next_sn == center_sn else 1.0
                neighbor_validity = np.stack(
                    [
                        np.full_like(center_img, prev_valid, dtype=np.float32),
                        np.full_like(center_img, next_valid, dtype=np.float32),
                    ],
                    axis=0,
                )
                neighbor_policy = "normal"
                if prev_sn == center_sn:
                    neighbor_policy = "left_edge_repeat"
                elif next_sn == center_sn:
                    neighbor_policy = "right_edge_repeat"

                sample = {
                    "image": image, "label": label,
                    "case_id": case, "slice_id": case,
                    "center_slice_id": center_sn,
                    "prev_slice_id": prev_sn,
                    "next_slice_id": next_sn,
                    "neighbor_policy": neighbor_policy,
                    "input_mode": self.input_mode,
                }
                if self.input_mode == "25d_neighbor_relation":
                    sample["neighbor_validity"] = neighbor_validity
            else:
                # 2D mode (original behavior)
                h5f = h5py.File(center_path, "r")
                image = h5f["image"][:]
                label = h5f["label"][:]
                sample = {
                    "image": image, "label": label,
                    "case_id": case, "slice_id": case,
                    "center_slice_id": -1,
                    "prev_slice_id": -1,
                    "next_slice_id": -1,
                    "neighbor_policy": "n/a",
                    "input_mode": "2d",
                }
        else:
            # val/test: load full volume, extract center slice data for triplet
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
            volume_image = h5f["image"][:]      # [D, H, W]
            volume_label = h5f["label"][:]      # [D, H, W]
            D = volume_image.shape[0]
            # We still return the full volume; post-processing in training loop handles per-slice eval
            sample = {
                "image": volume_image, "label": volume_label,
                "case_id": case, "slice_id": case,
                "center_slice_id": -1,
                "prev_slice_id": -1,
                "next_slice_id": -1,
                "neighbor_policy": "n/a",
                "input_mode": self.input_mode if self.input_mode not in ("25d_3ch", "25d_neighbor_relation") else "volume_25d",  # volume mode for eval
                "volume_depth": D,
            }

        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        if self.split == "train":
            sample["case_id"] = "_".join(case.split("_")[:2])
            sample["slice_id"] = case
        sample["idx"] = idx
        return sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)  
    image = np.rot90(image, k)  
    axis = np.random.randint(0, 2)  
    image = np.flip(image, axis=axis).copy()  
    if label is not None:  
        label = np.rot90(label, k)  
        label = np.flip(label, axis=axis).copy()  
        return image, label
    else:
        return image

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)  
    image = ndimage.rotate(image, angle, order=0, reshape=False)  
    label = ndimage.rotate(label, angle, order=0, reshape=False)  
    return image, label

def blur(image,p=0.5):
    if random.random()<p:
        max = np.max(image)
        min = np.min(image)
        sigma = np.random.uniform(0.1,2.0)
        image = Image.fromarray(((image-min)/(max-min)*255).astype('uint8'))
        image = np.array(image.filter(ImageFilter.GaussianBlur(radius=sigma)))
        image = min + image*(max-min)/255
    return image
def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def cutout_gray(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=1, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.randint(value_min, value_max + 1, (erase_h, erase_w))
        else:
            value = np.random.randint(value_min, value_max + 1)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 0

    return img, mask


def cutout_25d_shared_spatial(img, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=1, pixel_level=True):
    """Apply one shared cutout rectangle to a [3, H, W] 2.5D triplet."""
    if random.random() >= p:
        return img

    img = np.array(img).copy()
    if img.ndim != 3:
        raise ValueError("cutout_25d_shared_spatial expects [C, H, W] input")

    _, img_h, img_w = img.shape

    while True:
        size = np.random.uniform(size_min, size_max) * img_h * img_w
        ratio = np.random.uniform(ratio_1, ratio_2)
        erase_w = int(np.sqrt(size / ratio))
        erase_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_w)
        y = np.random.randint(0, img_h)

        if x + erase_w <= img_w and y + erase_h <= img_h:
            break

    if pixel_level:
        value = np.random.randint(value_min, value_max + 1, (erase_h, erase_w))
    else:
        value = np.random.randint(value_min, value_max + 1)

    img[:, y:y + erase_h, x:x + erase_w] = value
    return img


def shared_brightness_contrast_25d(image, brightness=0.8, contrast=0.8):
    """Apply the same brightness/contrast factors to every slice channel."""
    if image.ndim != 3:
        raise ValueError("shared_brightness_contrast_25d expects [C, H, W] input")

    out = image.float()
    if brightness > 0:
        brightness_factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        out = out * brightness_factor
    if contrast > 0:
        contrast_factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        channel_mean = out.mean(dim=(1, 2), keepdim=True)
        out = (out - channel_mean) * contrast_factor + channel_mean
    return torch.clamp(out, 0.0, 1.0)

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        is_25d = isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] == 3

        if is_25d:
            # [3, H, W] -> apply SAME spatial transform to all 3 channels.
            validity = sample.get("neighbor_validity", None)
            parts = [image, label[np.newaxis, :, :]]
            if validity is not None:
                parts.append(validity)
            combined = np.concatenate(parts, axis=0)  # [4 or 6, H, W]
            if random.random() > 0.5:
                combined = self._random_rot_flip_multich(combined)
            elif random.random() > 0.5:
                combined = self._random_rotate_multich(combined)
            H_in, W_in = combined.shape[1], combined.shape[2]
            resized = np.stack([
                zoom(combined[i], (self.output_size[0] / H_in, self.output_size[1] / W_in), order=0)
                for i in range(combined.shape[0])
            ], axis=0)
            image = torch.from_numpy(resized[:3].astype(np.float32))        # [3, H, W]
            label = torch.from_numpy(resized[3].astype(np.uint8))           # [H, W]
            validity_t = torch.from_numpy(resized[4:6].astype(np.float32)) if validity is not None else None
        else:
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # [1, H, W]
            label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        if is_25d and validity_t is not None:
            sample["neighbor_validity"] = validity_t
        return sample

    @staticmethod
    def _random_rot_flip_multich(combined):
        k = np.random.randint(0, 4)
        combined = np.rot90(combined, k, axes=(1, 2))
        axis = np.random.randint(0, 2)
        combined = np.flip(combined, axis=axis + 1).copy()
        return combined

    @staticmethod
    def _random_rotate_multich(combined):
        angle = np.random.randint(-20, 20)
        C = combined.shape[0]
        rotated = np.stack([
            ndimage.rotate(combined[i], angle, order=0, reshape=False)
            for i in range(C)
        ], axis=0)
        return rotated


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images
    Args:
        object (tuple): output size of network
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        is_25d = isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] == 3

        if is_25d:
            # [3, H, W] shared spatial transform
            validity = sample.get("neighbor_validity", None)
            parts = [image, label[np.newaxis, :, :]]
            if validity is not None:
                parts.append(validity)
            combined = np.concatenate(parts, axis=0)  # [4 or 6, H, W]
            if random.random() > 0.5:
                combined = RandomGenerator._random_rot_flip_multich(combined)
            elif random.random() > 0.5:
                combined = RandomGenerator._random_rotate_multich(combined)
            H_in, W_in = combined.shape[1], combined.shape[2]
            resized = np.stack([
                zoom(combined[i], (self.output_size[0] / H_in, self.output_size[1] / W_in), order=0)
                for i in range(combined.shape[0])
            ], axis=0)
            image_weak = resized[:3].astype(np.float32)   # [3, H, W]
            label_img = resized[3]
            validity_t = torch.from_numpy(resized[4:6].astype(np.float32)) if validity is not None else None

            # Strong augmentation: use shared spatial/intensity parameters for the triplet.
            image_strong = cutout_25d_shared_spatial(np.copy(image_weak), p=0.5)
            image_strong_t = shared_brightness_contrast_25d(torch.from_numpy(image_strong))

            image_weak_t = torch.from_numpy(image_weak)
            label_t = torch.from_numpy(label_img.astype(np.uint8))
            label_strong_t = label_t.clone()  # same spatial label

            sample = {
                "image": image_weak_t,
                "image_strong": image_strong_t,
                "label": label_t,
                "label_strong": label_strong_t,
            }
            if validity_t is not None:
                sample["neighbor_validity"] = validity_t
                sample["neighbor_validity_strong"] = validity_t.clone()
        else:
            # Original 2D code path (unchanged)
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

            image_strong, label_strong = cutout_gray(image, label, p=0.5)
            image_strong = color_jitter(image_strong).type("torch.FloatTensor")
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

            label = torch.from_numpy(label.astype(np.uint8))
            label_strong = torch.from_numpy(label_strong.astype(np.uint8))
            sample = {
                "image": image,
                "image_strong": image_strong,
                "label": label,
                "label_strong": label_strong
            }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size),
        )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
