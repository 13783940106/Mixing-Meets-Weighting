import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, WeakStrongAugment
from networks.net_factory import BCP_net
from utils import losses
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, required=True)
parser.add_argument("--labeled_list", type=str, required=True)
parser.add_argument("--val_list", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--run_root", type=str, required=True)
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--pre_iterations", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--labeled_bs", type=int, default=12)
parser.add_argument("--labelnum", type=int, default=12)
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)
ce_loss = nn.CrossEntropyLoss()


def save_net_opt(net, optimizer, path):
    torch.save({"net": net.state_dict(), "opt": optimizer.state_dict()}, str(path))


def generate_mask(img):
    batch_size, _, img_x, img_y = img.shape
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 3 / 5), int(img_y * 3 / 5)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    ce = nn.CrossEntropyLoss(reduction="none")
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (ce(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (ce(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce


class RepeatBatchSampler(Sampler):
    def __init__(self, indices, batch_size):
        self.indices = list(indices)
        self.batch_size = batch_size
        if len(self.indices) < batch_size:
            raise ValueError("labeled list must contain at least batch_size samples")

    def __iter__(self):
        while True:
            perm = np.random.permutation(self.indices).tolist()
            for i in range(0, len(perm) - self.batch_size + 1, self.batch_size):
                yield perm[i:i + self.batch_size]

    def __len__(self):
        return len(self.indices) // self.batch_size


def volume_id_from_slice(slice_id):
    return "_".join(slice_id.split("_")[:2])


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(run_root / "train.log"), level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    labeled_entries = [x.strip() for x in Path(args.labeled_list).read_text().splitlines() if x.strip()]
    val_entries = [x.strip() for x in Path(args.val_list).read_text().splitlines() if x.strip()]
    labeled_cases = sorted({volume_id_from_slice(x) for x in labeled_entries})
    Path(output_dir / "labeled_cases.txt").write_text("\n".join(labeled_cases) + "\n")
    Path(output_dir / "val_cases.txt").write_text("\n".join(val_entries) + "\n")

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([WeakStrongAugment(args.image_size)]), list_file=args.labeled_list)
    db_val = BaseDataSets(base_dir=args.root_path, split="val", list_file=args.val_list)
    trainloader = DataLoader(db_train, batch_sampler=RepeatBatchSampler(range(len(db_train)), args.batch_size), num_workers=4, pin_memory=True, worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id))
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model = BCP_net(in_chns=1, class_num=args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(str(run_root / "log"))
    logging.info("Start clean supervised pre_training")
    logging.info("%s iterations per epoch", len(trainloader))

    iter_num = 0
    max_epoch = args.pre_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_step = 0
    model.train()
    for _ in tqdm(range(max_epoch), ncols=70):
        for sampled_batch in trainloader:
            volume_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()
            img_a, img_b = volume_batch[:args.labeled_bs // 2], volume_batch[args.labeled_bs // 2:args.labeled_bs]
            lab_a, lab_b = label_batch[:args.labeled_bs // 2], label_batch[args.labeled_bs // 2:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            loss = (loss_dice + loss_ce) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/mix_dice", loss_dice, iter_num)
            writer.add_scalar("info/mix_ce", loss_ce, iter_num)
            logging.info("iteration %d: loss: %f, mix_dice: %f, mix_ce: %f", iter_num, loss, loss_dice, loss_ce)

            if iter_num % 20 == 0:
                writer.add_image("pre_train/Mixed_Image", net_input[1, 0:1, :, :], iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image("pre_train/Mixed_Prediction", outputs[1, ...] * 50, iter_num)
                writer.add_image("pre_train/Mixed_GroundTruth", gt_mixl[1, ...].unsqueeze(0) * 50, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for sampled_val in valloader:
                    metric_i = test_single_volume(sampled_val["image"], sampled_val["label"], model, classes=args.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(args.num_classes - 1):
                    writer.add_scalar("info/val_{}_dice".format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar("info/val_{}_hd95".format(class_i + 1), metric_list[class_i, 1], iter_num)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                if performance > best_performance:
                    best_performance = float(performance)
                    best_step = iter_num
                    save_net_opt(model, optimizer, output_dir / "iter_{}_dice_{}.pth".format(iter_num, round(best_performance, 4)))
                    save_net_opt(model, optimizer, output_dir / "unet_best_model.pth")
                logging.info("iteration %d : mean_dice : %f", iter_num, performance)
                model.train()

            if iter_num >= args.pre_iterations:
                break
        if iter_num >= args.pre_iterations:
            break

    metadata = {
        "split_name": "clean",
        "labeled_frame_volumes": len(labeled_cases),
        "labeled_slices": len(labeled_entries),
        "val_frame_volumes": len(val_entries),
        "seed": args.seed,
        "labeled_list": args.labeled_list,
        "val_list": args.val_list,
        "training_entry": __file__,
        "model_type": "BCP_net/UNet_2d",
        "pre_iterations": args.pre_iterations,
        "best_validation_metric": best_performance,
        "best_step": best_step,
    }
    (output_dir / "pretrain_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    writer.close()


if __name__ == "__main__":
    main()
