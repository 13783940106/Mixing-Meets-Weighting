import argparse
import logging
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch, gc
from medpy import metric
from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment)
from dataloaders.promise12 import Promise12, cutout_gray, color_jitter
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.config import get_config
from utils import losses, ramps
from val_2D import test_single_volume_promise
from utils.displacement import process_image_batches
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default="../AD-MT/code/data_promise12", help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='train_PROMISE12', help='experiment_name')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model1_name')
parser.add_argument('--model_2', type=str,
                    default='swin_unet', help='model2_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--image_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--cfg', type=str,
                    default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+', )
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--input_mode', type=str, default='2d', choices=['2d', '25d_3ch'],
                    help='input mode: 2d keeps the canonical parent behavior; 25d_3ch uses prev/center/next')
parser.add_argument('--center_anchor_reg_weight', type=float, default=0.0,
                    help='default-off P-B 2.5D center-anchor regularization on first conv prev/next channels')
parser.add_argument('--boundary_neighbor_suppression', action='store_true',
                    help='default-off P-B 2.5D boundary-risk neighbor suppression before model forward')
parser.add_argument('--boundary_neighbor_suppression_strength', type=float, default=0.75,
                    help='blend strength for high-risk prev/next pixels; formal v1 uses the default')
parser.add_argument('--boundary_neighbor_suppression_quantile', type=float, default=0.85,
                    help='per-sample high-risk quantile for cross-slice difference mask; formal v1 uses the default')
parser.add_argument('--shape_audit_only', action='store_true',
                    help='run a default-off dataloader/model shape audit and exit')
parser.add_argument('--phase_transition_audit_only', action='store_true',
                    help='run a default-off pretrain-to-self-train handoff audit and exit')
parser.add_argument('--audit_output', type=str, default=None,
                    help='optional audit output path for audit-only modes')
parser.add_argument('--pretrain_max_iterations_override', type=int, default=None,
                    help='default-off engineering override; formal run leaves this unset')
parser.add_argument('--self_train_max_iterations_override', type=int, default=None,
                    help='default-off engineering override; formal run leaves this unset')
# costs
parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# patch size
parser.add_argument('--patch_size', type=int, default=56, help='patch_size')
parser.add_argument('--h_size', type=int, default=4, help='h_size')
parser.add_argument('--w_size', type=int, default=4, help='w_size')
# top num
parser.add_argument('--top_num', type=int, default=4, help='top_num')
args = parser.parse_args()
config = get_config(args)
dice_loss = losses.DiceLoss(2)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch,args.consistency_rampup)  # args.consistency=0.1 # args.consistency_rampup=200

def center_anchor_regularization(model):
    anchor = next(model.parameters()).sum() * 0.0
    if args.input_mode != '25d_3ch' or args.center_anchor_reg_weight <= 0:
        return anchor
    weight = model.encoder.in_conv.conv_conv[0].weight
    if weight.ndim != 4 or weight.shape[1] != 3:
        return anchor
    side_l2 = weight[:, 0].pow(2).mean() + weight[:, 2].pow(2).mean()
    return args.center_anchor_reg_weight * side_l2

def first_conv_channel_l1_fractions(model):
    weight = model.encoder.in_conv.conv_conv[0].weight.detach()
    if weight.ndim != 4 or weight.shape[1] != 3:
        return (float('nan'), float('nan'), float('nan'))
    vals = weight.abs().sum(dim=(0, 2, 3))
    total = vals.sum().clamp_min(1e-12)
    return tuple((vals / total).cpu().tolist())

def boundary_neighbor_suppression_mask(batch):
    if args.input_mode != '25d_3ch' or not args.boundary_neighbor_suppression:
        return None
    if batch.ndim != 4 or batch.shape[1] != 3:
        return None
    center = batch[:, 1:2]
    diff_prev = (batch[:, 0:1] - center).abs()
    diff_next = (batch[:, 2:3] - center).abs()
    risk_signal = F.avg_pool2d(torch.maximum(diff_prev, diff_next), kernel_size=3, stride=1, padding=1)
    flat = risk_signal.detach().flatten(1)
    q = min(max(args.boundary_neighbor_suppression_quantile, 0.0), 1.0)
    threshold = torch.quantile(flat, q, dim=1, keepdim=True).view(-1, 1, 1, 1)
    return (risk_signal >= threshold).to(dtype=batch.dtype)

def prepare_25d_input_for_model(batch):
    mask = boundary_neighbor_suppression_mask(batch)
    if mask is None:
        return batch
    strength = min(max(args.boundary_neighbor_suppression_strength, 0.0), 1.0)
    center = batch[:, 1:2]
    out = batch.clone()
    out[:, 0:1] = batch[:, 0:1] * (1.0 - strength * mask) + center * (strength * mask)
    out[:, 2:3] = batch[:, 2:3] * (1.0 - strength * mask) + center * (strength * mask)
    return out

def boundary_neighbor_suppression_stats(batch):
    mask = boundary_neighbor_suppression_mask(batch)
    if mask is None:
        return (0.0, 0.0)
    with torch.no_grad():
        prepared = prepare_25d_input_for_model(batch)
        return (float(mask.mean().detach().cpu()), float((prepared - batch).abs().mean().detach().cpu()))

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    return dice

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*1/4), int(img_y*1/4)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    # 相当于           11111       这样生成模拟遮盖或缺失区域
    #                 11111
    #                 00000
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    # 交叉熵损失初始化
    CE = nn.CrossEntropyLoss(reduction='none')
    # 将 img_l 和 patch_l 转换为 torch.int64 类型：
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    # 对 output 进行 softmax 以得到每个类别的概率
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    # 是否使用权重
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    # 通过取反得到补丁掩码：
    patch_mask = 1 - mask
    # 对图像和补丁分别计算 Dice 损失，并乘以相应权重：
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

class Promise12_25D:
    """Center-supervised prev/center/next triplet wrapper for the P-B 2.5D formal rebuild."""
    def __init__(self, data_dir, mode, out_size):
        self.base = Promise12(data_dir, mode, out_size)
        self.mode = mode
        if self.mode == 'train':
            self.X_train = self.base.X_train
            self.y_train = self.base.y_train
            self.N = self.X_train.shape[0]

    def __len__(self):
        return len(self.base)

    def _triplet_from_train_index(self, idx):
        prev_idx = idx - 1 if idx > 0 else idx
        next_idx = idx + 1 if idx < self.N - 1 else idx
        return np.stack([self.X_train[prev_idx], self.X_train[idx], self.X_train[next_idx]], axis=0).astype(np.float32)

    def __getitem__(self, idx):
        if self.mode != 'train':
            return self.base[idx]

        img25 = self._triplet_from_train_index(idx)
        mask = self.y_train[idx].copy()
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            axis = np.random.randint(0, 2)
            img25 = np.rot90(img25, k, axes=(1, 2))
            img25 = np.flip(img25, axis=axis + 1).copy()
            mask = np.rot90(mask, k)
            mask = np.flip(mask, axis=axis).copy()
        elif random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            img25 = np.stack([ndimage.rotate(img25[c], angle, order=0, reshape=False) for c in range(3)], axis=0)
            mask = ndimage.rotate(mask, angle, order=0, reshape=False)

        center_aug, mask_strong_np = cutout_gray(img25[1].copy(), mask.copy(), p=0.5)
        image = torch.from_numpy(img25.astype(np.float32))
        image_strong = image.clone()
        image_strong[1] = color_jitter(torch.from_numpy(center_aug).unsqueeze(0).float()).squeeze(0)
        return {
            "image": image,
            "image_strong": image_strong,
            "mask": torch.from_numpy(mask.astype(np.uint8)),
            "mask_strong": torch.from_numpy(mask_strong_np.astype(np.uint8)),
        }

def test_single_volume_promise_25d(image, label, net, classes):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        prev_ind = ind - 1 if ind > 0 else ind
        next_ind = ind + 1 if ind < image.shape[0] - 1 else ind
        triplet = np.stack([image[prev_ind], image[ind], image[next_ind]], axis=0).astype(np.float32)
        input_tensor = torch.from_numpy(triplet).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(prepare_25d_input_for_model(input_tensor)), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def model_input_channels():
    return 3 if args.input_mode == '25d_3ch' else 1

def convert_center_identity_state_1ch_to_3ch(state_dict):
    converted = {}
    for key, value in state_dict.items():
        if value.ndim == 4 and value.shape[1] == 1:
            new_value = torch.zeros(value.shape[0], 3, value.shape[2], value.shape[3], dtype=value.dtype)
            new_value[:, 1:2, :, :] = value
            converted[key] = new_value
        else:
            converted[key] = value
    return converted

def create_center_identity_25d_model(ema=False):
    model_2d = net_factory(net_type=args.model_1, in_chns=1, class_num=args.num_classes)
    state_2d = {k: v.detach().cpu().clone() for k, v in model_2d.state_dict().items()}
    del model_2d
    model = net_factory(net_type=args.model_1, in_chns=3, class_num=args.num_classes)
    model.load_state_dict(convert_center_identity_state_1ch_to_3ch(state_2d), strict=True)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def process_image_batches_for_model(image_batch_a, image_batch_b, threshold):
    if args.input_mode != '25d_3ch':
        return process_image_batches(image_batch_a, image_batch_b, threshold).unsqueeze(1)

    assert image_batch_a.shape == image_batch_b.shape
    _, _, rows, cols = image_batch_a.shape
    mask = torch.zeros((1, 1, rows, cols), dtype=torch.float32, device=image_batch_a.device)
    crow, ccol = rows // 2, cols // 2
    mask[:, :, crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1

    def low_freq(batch):
        f_transform = torch.fft.fft2(batch, dim=(-2, -1))
        f_shift = torch.fft.fftshift(f_transform, dim=(-2, -1))
        f_low_freq = f_shift * mask
        f_ishift = torch.fft.ifftshift(f_low_freq, dim=(-2, -1))
        return torch.abs(torch.fft.ifft2(f_ishift, dim=(-2, -1)))

    low_a = low_freq(image_batch_a)
    low_b = low_freq(image_batch_b)
    swapped_a = image_batch_a - low_a + low_b
    swapped_b = image_batch_b - low_b + low_a
    return torch.cat([swapped_a, swapped_b], dim=0)

def write_audit_lines(lines):
    text = "\n".join(lines) + "\n"
    if args.audit_output:
        out_dir = os.path.dirname(args.audit_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.audit_output, "w") as f:
            f.write(text)
    print(text)

def run_shape_audit_only():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    lines = [
        "# P-B 2.5D Shape Audit Only",
        "",
        f"input_mode: {args.input_mode}",
        "test_access: none",
    ]
    db_train = Promise12_25D(args.root_path, mode='train', out_size=224) if args.input_mode == '25d_3ch' else Promise12(args.root_path, mode='train', out_size=224)
    db_val = Promise12(args.root_path, mode='val', out_size=224)
    sample = db_train[0]
    image = sample["image"].unsqueeze(0).float().cuda()
    mask = sample["mask"].unsqueeze(0).long().cuda()
    model = create_model().cuda().eval()
    with torch.no_grad():
        output = model(prepare_25d_input_for_model(image))
    lines.extend([
        f"train_image_shape: {tuple(image.shape)}",
        f"train_mask_shape: {tuple(mask.shape)}",
        f"model_first_conv_shape: {tuple(model.encoder.in_conv.conv_conv[0].weight.shape)}",
        f"model_output_shape: {tuple(output.shape)}",
        f"val_case_count: {len(db_val)}",
        f"boundary_neighbor_suppression: {args.boundary_neighbor_suppression}",
        f"boundary_neighbor_suppression_strength: {args.boundary_neighbor_suppression_strength}",
        f"boundary_neighbor_suppression_quantile: {args.boundary_neighbor_suppression_quantile}",
    ])
    if args.input_mode == '25d_3ch':
        mask_ratio, input_delta = boundary_neighbor_suppression_stats(image)
        lines.extend([
            f"boundary_neighbor_suppression_mask_ratio: {mask_ratio:.6f}",
            f"boundary_neighbor_suppression_input_abs_delta: {input_delta:.6f}",
        ])
    if args.input_mode == '25d_3ch':
        model_2d = net_factory(net_type=args.model_1, in_chns=1, class_num=args.num_classes).cuda().eval()
        state_2d = {k: v.detach().cpu().clone() for k, v in model_2d.state_dict().items()}
        model_25d = net_factory(net_type=args.model_1, in_chns=3, class_num=args.num_classes).cuda().eval()
        model_25d.load_state_dict(convert_center_identity_state_1ch_to_3ch(state_2d), strict=True)
        with torch.no_grad():
            out_2d = model_2d(image[:, 1:2])
            out_25d = model_25d(image)
        center_identity_error = torch.max(torch.abs(out_2d - out_25d)).item()
        lines.extend([
            "triplet_order: prev[0] / center[1] / next[2]",
            f"center_identity_output_max_error: {center_identity_error:.10f}",
        ])
    finite = torch.isfinite(output).all().item()
    lines.append(f"output_finite: {finite}")
    lines.append("decision: PASS" if finite else "decision: FAIL")
    write_audit_lines(lines)

def run_phase_transition_audit_only():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    audit_base = args.audit_output
    if audit_base and audit_base.endswith(".md"):
        audit_dir = os.path.splitext(audit_base)[0] + "_tmp"
    else:
        audit_dir = audit_base or "./phase_transition_audit_tmp"
    pre_snapshot_path = os.path.join(audit_dir, "pre_train")
    snapshot_path = os.path.join(audit_dir, "self_train")
    os.makedirs(pre_snapshot_path, exist_ok=True)
    os.makedirs(snapshot_path, exist_ok=True)

    pre_model = create_model()
    optimizer = optim.SGD(pre_model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    save_net_opt(pre_model, optimizer, os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model_1)))
    old_override = args.self_train_max_iterations_override
    args.self_train_max_iterations_override = 1
    try:
        self_train(args, pre_snapshot_path, snapshot_path)
        lines = [
            "# P-B 2.5D Phase Transition Audit Only",
            "",
            f"input_mode: {args.input_mode}",
            "pretrain_checkpoint_format: save_net_opt",
            "self_train_iterations_executed: 1",
            "handoff_load: strict load_net/load_net_opt",
            "pseudo_label_path: executed",
            "bcp_path: executed",
            "fft_path: executed",
            "backward: executed",
            "optimizer_step: executed",
            "ema_update: executed",
            "test_access: none",
            "decision: PASS",
        ]
    except Exception as exc:
        lines = [
            "# P-B 2.5D Phase Transition Audit Only",
            "",
            f"input_mode: {args.input_mode}",
            f"error: {type(exc).__name__}: {exc}",
            "decision: FAIL",
        ]
        write_audit_lines(lines)
        raise
    finally:
        args.self_train_max_iterations_override = old_override
    write_audit_lines(lines)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pretrain_max_iterations_override or 10000
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)
    batch_size = args.batch_size

    model = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Promise12_25D(args.root_path, mode='train', out_size=224) if args.input_mode == '25d_3ch' else Promise12(args.root_path, mode='train', out_size=224)
    db_val = Promise12(args.root_path, mode='val', out_size=224)
    
    total_slices = len(db_train)
    labeled_slice = 202 # 标记的切片数目
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    # batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
    #                                       args.batch_size - args.labeled_bs)
    #
    # trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
    #                          worker_init_fn=worker_init_fn)
    #
    # valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)  # args.labeled_bs=8
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)



    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            # gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(prepare_25d_input_for_model(net_input))
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            center_anchor_reg = center_anchor_regularization(model)
            loss = (loss_dice + loss_ce) / 2 + center_anchor_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/center_anchor_reg', center_anchor_reg, iter_num)

            if args.input_mode == '25d_3ch' and iter_num % 200 == 0:
                prev_frac, center_frac, next_frac = first_conv_channel_l1_fractions(model)
                suppression_mask_ratio, suppression_input_delta = boundary_neighbor_suppression_stats(net_input)
                logging.info('iteration %d: center_anchor_reg: %f, first_conv_l1_frac_prev_center_next: %.6f %.6f %.6f' %
                             (iter_num, center_anchor_reg.item(), prev_frac, center_frac, next_frac))
                logging.info('iteration %d: boundary_neighbor_suppression_enabled: %s, suppression_mask_ratio: %.6f, suppression_input_abs_delta: %.6f' %
                             (iter_num, str(args.boundary_neighbor_suppression), suppression_mask_ratio, suppression_input_delta))

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))



            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    if args.input_mode == '25d_3ch':
                        metric_i = test_single_volume_promise_25d(sampled_batch["image"], sampled_batch["mask"], model,
                                                              classes=num_classes)
                    else:
                        metric_i = test_single_volume_promise(sampled_batch["image"], sampled_batch["mask"], model,
                                                              classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)



                performance = np.mean(metric_list, axis=0)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model_1))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    if args.pretrain_max_iterations_override is not None:
        save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model_1))
        if not os.path.exists(save_best_path):
            save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
            save_net_opt(model, optimizer, save_mode_path)
            save_net_opt(model, optimizer, save_best_path)
            logging.info('engineering_override_saved_pretrain_checkpoint iteration %d : mean_dice : %f' % (iter_num, best_performance))
    writer.close()

def create_model(ema=False):
    # Network definition
    if args.input_mode == '25d_3ch':
        return create_center_identity_25d_model(ema=ema)
    model = net_factory(net_type=args.model_1, in_chns=1, class_num=2)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model
def self_train(args, pre_snapshot_path,snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)
    max_iterations = args.self_train_max_iterations_override or args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model_1))


    model1 = create_model()
    model2 = create_model()
    ema_model = create_model(ema=True)

    # model2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()
    # model2.load_from(config)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Promise12_25D(args.root_path, mode='train', out_size=224) if args.input_mode == '25d_3ch' else Promise12(args.root_path, mode='train', out_size=224)
    db_val = Promise12(args.root_path, mode='val', out_size=224)
# P-B PATCH: test data loading removed for strict test-once protocol

    total_slices = len(db_train)
    labeled_slice = 202  # args.labeled_num=7
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)  # args.labeled_bs=8
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    
    
# P-B PATCH: test dataloader removed for strict test-once protocol

    model1.train()
    model2.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    load_net(ema_model, pre_trained_model)
    load_net_opt(model1, optimizer1, pre_trained_model)
    load_net_opt(model2, optimizer2, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1

    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)  #

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['mask_strong']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            # bcp
            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            # unlabeled_sub_bs = (batch_size - labeled_bs)/2
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[
                                                                                               args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]

            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[
                                 args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[
                                                                                      args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a = ema_model(prepare_25d_input_for_model(uimg_a))
                pre_b = ema_model(prepare_25d_input_for_model(uimg_b))
                pre_a_s = ema_model(prepare_25d_input_for_model(uimg_a_s))
                pre_b_s = ema_model(prepare_25d_input_for_model(uimg_b_s))
                
                plab_a = torch.argmax(pre_a, dim=1, keepdim=False)
                plab_b = torch.argmax(pre_b, dim=1, keepdim=False)
                plab_a_s = torch.argmax(pre_a_s, dim=1, keepdim=False)
                plab_b_s = torch.argmax(pre_b_s, dim=1, keepdim=False)
                
                img_mask, loss_mask = generate_mask(img_a)


            #
            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0) 

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask)
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)
            
             # Model1 Loss
            out_unl_1 = model1(prepare_25d_input_for_model(net_input_unl_1))
            out_l_1 = model1(prepare_25d_input_for_model(net_input_l_1))
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            out_soft_1 = torch.softmax(out_1, dim=1)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2 = model2(prepare_25d_input_for_model(net_input_unl_2))
            out_l_2 = model2(prepare_25d_input_for_model(net_input_l_2))
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_soft_2 = torch.softmax(out_2, dim=1)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False) 
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            # Model1 & Model2 Cross Pseudo Supervision
            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))  
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1))  
            
            
            
            #      
            # for student1
            mix_factors = np.random.beta(
                0.5, 0.5, size=(args.labeled_bs//2, 1, 1, 1))
            mix_factors = torch.tensor(
                mix_factors, dtype=torch.float).cuda()
            unlabeled_volume_batch_0 = uimg_a
            unlabeled_volume_batch_1 = uimg_b

            # Mix images
            batch_ux_mixed = unlabeled_volume_batch_0 * \
                (1.0 - mix_factors) + \
                unlabeled_volume_batch_1 * mix_factors
            input_volume_batch = torch.cat(
                [labeled_volume_batch, batch_ux_mixed], dim=0)
            outputs = model1(prepare_25d_input_for_model(input_volume_batch))
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(
                    ema_model(prepare_25d_input_for_model(unlabeled_volume_batch_0)), dim=1)
                ema_output_ux1 = torch.softmax(
                    ema_model(prepare_25d_input_for_model(unlabeled_volume_batch_1)), dim=1)
                batch_pred_mixed = ema_output_ux0 * \
                    (1.0 - mix_factors) + ema_output_ux1 * mix_factors
            
            loss_ce_ict = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:].long())
            loss_dice_ict = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            supervised_loss_ict = 0.5 * (loss_dice_ict + loss_ce_ict)
                
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            consistency_loss = torch.mean(
                (outputs_soft[args.labeled_bs:] - batch_pred_mixed) ** 2)*consistency_weight
            
            
            
            
            
            # FFT

            image_patch_last = process_image_batches_for_model(net_input_1, net_input_2,30)
            image_output_1 = model1(prepare_25d_input_for_model(image_patch_last))
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_output_2 = model2(prepare_25d_input_for_model(image_patch_last))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            # Model1 & Model2 Second Step Cross Pseudo Supervision
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4
            # loss = loss_1 + loss_2
            center_anchor_reg1 = center_anchor_regularization(model1)
            center_anchor_reg2 = center_anchor_regularization(model2)
            center_anchor_reg = center_anchor_reg1 + center_anchor_reg2
            loss = loss_1 + loss_2 + consistency_loss + supervised_loss_ict + center_anchor_reg

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()


            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
    

            update_model_ema(model1, ema_model, 0.99)
            


            if args.input_mode == '25d_3ch' and iter_num % 200 == 0:
                p1, c1, n1 = first_conv_channel_l1_fractions(model1)
                p2, c2, n2 = first_conv_channel_l1_fractions(model2)
                suppression_mask_ratio, suppression_input_delta = boundary_neighbor_suppression_stats(net_input_1)
                logging.info('iteration %d: center_anchor_reg: %f, model1_first_conv_l1_frac_prev_center_next: %.6f %.6f %.6f, model2_first_conv_l1_frac_prev_center_next: %.6f %.6f %.6f' %
                             (iter_num, center_anchor_reg.item(), p1, c1, n1, p2, c2, n2))
                logging.info('iteration %d: boundary_neighbor_suppression_enabled: %s, suppression_mask_ratio: %.6f, suppression_input_abs_delta: %.6f' %
                             (iter_num, str(args.boundary_neighbor_suppression), suppression_mask_ratio, suppression_input_delta))

            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f' % (iter_num, loss, loss_1, loss_2))
            
            # logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
             
            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(loader):
                    if args.input_mode == '25d_3ch':
                        metric_i = test_single_volume_promise_25d(sampled_batch["image"], sampled_batch["mask"], model1, classes=num_classes)
                    else:
                        metric_i = test_single_volume_promise(sampled_batch["image"], sampled_batch["mask"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                writer.add_scalar('info/model1_val_{}_dice'.format(1),metric_list[0], iter_num)
                performance1 = np.mean(metric_list, axis=0)
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model_1))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info('iteration %d : model1_mean_dice : %f' % (iter_num, performance1))
                metric_list = 0.0
# P-B PATCH:                 for i_batch, sampled_batch in enumerate(test_loader):
# P-B PATCH:                     metric_i = test_single_volume_promise(sampled_batch["image"], sampled_batch["mask"], model1, classes=num_classes)
# P-B PATCH:                     metric_list += np.array(metric_i)
# P-B PATCH:                 metric_list = metric_list / len(db_test)
# P-B PATCH:                 writer.add_scalar('info/model1_test_{}_dice'.format(1),metric_list[0], iter_num)
# P-B PATCH:                 performance1_test = np.mean(metric_list, axis=0)
# P-B PATCH:                 
# P-B PATCH:                 logging.info('iteration %d : model1_mean_test_dice : %f' % (iter_num, performance1_test))
                model1.train()

                # model2.eval()
                # metric_list = 0.0
                # for i_batch, sampled_batch in enumerate(loader):
                #     metric_i = test_single_volume_promise(sampled_batch["image"], sampled_batch["mask"], model2, classes=num_classes)
                #     metric_list += np.array(metric_i)
                # metric_list = metric_list / len(db_val)
                # writer.add_scalar('info/model2_val_{}_dice'.format(1),metric_list[0], iter_num)
                # performance2 = np.mean(metric_list, axis=0)
                # if performance2 > best_performance2:
                #     best_performance2 = performance2
                #     save_mode_path = os.path.join(snapshot_path,'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                #     save_best = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model_2))
                #     torch.save(model2.state_dict(), save_mode_path)
                #     torch.save(model2.state_dict(), save_best)
                # logging.info('iteration %d : model2_mean_dice : %f' % (iter_num, performance2))
                # model2.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.shape_audit_only:
        run_shape_audit_only()
        sys.exit(0)
    if args.phase_transition_audit_only:
        run_phase_transition_audit_only()
        sys.exit(0)

    pre_snapshot_path = "./beta_FFT/PROMISE12_bcp_{}_{}/pre_train".format(args.exp, args.labeled_num)
    snapshot_path = "./beta_FFT/PROMISE12_bcp_fft_test_seed_{}_{}/self_train".format(args.exp, args.labeled_num)
    if not os.path.exists(pre_snapshot_path):
        os.makedirs(pre_snapshot_path)

    



    # 设置日志记录
    # logging.basicConfig(filename=os.path.join(pre_snapshot_path, "log.txt"), level=logging.INFO)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path,exist_ok=True)

    shutil.copy('./train_promise12_bcp_fft.py', snapshot_path)
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)


    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args,pre_snapshot_path, snapshot_path)
