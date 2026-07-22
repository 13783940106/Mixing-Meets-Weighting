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
from dataloaders.promise12 import Promise12, color_jitter, cutout_gray
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
parser.add_argument('--use_mask_token_prehead_router', action='store_true',
                    help='enable decoder pre-head mask-token router')
parser.add_argument('--mask_token_num', type=int, default=4,
                    help='number of shared decoder pre-head mask tokens')
parser.add_argument('--mask_token_dim', type=int, default=16,
                    help='token projection dimension for decoder pre-head mask-token router')
parser.add_argument('--mask_token_log_interval', type=int, default=200,
                    help='iteration interval for mask-token router endpoint logging')
parser.add_argument('--use_m3hl_mmm_lite', action='store_true',
                    help='enable M3HL-inspired mutual multi-mask mix consistency')
parser.add_argument('--m3hl_mmm_weight', type=float, default=0.05,
                    help='weight for M3HL-MMM-Lite auxiliary consistency')
parser.add_argument('--m3hl_mmm_mask_ratio', type=float, default=0.50,
                    help='expected spatial mask ratio for M3HL-MMM-Lite')
parser.add_argument('--m3hl_mmm_patch_size', type=int, default=56,
                    help='block size for M3HL-MMM-Lite mask generation')
parser.add_argument('--m3hl_mmm_log_interval', type=int, default=200,
                    help='iteration interval for M3HL-MMM-Lite endpoint logging')
parser.add_argument('--use_dpl_msp_logit_lite', action='store_true',
                    help='enable DPL-MSP-inspired logit prototype consistency lite')
parser.add_argument('--dpl_msp_logit_lite_weight', type=float, default=0.01,
                    help='weight for DPL-MSP logit prototype lite auxiliary loss')
parser.add_argument('--dpl_msp_log_interval', type=int, default=200,
                    help='iteration interval for DPL-MSP logit prototype endpoint logging')
parser.add_argument('--use_prediction_disagreement_2d_adapt', action='store_true',
                    help='enable 2D peer prediction-disagreement consistency on high-disagreement unlabeled pixels')
parser.add_argument('--prediction_disagreement_2d_weight', type=float, default=0.05,
                    help='weight for 2D prediction disagreement auxiliary loss')
parser.add_argument('--prediction_disagreement_2d_top_quantile', type=float, default=0.50,
                    help='batch quantile threshold for high-disagreement pixel selection')
parser.add_argument('--prediction_disagreement_2d_log_interval', type=int, default=200,
                    help='iteration interval for 2D prediction disagreement endpoint logging')
parser.add_argument('--use_reliable_peer_cps_gate', action='store_true',
                    help='enable confidence/agreement reliability gate for peer CPS losses')
parser.add_argument('--reliable_peer_cps_conf_threshold', type=float, default=0.75,
                    help='confidence threshold for reliable peer CPS gate')
parser.add_argument('--reliable_peer_cps_agreement_threshold', type=float, default=0.50,
                    help='agreement threshold for reliable peer CPS gate')
parser.add_argument('--reliable_peer_cps_floor', type=float, default=0.25,
                    help='minimum multiplier for reliable peer CPS gate')
parser.add_argument('--reliable_peer_cps_log_interval', type=int, default=200,
                    help='iteration interval for reliable peer CPS gate endpoint logging')
parser.add_argument('--use_boundary_aware_pseudo_consistency', action='store_true',
                    help='enable boundary-aware peer pseudo-label consistency')
parser.add_argument('--boundary_aware_pseudo_weight', type=float, default=0.05,
                    help='weight for boundary-aware peer pseudo-label consistency')
parser.add_argument('--boundary_aware_pseudo_alpha', type=float, default=2.0,
                    help='extra CE weight multiplier on pseudo-label boundary pixels')
parser.add_argument('--boundary_aware_pseudo_kernel', type=int, default=3,
                    help='odd kernel size used to detect pseudo-label boundaries')
parser.add_argument('--boundary_aware_pseudo_log_interval', type=int, default=200,
                    help='iteration interval for boundary-aware pseudo consistency logging')
parser.add_argument('--use_quality_aware_ict_reweighting', action='store_true',
                    help='enable EMA confidence quality-aware reweighting for ICT consistency')
parser.add_argument('--quality_aware_ict_conf_threshold', type=float, default=0.70,
                    help='EMA confidence threshold for quality-aware ICT reweighting')
parser.add_argument('--quality_aware_ict_floor', type=float, default=0.25,
                    help='minimum pixel weight for quality-aware ICT reweighting')
parser.add_argument('--quality_aware_ict_log_interval', type=int, default=200,
                    help='iteration interval for quality-aware ICT reweighting logging')
parser.add_argument('--use_sdcl_discrepancy_correction_lite', action='store_true',
                    help='enable SDCL-inspired peer discrepancy correction lite')
parser.add_argument('--sdcl_discrepancy_weight', type=float, default=0.02,
                    help='weight for SDCL-inspired peer discrepancy correction lite')
parser.add_argument('--sdcl_discrepancy_conf_threshold', type=float, default=0.70,
                    help='agreement-region confidence threshold for SDCL discrepancy correction')
parser.add_argument('--sdcl_discrepancy_entropy_weight', type=float, default=0.25,
                    help='entropy-gap weight for peer disagreement regions')
parser.add_argument('--sdcl_discrepancy_log_interval', type=int, default=200,
                    help='iteration interval for SDCL discrepancy correction logging')
parser.add_argument('--use_shape_prior_area_lite', action='store_true',
                    help='enable lightweight foreground-area shape prior for unlabeled predictions')
parser.add_argument('--shape_prior_area_weight', type=float, default=0.01,
                    help='weight for lightweight foreground-area shape prior')
parser.add_argument('--shape_prior_area_min', type=float, default=0.001,
                    help='minimum allowed labeled-batch foreground ratio anchor')
parser.add_argument('--shape_prior_area_max', type=float, default=0.35,
                    help='maximum allowed labeled-batch foreground ratio anchor')
parser.add_argument('--shape_prior_area_log_interval', type=int, default=200,
                    help='iteration interval for shape-prior-area logging')
parser.add_argument('--use_uac_cps_delay3k_v3', action='store_true',
                    help='enable delayed uncertainty/agreement calibrated main CPS scaling')
parser.add_argument('--uac_cps_delay3k_min_weight', type=float, default=0.75,
                    help='minimum CPS multiplier for delayed UAC-CPS')
parser.add_argument('--uac_cps_delay3k_power', type=float, default=1.0,
                    help='agreement exponent for delayed UAC-CPS')
parser.add_argument('--uac_cps_delay3k_start_iter', type=int, default=3000,
                    help='iteration at which delayed UAC-CPS starts')
parser.add_argument('--uac_cps_delay3k_log_interval', type=int, default=200,
                    help='iteration interval for delayed UAC-CPS logging')
parser.add_argument('--use_strubar_ipba_lite', action='store_true',
                    help='enable StruBAR-inspired interior-preserving bidirectional alignment lite')
parser.add_argument('--strubar_ipba_lite_weight', type=float, default=0.01,
                    help='weight for StruBAR-IPBA lite auxiliary loss')
parser.add_argument('--strubar_ipba_log_interval', type=int, default=200,
                    help='iteration interval for StruBAR-IPBA lite endpoint logging')
parser.add_argument('--use_bdw_fft_power16_v2', action='store_true',
                    help='enable BDW FFT agreement-weighted FFT CPS loss')
parser.add_argument('--bdw_fft_power16_min_weight', type=float, default=0.50,
                    help='minimum FFT CPS multiplier for BDW FFT')
parser.add_argument('--bdw_fft_power16_power', type=float, default=1.0,
                    help='agreement exponent for BDW FFT')
parser.add_argument('--bdw_fft_power16_log_interval', type=int, default=200,
                    help='iteration interval for BDW FFT endpoint logging')
parser.add_argument('--use_fos_fft_v1', action='store_true',
                    help='enable FOS FFT original-to-frequency soft stabilization')
parser.add_argument('--fos_fft_v1_weight', type=float, default=0.05,
                    help='weight for FOS FFT auxiliary loss')
parser.add_argument('--fos_fft_v1_log_interval', type=int, default=200,
                    help='iteration interval for FOS FFT endpoint logging')
parser.add_argument('--input_mode', type=str, default='2d', choices=['2d', '25d_3ch'],
                    help='input mode: 2d keeps original behavior; 25d_3ch uses prev/center/next')
parser.add_argument('--prediction_disagreement_neighbor_regularization', action='store_true',
                    help='enable 2.5D center-repeat vs neighbor-view prediction disagreement regularization')
parser.add_argument('--prediction_disagreement_reg_weight', type=float, default=0.0,
                    help='weight for 2.5D prediction disagreement neighbor regularization')
parser.add_argument('--residual_harm_gated_prediction_disagreement_v2', action='store_true',
                    help='enable residual-harm gated component for 2.5D prediction disagreement')
parser.add_argument('--prediction_disagreement_neighbor_log_interval', type=int, default=200,
                    help='iteration interval for 2.5D prediction disagreement endpoint logging')
parser.add_argument('--pretrain_max_iterations_override', type=int, default=None,
                    help='engineering smoke/short override for pre-train iterations; default preserves original 10000')
parser.add_argument('--self_train_max_iterations_override', type=int, default=None,
                    help='engineering smoke/short override for self-train iterations; default uses max_iterations')
args = parser.parse_args()
config = get_config(args)
dice_loss = losses.DiceLoss(2)

def log_mask_token_router_stats(model, model_name, iter_num):
    decoder = getattr(model, "decoder", None)
    router = getattr(decoder, "mask_token_router", None)
    if router is None:
        logging.info(
            "iteration %d : %s_mask_token_prehead_router_enabled : False" %
            (iter_num, model_name)
        )
        return
    stats = getattr(router, "last_stats", {}) or {}
    logging.info(
        "iteration %d : %s_mask_token_prehead_router_enabled : True, "
        "token_utilization : %s, routing_entropy : %s, activation_coverage : %s, "
        "gamma : %s, residual_to_feature_ratio : %s, finite : %s" %
        (
            iter_num,
            model_name,
            stats.get("token_utilization", "NA"),
            stats.get("routing_entropy", "NA"),
            stats.get("activation_coverage", "NA"),
            stats.get("gamma", "NA"),
            stats.get("residual_to_feature_ratio", "NA"),
            stats.get("finite", "NA"),
        )
    )

def strubar_soft_boundary_from_fg_prob(fg_prob):
    dx = torch.abs(fg_prob[:, :, 1:, :] - fg_prob[:, :, :-1, :])
    dy = torch.abs(fg_prob[:, :, :, 1:] - fg_prob[:, :, :, :-1])
    dx = F.pad(dx, (0, 0, 0, 1))
    dy = F.pad(dy, (0, 1, 0, 0))
    return (dx + dy).clamp(0.0, 1.0)

def strubar_ipba_lite_loss(logits1, logits2, fft_logits1, fft_logits2):
    prob1 = torch.softmax(logits1, dim=1)
    prob2 = torch.softmax(logits2, dim=1)
    fft_prob1 = torch.softmax(fft_logits1, dim=1)
    fft_prob2 = torch.softmax(fft_logits2, dim=1)

    fg1 = prob1[:, 1:2]
    fg2 = prob2[:, 1:2]
    fft_fg1 = fft_prob1[:, 1:2]
    fft_fg2 = fft_prob2[:, 1:2]

    b1 = strubar_soft_boundary_from_fg_prob(fg1)
    b2 = strubar_soft_boundary_from_fg_prob(fg2)
    fb1 = strubar_soft_boundary_from_fg_prob(fft_fg1)
    fb2 = strubar_soft_boundary_from_fg_prob(fft_fg2)

    interior_weight = ((1.0 - b1) * (1.0 - b2)).detach().clamp(0.0, 1.0)
    fft_interior_weight = ((1.0 - fb1) * (1.0 - fb2)).detach().clamp(0.0, 1.0)

    region_loss = 0.5 * (
        (((prob1 - prob2.detach()) ** 2) * interior_weight).mean() +
        (((prob2 - prob1.detach()) ** 2) * interior_weight).mean()
    )
    fft_region_loss = 0.5 * (
        (((fft_prob1 - fft_prob2.detach()) ** 2) * fft_interior_weight).mean() +
        (((fft_prob2 - fft_prob1.detach()) ** 2) * fft_interior_weight).mean()
    )
    boundary_loss = 0.5 * (
        F.mse_loss(b1, b2.detach()) + F.mse_loss(b2, b1.detach())
    )
    fft_boundary_loss = 0.5 * (
        F.mse_loss(fb1, fb2.detach()) + F.mse_loss(fb2, fb1.detach())
    )

    loss = region_loss + fft_region_loss + 0.25 * (boundary_loss + fft_boundary_loss)
    with torch.no_grad():
        stats = {
            "region_loss": float(region_loss.detach().cpu()),
            "fft_region_loss": float(fft_region_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            "fft_boundary_loss": float(fft_boundary_loss.detach().cpu()),
            "interior_weight_mean": float(interior_weight.mean().detach().cpu()),
            "fft_interior_weight_mean": float(fft_interior_weight.mean().detach().cpu()),
            "boundary_mean_model1": float(b1.mean().detach().cpu()),
            "boundary_mean_model2": float(b2.mean().detach().cpu()),
            "fft_boundary_mean_model1": float(fb1.mean().detach().cpu()),
            "fft_boundary_mean_model2": float(fb2.mean().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_strubar_ipba_lite_stats(iter_num, loss_value, stats):
    logging.info(
        "iteration %d : strubar_ipba_lite_enabled : True, "
        "strubar_ipba_lite_loss : %s, region_loss : %s, fft_region_loss : %s, "
        "boundary_loss : %s, fft_boundary_loss : %s, interior_weight_mean : %s, "
        "fft_interior_weight_mean : %s, boundary_mean_model1 : %s, "
        "boundary_mean_model2 : %s, fft_boundary_mean_model1 : %s, "
        "fft_boundary_mean_model2 : %s, finite : %s" %
        (
            iter_num,
            loss_value,
            stats.get("region_loss", "NA"),
            stats.get("fft_region_loss", "NA"),
            stats.get("boundary_loss", "NA"),
            stats.get("fft_boundary_loss", "NA"),
            stats.get("interior_weight_mean", "NA"),
            stats.get("fft_interior_weight_mean", "NA"),
            stats.get("boundary_mean_model1", "NA"),
            stats.get("boundary_mean_model2", "NA"),
            stats.get("fft_boundary_mean_model1", "NA"),
            stats.get("fft_boundary_mean_model2", "NA"),
            stats.get("finite", "NA"),
        )
    )

def bdw_fft_power16_weight_from_predictions(pseudo_1, pseudo_2, min_weight, power):
    min_weight = min(max(float(min_weight), 0.0), 1.0)
    power = max(float(power), 0.0)
    disagreement = (pseudo_1.detach() != pseudo_2.detach()).float().mean()
    agreement = torch.clamp(1.0 - disagreement, min=0.0, max=1.0)
    weight = min_weight + (1.0 - min_weight) * torch.pow(agreement, power)
    return weight, disagreement.detach(), agreement.detach()

def log_bdw_fft_power16_stats(iter_num, weight, disagreement, agreement, loss):
    finite = (
        bool(torch.isfinite(weight).detach().cpu().item()) and
        bool(torch.isfinite(disagreement).detach().cpu().item()) and
        bool(torch.isfinite(agreement).detach().cpu().item()) and
        bool(torch.isfinite(loss).detach().cpu().item())
    )

def fos_fft_v1_loss(image_output_soft_1, image_output_soft_2, out_soft_1, out_soft_2):
    fos_batch_1 = min(image_output_soft_1.shape[0], out_soft_1.shape[0])
    fos_batch_2 = min(image_output_soft_2.shape[0], out_soft_2.shape[0])
    loss = 0.5 * (
        F.mse_loss(image_output_soft_1[:fos_batch_1], out_soft_1[:fos_batch_1].detach()) +
        F.mse_loss(image_output_soft_2[:fos_batch_2], out_soft_2[:fos_batch_2].detach())
    )
    return loss

def log_fos_fft_v1_stats(iter_num, loss_value, total_loss):
    finite = (
        bool(torch.isfinite(loss_value).detach().cpu().item()) and
        bool(torch.isfinite(total_loss).detach().cpu().item())
    )
    logging.info(
        "iteration %d : fos_fft_v1_enabled : True, "
        "fos_fft_v1_loss : %f, fos_fft_v1_finite : %s" %
        (iter_num, loss_value.detach().item(), str(finite))
    )
    logging.info(
        "iteration %d : bdw_fft_power16_v2_enabled : True, "
        "bdw_fft_weight : %f, bdw_fft_disagreement : %f, "
        "bdw_fft_agreement : %f, bdw_fft_finite : %s" %
        (
            iter_num,
            weight.detach().item(),
            disagreement.detach().item(),
            agreement.detach().item(),
            str(finite),
        )
    )

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch,args.consistency_rampup)  # args.consistency=0.1 # args.consistency_rampup=200
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
    """Prev/center/next triplets with center-slice supervision."""
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
        return np.stack(
            [self.X_train[prev_idx], self.X_train[idx], self.X_train[next_idx]],
            axis=0,
        ).astype(np.float32)

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
            img25 = np.stack(
                [ndimage.rotate(img25[c], angle, order=0, reshape=False) for c in range(3)],
                axis=0,
            )
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

def model_input_channels():
    return 3 if args.input_mode == '25d_3ch' else 1

def prepare_25d_input_for_model(batch, apply_neighbor_policy=True):
    return batch

def center_repeat_25d_view(batch):
    if args.input_mode != '25d_3ch' or batch.ndim != 4 or batch.shape[1] != 3:
        return batch
    center = batch[:, 1:2]
    return center.repeat(1, 3, 1, 1)

def prediction_disagreement_neighbor_regularization_loss(model, batch, neighbor_logits):
    zero = neighbor_logits.sum() * 0.0
    if (
        args.input_mode != '25d_3ch'
        or not args.prediction_disagreement_neighbor_regularization
        or args.prediction_disagreement_reg_weight <= 0
    ):
        return zero, {
            "enabled": False,
            "finite": True,
        }

    with torch.no_grad():
        center_logits = model(center_repeat_25d_view(batch))
        center_soft = torch.softmax(center_logits, dim=1).detach()
        center_pred = torch.argmax(center_soft, dim=1)
    neighbor_soft = torch.softmax(neighbor_logits, dim=1)
    neighbor_pred = torch.argmax(neighbor_soft.detach(), dim=1)
    diff = (neighbor_soft - center_soft) ** 2
    global_mse = torch.mean(diff)
    gated_mse = global_mse
    gate_ratio = 1.0
    if args.residual_harm_gated_prediction_disagreement_v2:
        detached_risk = diff.detach().mean(dim=1)
        threshold = detached_risk.flatten(1).mean(dim=1).view(-1, 1, 1)
        risk_gate = (detached_risk >= threshold).float()
        gate_ratio = float(risk_gate.mean().detach().cpu())
        gated_mse = (diff.mean(dim=1) * risk_gate).sum() / (risk_gate.sum() + 1e-6)
    mse = 0.5 * global_mse + 0.5 * gated_mse if args.residual_harm_gated_prediction_disagreement_v2 else global_mse
    loss = args.prediction_disagreement_reg_weight * mse
    disagreement = torch.mean((neighbor_pred != center_pred).float())
    stats = {
        "enabled": True,
        "loss": float(loss.detach().cpu()),
        "disagreement_ratio": float(disagreement.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "global_mse": float(global_mse.detach().cpu()),
        "gated_mse": float(gated_mse.detach().cpu()),
        "gate_ratio": gate_ratio,
        "finite": bool(torch.isfinite(loss).detach().cpu().item()),
    }
    return loss, stats

def log_prediction_disagreement_neighbor_stats(iter_num, stats_1, stats_2):
    logging.info(
        "iteration %d: prediction_disagreement_neighbor_regularization_enabled: %s, "
        "residual_harm_gated_prediction_disagreement_v2_enabled: %s, "
        "prediction_disagreement_reg_weight: %.6f, "
        "prediction_disagreement_reg_model1: %s, prediction_disagreement_reg_model2: %s, "
        "prediction_disagreement_ratio_model1: %s, prediction_disagreement_ratio_model2: %s, "
        "prediction_disagreement_global_mse_model1: %s, prediction_disagreement_global_mse_model2: %s, "
        "prediction_disagreement_gated_mse_model1: %s, prediction_disagreement_gated_mse_model2: %s, "
        "prediction_disagreement_gate_ratio_model1: %s, prediction_disagreement_gate_ratio_model2: %s, "
        "finite_model1: %s, finite_model2: %s" %
        (
            iter_num,
            str(args.prediction_disagreement_neighbor_regularization),
            str(args.residual_harm_gated_prediction_disagreement_v2),
            args.prediction_disagreement_reg_weight,
            stats_1.get("loss", "NA"),
            stats_2.get("loss", "NA"),
            stats_1.get("disagreement_ratio", "NA"),
            stats_2.get("disagreement_ratio", "NA"),
            stats_1.get("global_mse", "NA"),
            stats_2.get("global_mse", "NA"),
            stats_1.get("gated_mse", "NA"),
            stats_2.get("gated_mse", "NA"),
            stats_1.get("gate_ratio", "NA"),
            stats_2.get("gate_ratio", "NA"),
            stats_1.get("finite", "NA"),
            stats_2.get("finite", "NA"),
        )
    )

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

def test_single_volume_promise_for_model(image, label, net, classes):
    if args.input_mode != '25d_3ch':
        return test_single_volume_promise(image, label, net, classes=classes)
    image_np = image.squeeze(0).cpu().detach().numpy()
    label_np = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label_np)
    for ind in range(image_np.shape[0]):
        prev_ind = ind - 1 if ind > 0 else ind
        next_ind = ind + 1 if ind < image_np.shape[0] - 1 else ind
        triplet = np.stack(
            [image_np[prev_ind], image_np[ind], image_np[next_ind]],
            axis=0,
        ).astype(np.float32)
        input_tensor = torch.from_numpy(triplet).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(
                torch.softmax(net(prepare_25d_input_for_model(input_tensor, apply_neighbor_policy=False)), dim=1),
                dim=1,
            ).squeeze(0)
            prediction[ind] = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label_np == i))
    return metric_list

def generate_m3hl_mmm_mask(img, mask_ratio=0.5, patch_size=56):
    batch_size, _, img_x, img_y = img.shape
    grid_x = max(1, int(np.ceil(float(img_x) / float(patch_size))))
    grid_y = max(1, int(np.ceil(float(img_y) / float(patch_size))))
    low_res = (torch.rand(batch_size, 1, grid_x, grid_y, device=img.device) < mask_ratio).float()
    return F.interpolate(low_res, size=(img_x, img_y), mode='nearest')

def m3hl_mmm_lite_consistency(model, img_a, img_b, logits_a, logits_b, mask_ratio, patch_size):
    mask = generate_m3hl_mmm_mask(img_a, mask_ratio=mask_ratio, patch_size=patch_size)
    mixed = img_a * mask + img_b * (1.0 - mask)
    with torch.no_grad():
        soft_a = torch.softmax(logits_a, dim=1)
        soft_b = torch.softmax(logits_b, dim=1)
        target = soft_a * mask + soft_b * (1.0 - mask)
    pred = torch.softmax(model(mixed), dim=1)
    loss = F.mse_loss(pred, target.detach())
    finite = bool(torch.isfinite(loss).detach().cpu().item())
    return loss, mask.mean().detach(), finite

def _weighted_logit_prototype(logits, weights):
    weights = weights.float()
    denom = weights.sum(dim=(0, 2, 3)).clamp_min(1e-6)
    return (logits * weights).sum(dim=(0, 2, 3)) / denom

def _prototype_pair_loss(proto_a, proto_b):
    return 1.0 - F.cosine_similarity(proto_a.unsqueeze(0), proto_b.unsqueeze(0), dim=1).mean()

def _class_weight_from_labels_or_probs(labels, probs, class_id):
    gt_weight = (labels == class_id).float().unsqueeze(1)
    if gt_weight.sum().detach().item() > 1.0:
        return gt_weight
    return probs[:, class_id:class_id + 1].detach()

def _soft_class_weight(probs, class_id):
    return probs[:, class_id:class_id + 1].detach()

def dpl_msp_logit_lite_loss(out_1, out_2, fft_out_1, fft_out_2, ict_outputs, labels, labeled_bs):
    prob_1 = torch.softmax(out_1, dim=1)
    prob_2 = torch.softmax(out_2, dim=1)
    fft_prob_1 = torch.softmax(fft_out_1, dim=1)
    fft_prob_2 = torch.softmax(fft_out_2, dim=1)
    ict_prob = torch.softmax(ict_outputs[:labeled_bs], dim=1)
    ict_logits = ict_outputs[:labeled_bs]
    labeled_gt = labels[:labeled_bs]

    branch_losses = []
    fft_losses = []
    anchor_losses = []
    sep_losses = []
    stats = {}
    for class_id in range(args.num_classes):
        branch_proto_1 = _weighted_logit_prototype(out_1, _soft_class_weight(prob_1, class_id))
        branch_proto_2 = _weighted_logit_prototype(out_2, _soft_class_weight(prob_2, class_id))
        fft_proto_1 = _weighted_logit_prototype(fft_out_1, _soft_class_weight(fft_prob_1, class_id))
        fft_proto_2 = _weighted_logit_prototype(fft_out_2, _soft_class_weight(fft_prob_2, class_id))
        anchor_weight = _class_weight_from_labels_or_probs(labeled_gt, ict_prob, class_id)
        anchor_proto = _weighted_logit_prototype(ict_logits, anchor_weight)

        branch_losses.append(_prototype_pair_loss(branch_proto_1, branch_proto_2))
        fft_losses.append(_prototype_pair_loss(fft_proto_1, fft_proto_2))
        anchor_losses.extend([
            _prototype_pair_loss(branch_proto_1, anchor_proto),
            _prototype_pair_loss(branch_proto_2, anchor_proto),
            _prototype_pair_loss(fft_proto_1, anchor_proto),
            _prototype_pair_loss(fft_proto_2, anchor_proto),
        ])

        stats["class_%d_branch_proto_norm_model1" % class_id] = float(branch_proto_1.norm().detach().cpu())
        stats["class_%d_branch_proto_norm_model2" % class_id] = float(branch_proto_2.norm().detach().cpu())
        stats["class_%d_fft_proto_norm_model1" % class_id] = float(fft_proto_1.norm().detach().cpu())
        stats["class_%d_fft_proto_norm_model2" % class_id] = float(fft_proto_2.norm().detach().cpu())
        stats["class_%d_anchor_proto_norm" % class_id] = float(anchor_proto.norm().detach().cpu())

    if args.num_classes >= 2:
        p0_1 = _weighted_logit_prototype(out_1, _soft_class_weight(prob_1, 0))
        p1_1 = _weighted_logit_prototype(out_1, _soft_class_weight(prob_1, 1))
        p0_2 = _weighted_logit_prototype(out_2, _soft_class_weight(prob_2, 0))
        p1_2 = _weighted_logit_prototype(out_2, _soft_class_weight(prob_2, 1))
        sep_losses.append(F.relu(F.cosine_similarity(p0_1.unsqueeze(0), p1_1.unsqueeze(0), dim=1).mean() - 0.1))
        sep_losses.append(F.relu(F.cosine_similarity(p0_2.unsqueeze(0), p1_2.unsqueeze(0), dim=1).mean() - 0.1))

    branch_loss = torch.stack(branch_losses).mean()
    fft_loss = torch.stack(fft_losses).mean()
    anchor_loss = torch.stack(anchor_losses).mean()
    sep_loss = torch.stack(sep_losses).mean() if sep_losses else out_1.new_tensor(0.0)
    loss = branch_loss + fft_loss + 0.5 * anchor_loss + 0.1 * sep_loss
    with torch.no_grad():
        stats.update({
            "branch_proto_loss": float(branch_loss.detach().cpu()),
            "fft_proto_loss": float(fft_loss.detach().cpu()),
            "anchor_proto_loss": float(anchor_loss.detach().cpu()),
            "class_sep_loss": float(sep_loss.detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu()),
        })
    return loss, stats

def log_dpl_msp_logit_lite_stats(iter_num, loss_value, stats):
    logging.info(
        "iteration %d : dpl_msp_logit_lite_enabled : True, "
        "dpl_msp_logit_lite_loss : %s, branch_proto_loss : %s, "
        "fft_proto_loss : %s, anchor_proto_loss : %s, class_sep_loss : %s, "
        "fg_branch_proto_norm_model1 : %s, fg_branch_proto_norm_model2 : %s, "
        "fg_fft_proto_norm_model1 : %s, fg_fft_proto_norm_model2 : %s, "
        "fg_anchor_proto_norm : %s, finite : %s" %
        (
            iter_num,
            loss_value,
            stats.get("branch_proto_loss", "NA"),
            stats.get("fft_proto_loss", "NA"),
            stats.get("anchor_proto_loss", "NA"),
            stats.get("class_sep_loss", "NA"),
            stats.get("class_1_branch_proto_norm_model1", "NA"),
            stats.get("class_1_branch_proto_norm_model2", "NA"),
            stats.get("class_1_fft_proto_norm_model1", "NA"),
            stats.get("class_1_fft_proto_norm_model2", "NA"),
            stats.get("class_1_anchor_proto_norm", "NA"),
            stats.get("finite", "NA"),
        )
    )

def prediction_disagreement_2d_adapt_loss(out_unl_1, out_unl_2, top_quantile):
    prob_1 = torch.softmax(out_unl_1, dim=1)
    prob_2 = torch.softmax(out_unl_2, dim=1)
    disagreement = torch.mean(torch.abs(prob_1.detach() - prob_2.detach()), dim=1, keepdim=True)
    flat = disagreement.reshape(disagreement.shape[0], -1)
    quantile = min(max(float(top_quantile), 0.0), 0.99)
    threshold = torch.quantile(flat, quantile, dim=1, keepdim=True).view(-1, 1, 1, 1)
    mask = (disagreement >= threshold).float()
    selected = mask.sum().clamp_min(1.0)
    loss = (((prob_1 - prob_2.detach()) ** 2 + (prob_2 - prob_1.detach()) ** 2) * mask).sum() / selected
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "mask_ratio": float(mask.mean().detach().cpu()),
            "mean_disagreement": float(disagreement.mean().detach().cpu()),
            "threshold_mean": float(threshold.mean().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu()),
        }
    return loss, stats

def log_prediction_disagreement_2d_stats(iter_num, stats):
    logging.info(
        "iteration %d : prediction_disagreement_2d_adapt_enabled : True, "
        "prediction_disagreement_2d_loss : %s, mask_ratio : %s, "
        "mean_disagreement : %s, threshold_mean : %s, finite : %s" %
        (
            iter_num,
            stats.get("loss", "NA"),
            stats.get("mask_ratio", "NA"),
            stats.get("mean_disagreement", "NA"),
            stats.get("threshold_mean", "NA"),
            stats.get("finite", "NA"),
        )
    )

def reliable_peer_cps_gate(
    soft_1,
    pseudo_1,
    soft_2,
    pseudo_2,
    conf_threshold,
    agreement_threshold,
    floor,
    prefix,
):
    with torch.no_grad():
        floor = min(max(float(floor), 0.0), 1.0)
        conf_threshold = min(max(float(conf_threshold), 0.0), 0.99)
        agreement_threshold = min(max(float(agreement_threshold), 0.0), 0.99)
        conf_1 = torch.max(soft_1.detach(), dim=1)[0].mean()
        conf_2 = torch.max(soft_2.detach(), dim=1)[0].mean()
        agreement = (pseudo_1.detach() == pseudo_2.detach()).float().mean()
        conf_score_1 = ((conf_1 - conf_threshold) / (1.0 - conf_threshold)).clamp(0.0, 1.0)
        conf_score_2 = ((conf_2 - conf_threshold) / (1.0 - conf_threshold)).clamp(0.0, 1.0)
        agreement_score = ((agreement - agreement_threshold) / (1.0 - agreement_threshold)).clamp(0.0, 1.0)
        gate_from_1 = floor + (1.0 - floor) * (0.5 * conf_score_1 + 0.5 * agreement_score)
        gate_from_2 = floor + (1.0 - floor) * (0.5 * conf_score_2 + 0.5 * agreement_score)
        stats = {
            prefix + "_gate_from_model1": float(gate_from_1.detach().cpu()),
            prefix + "_gate_from_model2": float(gate_from_2.detach().cpu()),
            prefix + "_conf_model1": float(conf_1.detach().cpu()),
            prefix + "_conf_model2": float(conf_2.detach().cpu()),
            prefix + "_agreement": float(agreement.detach().cpu()),
            prefix + "_finite": bool(
                torch.isfinite(gate_from_1).all().detach().cpu()
                and torch.isfinite(gate_from_2).all().detach().cpu()
            ),
        }
    return gate_from_1, gate_from_2, stats

def log_reliable_peer_cps_gate_stats(iter_num, stats):
    logging.info(
        "iteration %d : reliable_peer_cps_gate_enabled : True, "
        "main_gate_from_model1 : %s, main_gate_from_model2 : %s, "
        "main_conf_model1 : %s, main_conf_model2 : %s, main_agreement : %s, "
        "fft_gate_from_model1 : %s, fft_gate_from_model2 : %s, "
        "fft_conf_model1 : %s, fft_conf_model2 : %s, fft_agreement : %s, "
        "main_finite : %s, fft_finite : %s" %
        (
            iter_num,
            stats.get("main_gate_from_model1", "NA"),
            stats.get("main_gate_from_model2", "NA"),
            stats.get("main_conf_model1", "NA"),
            stats.get("main_conf_model2", "NA"),
            stats.get("main_agreement", "NA"),
            stats.get("fft_gate_from_model1", "NA"),
            stats.get("fft_gate_from_model2", "NA"),
            stats.get("fft_conf_model1", "NA"),
            stats.get("fft_conf_model2", "NA"),
            stats.get("fft_agreement", "NA"),
            stats.get("main_finite", "NA"),
            stats.get("fft_finite", "NA"),
        )
    )

def pseudo_label_boundary_map(pseudo_label, kernel_size):
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    pseudo = pseudo_label.float().unsqueeze(1)
    pad = kernel_size // 2
    local_max = F.max_pool2d(pseudo, kernel_size=kernel_size, stride=1, padding=pad)
    local_min = -F.max_pool2d(-pseudo, kernel_size=kernel_size, stride=1, padding=pad)
    return (local_max - local_min).clamp(0.0, 1.0).detach()

def boundary_aware_pseudo_consistency_loss(
    logits_1,
    pseudo_1,
    logits_2,
    pseudo_2,
    alpha,
    kernel_size,
    prefix,
):
    ce = nn.CrossEntropyLoss(reduction='none')
    boundary_1 = pseudo_label_boundary_map(pseudo_1.detach(), kernel_size).squeeze(1)
    boundary_2 = pseudo_label_boundary_map(pseudo_2.detach(), kernel_size).squeeze(1)
    weight_1 = 1.0 + float(alpha) * boundary_1
    weight_2 = 1.0 + float(alpha) * boundary_2
    loss_1 = (ce(logits_1, pseudo_2.detach().long()) * weight_2).sum() / weight_2.sum().clamp_min(1.0)
    loss_2 = (ce(logits_2, pseudo_1.detach().long()) * weight_1).sum() / weight_1.sum().clamp_min(1.0)
    loss = 0.5 * (loss_1 + loss_2)
    with torch.no_grad():
        stats = {
            prefix + "_loss": float(loss.detach().cpu()),
            prefix + "_boundary_ratio_model1": float(boundary_1.mean().detach().cpu()),
            prefix + "_boundary_ratio_model2": float(boundary_2.mean().detach().cpu()),
            prefix + "_weight_mean_model1": float(weight_1.mean().detach().cpu()),
            prefix + "_weight_mean_model2": float(weight_2.mean().detach().cpu()),
            prefix + "_finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_boundary_aware_pseudo_stats(iter_num, main_stats, fft_stats, total_loss):
    logging.info(
        "iteration %d : boundary_aware_pseudo_consistency_enabled : True, "
        "boundary_aware_pseudo_total_loss : %s, "
        "main_loss : %s, fft_loss : %s, "
        "main_boundary_ratio_model1 : %s, main_boundary_ratio_model2 : %s, "
        "fft_boundary_ratio_model1 : %s, fft_boundary_ratio_model2 : %s, "
        "main_finite : %s, fft_finite : %s, total_finite : %s" %
        (
            iter_num,
            float(total_loss.detach().cpu()),
            main_stats.get("main_loss", "NA"),
            fft_stats.get("fft_loss", "NA"),
            main_stats.get("main_boundary_ratio_model1", "NA"),
            main_stats.get("main_boundary_ratio_model2", "NA"),
            fft_stats.get("fft_boundary_ratio_model1", "NA"),
            fft_stats.get("fft_boundary_ratio_model2", "NA"),
            main_stats.get("main_finite", "NA"),
            fft_stats.get("fft_finite", "NA"),
            bool(torch.isfinite(total_loss).all().detach().cpu().item()),
        )
    )

def quality_aware_ict_consistency_loss(outputs_soft_unlabeled, target_soft, consistency_weight, conf_threshold, floor):
    floor = min(max(float(floor), 0.0), 1.0)
    conf_threshold = min(max(float(conf_threshold), 0.0), 0.99)
    diff = torch.mean((outputs_soft_unlabeled - target_soft.detach()) ** 2, dim=1)
    with torch.no_grad():
        confidence = torch.max(target_soft.detach(), dim=1)[0]
        quality = ((confidence - conf_threshold) / (1.0 - conf_threshold)).clamp(0.0, 1.0)
        weight = floor + (1.0 - floor) * quality
    loss = (diff * weight).sum() / weight.sum().clamp_min(1.0)
    loss = loss * consistency_weight
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "weight_mean": float(weight.mean().detach().cpu()),
            "weight_min": float(weight.min().detach().cpu()),
            "weight_max": float(weight.max().detach().cpu()),
            "confidence_mean": float(confidence.mean().detach().cpu()),
            "confidence_min": float(confidence.min().detach().cpu()),
            "confidence_max": float(confidence.max().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_quality_aware_ict_stats(iter_num, stats):
    logging.info(
        "iteration %d : quality_aware_ict_reweighting_enabled : True, "
        "quality_aware_ict_loss : %s, weight_mean : %s, weight_min : %s, weight_max : %s, "
        "confidence_mean : %s, confidence_min : %s, confidence_max : %s, finite : %s" %
        (
            iter_num,
            stats.get("loss", "NA"),
            stats.get("weight_mean", "NA"),
            stats.get("weight_min", "NA"),
            stats.get("weight_max", "NA"),
            stats.get("confidence_mean", "NA"),
            stats.get("confidence_min", "NA"),
            stats.get("confidence_max", "NA"),
            stats.get("finite", "NA"),
        )
    )

def sdcl_discrepancy_correction_lite_loss(
    logits_1,
    soft_1,
    logits_2,
    soft_2,
    conf_threshold,
    entropy_weight,
    prefix,
):
    ce = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        conf_threshold = min(max(float(conf_threshold), 0.0), 0.99)
        entropy_weight = max(float(entropy_weight), 0.0)
        conf_1, pseudo_1 = torch.max(soft_1.detach(), dim=1)
        conf_2, pseudo_2 = torch.max(soft_2.detach(), dim=1)
        agree = pseudo_1 == pseudo_2
        avg_conf = 0.5 * (conf_1 + conf_2)
        review_mask = (agree & (avg_conf >= conf_threshold)).float()
        disagreement_mask = (~agree).float()
        consensus = torch.where(conf_1 >= conf_2, pseudo_1, pseudo_2).long()

    review_loss_1 = (ce(logits_1, consensus) * review_mask).sum() / review_mask.sum().clamp_min(1.0)
    review_loss_2 = (ce(logits_2, consensus) * review_mask).sum() / review_mask.sum().clamp_min(1.0)
    review_loss = 0.5 * (review_loss_1 + review_loss_2)

    max_entropy = float(np.log(max(int(logits_1.shape[1]), 2)))
    entropy_1 = -(soft_1 * torch.log(soft_1.clamp_min(1e-6))).sum(dim=1)
    entropy_2 = -(soft_2 * torch.log(soft_2.clamp_min(1e-6))).sum(dim=1)
    entropy_gap_1 = (max_entropy - entropy_1).clamp_min(0.0)
    entropy_gap_2 = (max_entropy - entropy_2).clamp_min(0.0)
    entropy_gap_loss = (
        (entropy_gap_1 + entropy_gap_2) * disagreement_mask
    ).sum() / (2.0 * disagreement_mask.sum().clamp_min(1.0))

    loss = review_loss + entropy_weight * entropy_gap_loss
    with torch.no_grad():
        stats = {
            prefix + "_loss": float(loss.detach().cpu()),
            prefix + "_review_loss": float(review_loss.detach().cpu()),
            prefix + "_entropy_gap_loss": float(entropy_gap_loss.detach().cpu()),
            prefix + "_agreement_ratio": float(agree.float().mean().detach().cpu()),
            prefix + "_review_ratio": float(review_mask.mean().detach().cpu()),
            prefix + "_disagreement_ratio": float(disagreement_mask.mean().detach().cpu()),
            prefix + "_avg_conf": float(avg_conf.mean().detach().cpu()),
            prefix + "_finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_sdcl_discrepancy_correction_stats(iter_num, main_stats, fft_stats, total_loss):
    logging.info(
        "iteration %d : sdcl_discrepancy_correction_lite_enabled : True, "
        "sdcl_discrepancy_total_loss : %s, "
        "main_loss : %s, main_review_loss : %s, main_entropy_gap_loss : %s, "
        "main_agreement_ratio : %s, main_review_ratio : %s, main_disagreement_ratio : %s, main_avg_conf : %s, "
        "fft_loss : %s, fft_review_loss : %s, fft_entropy_gap_loss : %s, "
        "fft_agreement_ratio : %s, fft_review_ratio : %s, fft_disagreement_ratio : %s, fft_avg_conf : %s, "
        "main_finite : %s, fft_finite : %s, total_finite : %s" %
        (
            iter_num,
            float(total_loss.detach().cpu()),
            main_stats.get("main_loss", "NA"),
            main_stats.get("main_review_loss", "NA"),
            main_stats.get("main_entropy_gap_loss", "NA"),
            main_stats.get("main_agreement_ratio", "NA"),
            main_stats.get("main_review_ratio", "NA"),
            main_stats.get("main_disagreement_ratio", "NA"),
            main_stats.get("main_avg_conf", "NA"),
            fft_stats.get("fft_loss", "NA"),
            fft_stats.get("fft_review_loss", "NA"),
            fft_stats.get("fft_entropy_gap_loss", "NA"),
            fft_stats.get("fft_agreement_ratio", "NA"),
            fft_stats.get("fft_review_ratio", "NA"),
            fft_stats.get("fft_disagreement_ratio", "NA"),
            fft_stats.get("fft_avg_conf", "NA"),
            main_stats.get("main_finite", "NA"),
            fft_stats.get("fft_finite", "NA"),
            bool(torch.isfinite(total_loss).all().detach().cpu().item()),
        )
    )

def shape_prior_area_lite_loss(out_unl_1, out_unl_2, label_batch, labeled_bs, min_area, max_area):
    min_area = min(max(float(min_area), 0.0), 1.0)
    max_area = min(max(float(max_area), min_area), 1.0)
    labeled_fg = (label_batch[:labeled_bs].float() > 0).float()
    with torch.no_grad():
        target_area = labeled_fg.mean(dim=(1, 2)).mean().clamp(min_area, max_area)
    prob_fg_1 = torch.softmax(out_unl_1, dim=1)[:, 1]
    prob_fg_2 = torch.softmax(out_unl_2, dim=1)[:, 1]
    pred_area_1 = prob_fg_1.mean(dim=(1, 2))
    pred_area_2 = prob_fg_2.mean(dim=(1, 2))
    loss_1 = F.smooth_l1_loss(pred_area_1, target_area.expand_as(pred_area_1))
    loss_2 = F.smooth_l1_loss(pred_area_2, target_area.expand_as(pred_area_2))
    loss = 0.5 * (loss_1 + loss_2)
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "target_area": float(target_area.detach().cpu()),
            "pred_area_model1": float(pred_area_1.mean().detach().cpu()),
            "pred_area_model2": float(pred_area_2.mean().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_shape_prior_area_lite_stats(iter_num, stats):
    logging.info(
        "iteration %d : shape_prior_area_lite_enabled : True, "
        "shape_prior_area_loss : %s, target_area : %s, "
        "pred_area_model1 : %s, pred_area_model2 : %s, finite : %s" %
        (
            iter_num,
            stats.get("loss", "NA"),
            stats.get("target_area", "NA"),
            stats.get("pred_area_model1", "NA"),
            stats.get("pred_area_model2", "NA"),
            stats.get("finite", "NA"),
        )
    )

def uac_cps_delay3k_weight_from_predictions(pseudo_1, pseudo_2, min_weight, power):
    min_weight = min(max(float(min_weight), 0.0), 1.0)
    power = max(float(power), 0.0)
    disagreement = (pseudo_1.detach() != pseudo_2.detach()).float().mean()
    agreement = torch.clamp(1.0 - disagreement, min=0.0, max=1.0)
    weight = min_weight + (1.0 - min_weight) * torch.pow(agreement, power)
    return weight, disagreement.detach(), agreement.detach()

def log_uac_cps_delay3k_stats(iter_num, active, weight, disagreement, agreement, loss):
    finite = (
        bool(torch.isfinite(weight).detach().cpu().item()) and
        bool(torch.isfinite(loss).detach().cpu().item())
    )
    logging.info(
        "iteration %d : uac_cps_delay3k_v3_enabled : True, "
        "uac_cps_delay3k_active : %s, uac_cps_delay3k_start_iter : %d, "
        "uac_cps_delay3k_weight : %f, uac_cps_delay3k_disagreement : %f, "
        "uac_cps_delay3k_agreement : %f, uac_cps_delay3k_finite : %s" %
        (
            iter_num,
            str(active),
            args.uac_cps_delay3k_start_iter,
            weight.detach().item(),
            disagreement.detach().item(),
            agreement.detach().item(),
            str(finite),
        )
    )

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
            out_mixl = model(prepare_25d_input_for_model(net_input, apply_neighbor_policy=False))
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))
            if args.use_mask_token_prehead_router and iter_num % args.mask_token_log_interval == 0:
                log_mask_token_router_stats(model, "pretrain_model", iter_num)



            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_promise_for_model(sampled_batch["image"], sampled_batch["mask"], model,
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
    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model_1))
    if not os.path.exists(save_best_path):
        save_net_opt(model, optimizer, save_best_path)
        logging.info('pretrain fallback saved final checkpoint for short/smoke: %s' % save_best_path)
    writer.close()

def create_model(ema=False):
    # Network definition
    model = net_factory(
        net_type=args.model_1,
        in_chns=model_input_channels(),
        class_num=2,
        use_mask_token_prehead_router=args.use_mask_token_prehead_router,
        mask_token_num=args.mask_token_num,
        mask_token_dim=args.mask_token_dim,
    )
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
    ema_model = create_model()

    # model2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()
    # model2.load_from(config)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Promise12_25D(args.root_path, mode='train', out_size=224) if args.input_mode == '25d_3ch' else Promise12(args.root_path, mode='train', out_size=224)
    db_val = Promise12(args.root_path, mode='val', out_size=224)
    db_test = Promise12(args.root_path, mode='test', out_size=224)

    total_slices = len(db_train)
    labeled_slice = 202  # args.labeled_num=7
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)  # args.labeled_bs=8
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    
    
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

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
                pre_a = ema_model(prepare_25d_input_for_model(uimg_a, apply_neighbor_policy=True))
                pre_b = ema_model(prepare_25d_input_for_model(uimg_b, apply_neighbor_policy=True))
                pre_a_s = ema_model(prepare_25d_input_for_model(uimg_a_s, apply_neighbor_policy=True))
                pre_b_s = ema_model(prepare_25d_input_for_model(uimg_b_s, apply_neighbor_policy=True))
                
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
            out_unl_1 = model1(prepare_25d_input_for_model(net_input_unl_1, apply_neighbor_policy=True))
            prediction_disagreement_neighbor_loss_1, prediction_disagreement_neighbor_stats_1 = prediction_disagreement_neighbor_regularization_loss(
                model1, net_input_unl_1, out_unl_1
            )
            out_l_1 = model1(prepare_25d_input_for_model(net_input_l_1, apply_neighbor_policy=False))
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            out_soft_1 = torch.softmax(out_1, dim=1)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2 = model2(prepare_25d_input_for_model(net_input_unl_2, apply_neighbor_policy=True))
            prediction_disagreement_neighbor_loss_2, prediction_disagreement_neighbor_stats_2 = prediction_disagreement_neighbor_regularization_loss(
                model2, net_input_unl_2, out_unl_2
            )
            out_l_2 = model2(prepare_25d_input_for_model(net_input_l_2, apply_neighbor_policy=False))
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
            uac_cps_delay3k_weight = out_1.new_tensor(1.0)
            uac_cps_delay3k_disagreement = out_1.new_tensor(0.0)
            uac_cps_delay3k_agreement = out_1.new_tensor(1.0)
            if args.use_uac_cps_delay3k_v3 and iter_num >= args.uac_cps_delay3k_start_iter:
                (
                    uac_cps_delay3k_weight,
                    uac_cps_delay3k_disagreement,
                    uac_cps_delay3k_agreement,
                ) = uac_cps_delay3k_weight_from_predictions(
                    out_pseudo_1,
                    out_pseudo_2,
                    args.uac_cps_delay3k_min_weight,
                    args.uac_cps_delay3k_power,
                )
                pseudo_supervision1 = pseudo_supervision1 * uac_cps_delay3k_weight
                pseudo_supervision2 = pseudo_supervision2 * uac_cps_delay3k_weight
            boundary_aware_main_loss = out_1.new_tensor(0.0)
            boundary_aware_main_stats = {}
            sdcl_main_loss = out_1.new_tensor(0.0)
            sdcl_main_stats = {}
            reliable_peer_cps_stats = {}
            if args.use_reliable_peer_cps_gate:
                main_gate_from_model1, main_gate_from_model2, reliable_peer_cps_stats = reliable_peer_cps_gate(
                    out_soft_1,
                    out_pseudo_1,
                    out_soft_2,
                    out_pseudo_2,
                    args.reliable_peer_cps_conf_threshold,
                    args.reliable_peer_cps_agreement_threshold,
                    args.reliable_peer_cps_floor,
                    "main",
                )
                pseudo_supervision1 = pseudo_supervision1 * main_gate_from_model2
                pseudo_supervision2 = pseudo_supervision2 * main_gate_from_model1
            if args.use_boundary_aware_pseudo_consistency:
                boundary_aware_main_loss, boundary_aware_main_stats = boundary_aware_pseudo_consistency_loss(
                    out_1,
                    out_pseudo_1,
                    out_2,
                    out_pseudo_2,
                    args.boundary_aware_pseudo_alpha,
                    args.boundary_aware_pseudo_kernel,
                    "main",
                )
            if args.use_sdcl_discrepancy_correction_lite:
                sdcl_main_loss, sdcl_main_stats = sdcl_discrepancy_correction_lite_loss(
                    out_1,
                    out_soft_1,
                    out_2,
                    out_soft_2,
                    args.sdcl_discrepancy_conf_threshold,
                    args.sdcl_discrepancy_entropy_weight,
                    "main",
                )
            
            
            
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
            outputs = model1(prepare_25d_input_for_model(input_volume_batch, apply_neighbor_policy=False))
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(
                    ema_model(prepare_25d_input_for_model(unlabeled_volume_batch_0, apply_neighbor_policy=True)), dim=1)
                ema_output_ux1 = torch.softmax(
                    ema_model(prepare_25d_input_for_model(unlabeled_volume_batch_1, apply_neighbor_policy=True)), dim=1)
                batch_pred_mixed = ema_output_ux0 * \
                    (1.0 - mix_factors) + ema_output_ux1 * mix_factors
            
            loss_ce_ict = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:].long())
            loss_dice_ict = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            supervised_loss_ict = 0.5 * (loss_dice_ict + loss_ce_ict)
                
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            quality_aware_ict_stats = {}
            if args.use_quality_aware_ict_reweighting:
                consistency_loss, quality_aware_ict_stats = quality_aware_ict_consistency_loss(
                    outputs_soft[args.labeled_bs:],
                    batch_pred_mixed,
                    consistency_weight,
                    args.quality_aware_ict_conf_threshold,
                    args.quality_aware_ict_floor,
                )
            else:
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled_bs:] - batch_pred_mixed) ** 2)*consistency_weight
            
            
            
            
            
            # FFT

            image_patch_last = process_image_batches_for_model(net_input_1, net_input_2,30)
            image_output_1 = model1(prepare_25d_input_for_model(image_patch_last, apply_neighbor_policy=True))
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_output_2 = model2(prepare_25d_input_for_model(image_patch_last, apply_neighbor_policy=True))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            # Model1 & Model2 Second Step Cross Pseudo Supervision
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))
            bdw_fft_power16_weight = image_output_1.new_tensor(1.0)
            bdw_fft_power16_disagreement = image_output_1.new_tensor(0.0)
            bdw_fft_power16_agreement = image_output_1.new_tensor(1.0)
            if args.use_bdw_fft_power16_v2:
                (
                    bdw_fft_power16_weight,
                    bdw_fft_power16_disagreement,
                    bdw_fft_power16_agreement,
                ) = bdw_fft_power16_weight_from_predictions(
                    pseudo_image_output_1,
                    pseudo_image_output_2,
                    args.bdw_fft_power16_min_weight,
                    args.bdw_fft_power16_power,
                )
                pseudo_supervision3 = pseudo_supervision3 * bdw_fft_power16_weight
                pseudo_supervision4 = pseudo_supervision4 * bdw_fft_power16_weight
            boundary_aware_fft_loss = image_output_1.new_tensor(0.0)
            boundary_aware_fft_stats = {}
            sdcl_fft_loss = image_output_1.new_tensor(0.0)
            sdcl_fft_stats = {}
            fos_fft_loss_value = image_output_1.new_tensor(0.0)
            if args.use_fos_fft_v1:
                fos_fft_loss_value = fos_fft_v1_loss(
                    image_output_soft_1,
                    image_output_soft_2,
                    out_soft_1,
                    out_soft_2,
                )
            if args.use_reliable_peer_cps_gate:
                fft_gate_from_model1, fft_gate_from_model2, fft_stats = reliable_peer_cps_gate(
                    image_output_soft_1,
                    pseudo_image_output_1,
                    image_output_soft_2,
                    pseudo_image_output_2,
                    args.reliable_peer_cps_conf_threshold,
                    args.reliable_peer_cps_agreement_threshold,
                    args.reliable_peer_cps_floor,
                    "fft",
                )
                pseudo_supervision3 = pseudo_supervision3 * fft_gate_from_model2
                pseudo_supervision4 = pseudo_supervision4 * fft_gate_from_model1
                reliable_peer_cps_stats.update(fft_stats)
            if args.use_boundary_aware_pseudo_consistency:
                boundary_aware_fft_loss, boundary_aware_fft_stats = boundary_aware_pseudo_consistency_loss(
                    image_output_1,
                    pseudo_image_output_1,
                    image_output_2,
                    pseudo_image_output_2,
                    args.boundary_aware_pseudo_alpha,
                    args.boundary_aware_pseudo_kernel,
                    "fft",
                )
            if args.use_sdcl_discrepancy_correction_lite:
                sdcl_fft_loss, sdcl_fft_stats = sdcl_discrepancy_correction_lite_loss(
                    image_output_1,
                    image_output_soft_1,
                    image_output_2,
                    image_output_soft_2,
                    args.sdcl_discrepancy_conf_threshold,
                    args.sdcl_discrepancy_entropy_weight,
                    "fft",
                )

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4
            # loss = loss_1 + loss_2
            loss = loss_1 + loss_2 + consistency_loss +supervised_loss_ict
            boundary_aware_pseudo_total_loss = boundary_aware_main_loss + boundary_aware_fft_loss
            if args.use_boundary_aware_pseudo_consistency:
                loss = loss + args.boundary_aware_pseudo_weight * boundary_aware_pseudo_total_loss
            sdcl_discrepancy_total_loss = sdcl_main_loss + sdcl_fft_loss
            if args.use_sdcl_discrepancy_correction_lite:
                loss = loss + args.sdcl_discrepancy_weight * sdcl_discrepancy_total_loss
            if args.use_fos_fft_v1:
                loss = loss + args.fos_fft_v1_weight * fos_fft_loss_value
            strubar_ipba_loss = loss.new_tensor(0.0)
            strubar_ipba_stats = {}
            if args.use_strubar_ipba_lite:
                strubar_ipba_loss, strubar_ipba_stats = strubar_ipba_lite_loss(
                    out_1,
                    out_2,
                    image_output_1,
                    image_output_2,
                )
                loss = loss + args.strubar_ipba_lite_weight * strubar_ipba_loss
            if args.prediction_disagreement_neighbor_regularization:
                loss = loss + prediction_disagreement_neighbor_loss_1 + prediction_disagreement_neighbor_loss_2

            shape_prior_area_loss = loss.new_tensor(0.0)
            shape_prior_area_stats = {}
            if args.use_shape_prior_area_lite:
                shape_prior_area_loss, shape_prior_area_stats = shape_prior_area_lite_loss(
                    out_unl_1,
                    out_unl_2,
                    label_batch,
                    args.labeled_bs,
                    args.shape_prior_area_min,
                    args.shape_prior_area_max,
                )
                loss = loss + args.shape_prior_area_weight * shape_prior_area_loss

            if args.use_uac_cps_delay3k_v3 and iter_num > 0 and iter_num % args.uac_cps_delay3k_log_interval == 0:
                log_uac_cps_delay3k_stats(
                    iter_num,
                    iter_num >= args.uac_cps_delay3k_start_iter,
                    uac_cps_delay3k_weight,
                    uac_cps_delay3k_disagreement,
                    uac_cps_delay3k_agreement,
                    loss,
                )
            if args.use_bdw_fft_power16_v2 and iter_num > 0 and iter_num % args.bdw_fft_power16_log_interval == 0:
                log_bdw_fft_power16_stats(
                    iter_num,
                    bdw_fft_power16_weight,
                    bdw_fft_power16_disagreement,
                    bdw_fft_power16_agreement,
                    loss,
                )
            if args.use_fos_fft_v1 and iter_num > 0 and iter_num % args.fos_fft_v1_log_interval == 0:
                log_fos_fft_v1_stats(iter_num, fos_fft_loss_value, loss)

            dpl_msp_logit_loss = loss.new_tensor(0.0)
            dpl_msp_logit_stats = {}
            if args.use_dpl_msp_logit_lite:
                dpl_msp_logit_loss, dpl_msp_logit_stats = dpl_msp_logit_lite_loss(
                    out_1, out_2, image_output_1, image_output_2, outputs, label_batch, args.labeled_bs
                )
                loss = loss + args.dpl_msp_logit_lite_weight * dpl_msp_logit_loss

            prediction_disagreement_2d_loss = loss.new_tensor(0.0)
            prediction_disagreement_2d_stats = {}
            if args.use_prediction_disagreement_2d_adapt:
                prediction_disagreement_2d_loss, prediction_disagreement_2d_stats = prediction_disagreement_2d_adapt_loss(
                    out_unl_1, out_unl_2, args.prediction_disagreement_2d_top_quantile
                )
                loss = loss + args.prediction_disagreement_2d_weight * prediction_disagreement_2d_loss

            if args.use_m3hl_mmm_lite:
                m3hl_loss_1, m3hl_mask_mean_1, m3hl_finite_1 = m3hl_mmm_lite_consistency(
                    model1, uimg_a, uimg_b, pre_a, pre_b,
                    args.m3hl_mmm_mask_ratio, args.m3hl_mmm_patch_size)
                m3hl_loss_2, m3hl_mask_mean_2, m3hl_finite_2 = m3hl_mmm_lite_consistency(
                    model2, uimg_a_s, uimg_b_s, pre_a_s, pre_b_s,
                    args.m3hl_mmm_mask_ratio, args.m3hl_mmm_patch_size)
                m3hl_mmm_loss = 0.5 * (m3hl_loss_1 + m3hl_loss_2)
                loss = loss + args.m3hl_mmm_weight * m3hl_mmm_loss
                if iter_num > 0 and iter_num % args.m3hl_mmm_log_interval == 0:
                    m3hl_mask_mean = 0.5 * (m3hl_mask_mean_1 + m3hl_mask_mean_2)
                    m3hl_finite = m3hl_finite_1 and m3hl_finite_2 and bool(torch.isfinite(loss).detach().cpu().item())
                    logging.info('iteration %d : m3hl_mmm_enabled : True, m3hl_mmm_loss : %f, m3hl_mmm_mask_mean : %f, m3hl_mmm_finite : %s' %
                                 (iter_num, m3hl_mmm_loss.detach().item(), m3hl_mask_mean.detach().item(), str(m3hl_finite)))

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
            


            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f' % (iter_num, loss, loss_1, loss_2))
            if args.use_mask_token_prehead_router and iter_num % args.mask_token_log_interval == 0:
                log_mask_token_router_stats(model1, "model1", iter_num)
                log_mask_token_router_stats(model2, "model2", iter_num)
            if args.use_dpl_msp_logit_lite and iter_num % args.dpl_msp_log_interval == 0:
                log_dpl_msp_logit_lite_stats(
                    iter_num,
                    float(dpl_msp_logit_loss.detach().cpu()),
                    dpl_msp_logit_stats,
                )
            if args.use_prediction_disagreement_2d_adapt and iter_num % args.prediction_disagreement_2d_log_interval == 0:
                log_prediction_disagreement_2d_stats(
                    iter_num,
                    prediction_disagreement_2d_stats,
                )
            if (
                args.prediction_disagreement_neighbor_regularization
                and iter_num % args.prediction_disagreement_neighbor_log_interval == 0
            ):
                log_prediction_disagreement_neighbor_stats(
                    iter_num,
                    prediction_disagreement_neighbor_stats_1,
                    prediction_disagreement_neighbor_stats_2,
                )
            if args.use_reliable_peer_cps_gate and iter_num % args.reliable_peer_cps_log_interval == 0:
                log_reliable_peer_cps_gate_stats(iter_num, reliable_peer_cps_stats)
            if (
                args.use_boundary_aware_pseudo_consistency
                and iter_num % args.boundary_aware_pseudo_log_interval == 0
            ):
                log_boundary_aware_pseudo_stats(
                    iter_num,
                    boundary_aware_main_stats,
                    boundary_aware_fft_stats,
                    boundary_aware_pseudo_total_loss,
                )
            if args.use_quality_aware_ict_reweighting and iter_num % args.quality_aware_ict_log_interval == 0:
                log_quality_aware_ict_stats(iter_num, quality_aware_ict_stats)
            if (
                args.use_sdcl_discrepancy_correction_lite
                and iter_num % args.sdcl_discrepancy_log_interval == 0
            ):
                log_sdcl_discrepancy_correction_stats(
                    iter_num,
                    sdcl_main_stats,
                    sdcl_fft_stats,
                    sdcl_discrepancy_total_loss,
                )
            if args.use_shape_prior_area_lite and iter_num % args.shape_prior_area_log_interval == 0:
                log_shape_prior_area_lite_stats(iter_num, shape_prior_area_stats)
            if args.use_strubar_ipba_lite and iter_num % args.strubar_ipba_log_interval == 0:
                log_strubar_ipba_lite_stats(
                    iter_num,
                    float(strubar_ipba_loss.detach().cpu()),
                    strubar_ipba_stats,
                )
	            
            # logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
             
            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(loader):
                    metric_i = test_single_volume_promise_for_model(sampled_batch["image"], sampled_batch["mask"], model1, classes=num_classes)
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
                for i_batch, sampled_batch in enumerate(test_loader):
                    metric_i = test_single_volume_promise_for_model(sampled_batch["image"], sampled_batch["mask"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_test)
                writer.add_scalar('info/model1_test_{}_dice'.format(1),metric_list[0], iter_num)
                performance1_test = np.mean(metric_list, axis=0)
                
                logging.info('iteration %d : model1_mean_test_dice : %f' % (iter_num, performance1_test))
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
    final_model_path = os.path.join(snapshot_path, 'model1_iter_{}_final.pth'.format(iter_num))
    final_alias_path = os.path.join(snapshot_path, '{}_final_model.pth'.format(args.model_1))
    torch.save(model1.state_dict(), final_model_path)
    torch.save(model1.state_dict(), final_alias_path)
    logging.info('self-train final checkpoint saved: %s' % final_model_path)
    logging.info('self-train final checkpoint alias saved: %s' % final_alias_path)
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
