import argparse
import logging
import math
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
parser.add_argument('--use_sba_bdl_v1', action='store_true',
                    help='enable supervised boundary-aware Dice auxiliary loss on labeled ICT branch')
parser.add_argument('--sba_bdl_v1_weight', type=float, default=0.05,
                    help='weight for supervised boundary-aware Dice auxiliary loss')
parser.add_argument('--sba_bdl_v1_log_interval', type=int, default=200,
                    help='iteration interval for SBA-BDL endpoint logging')
parser.add_argument('--use_astc_csab_lite_v1', action='store_true',
                    help='enable annotation-free structure-tensor/contour-structure alignment consistency lite')
parser.add_argument('--astc_csab_lite_weight', type=float, default=0.01,
                    help='weight for ASTC-CSAB lite auxiliary loss')
parser.add_argument('--astc_csab_log_interval', type=int, default=200,
                    help='iteration interval for ASTC-CSAB endpoint logging')
parser.add_argument('--use_cmsc_lite_v1', action='store_true',
                    help='enable centroid-moment shape consistency lite on the labeled supervised branch')
parser.add_argument('--cmsc_lite_v1_weight', type=float, default=0.01,
                    help='weight for CMSC lite auxiliary loss')
parser.add_argument('--cmsc_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for CMSC lite endpoint logging')
parser.add_argument('--use_bcsi_csi_lite_v1', action='store_true',
                    help='enable BCSI-inspired channel-selective semantic interaction lite')
parser.add_argument('--bcsi_csi_lite_v1_weight', type=float, default=0.01,
                    help='weight for BCSI-CSI lite auxiliary loss')
parser.add_argument('--bcsi_csi_lite_v1_top_ratio', type=float, default=0.25,
                    help='fraction of shared decoder feature channels selected for BCSI-CSI lite')
parser.add_argument('--bcsi_csi_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for BCSI-CSI lite endpoint logging')
parser.add_argument('--use_aupp_prehead_lite_v1', action='store_true',
                    help='enable parameter-free probabilistic decoder pre-head perturbation')
parser.add_argument('--aupp_prehead_lite_v1_strength', type=float, default=0.05,
                    help='fixed strength for AUPP pre-head stochastic perturbation')
parser.add_argument('--aupp_prehead_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for AUPP pre-head endpoint logging')
parser.add_argument('--use_sgrs_lite_region_weighting_v1', action='store_true',
                    help='enable SGRS-lite regional pseudo-label risk weighting for cross-pseudo supervision')
parser.add_argument('--sgrs_lite_v1_weight', type=float, default=0.1,
                    help='interpolation weight between original and SGRS-lite weighted cross-pseudo loss')
parser.add_argument('--sgrs_lite_v1_conf_threshold', type=float, default=0.75,
                    help='teacher confidence threshold for SGRS-lite consistent region')
parser.add_argument('--sgrs_lite_v1_margin_threshold', type=float, default=0.20,
                    help='teacher top-1/top-2 margin threshold for SGRS-lite consistent region')
parser.add_argument('--sgrs_lite_v1_min_weight', type=float, default=0.25,
                    help='minimum regional loss weight for uncertain/disagreement pseudo-label pixels')
parser.add_argument('--sgrs_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for SGRS-lite endpoint logging')
parser.add_argument('--use_stack_hard_case_late_reweight_v1', action='store_true',
                    help='enable late supervised hard-case reweighting for Mask Token + AdaMix stack')
parser.add_argument('--stack_hard_case_late_reweight_v1_weight', type=float, default=0.02,
                    help='weight for stack hard-case late reweighting auxiliary supervised loss')
parser.add_argument('--stack_hard_case_late_reweight_v1_start_iter', type=int, default=18000,
                    help='self-train iteration where stack hard-case late reweighting starts')
parser.add_argument('--stack_hard_case_late_reweight_v1_ramp_iters', type=int, default=4000,
                    help='number of iterations used to ramp stack hard-case late reweighting')
parser.add_argument('--stack_hard_case_late_reweight_v1_log_interval', type=int, default=200,
                    help='iteration interval for stack hard-case late reweighting logging')
parser.add_argument('--use_stack_confidence_band_fg_calibration_v1', action='store_true',
                    help='enable late foreground confidence-band calibration for Mask Token + AdaMix stack')
parser.add_argument('--stack_confidence_band_fg_calibration_v1_weight', type=float, default=0.01,
                    help='weight for stack confidence-band foreground calibration loss')
parser.add_argument('--stack_confidence_band_fg_calibration_v1_start_iter', type=int, default=22000,
                    help='self-train iteration where stack confidence-band foreground calibration starts')
parser.add_argument('--stack_confidence_band_fg_calibration_v1_ramp_iters', type=int, default=3000,
                    help='number of iterations used to ramp stack confidence-band foreground calibration')
parser.add_argument('--stack_confidence_band_fg_calibration_v1_low', type=float, default=0.25,
                    help='lower foreground probability bound for stack confidence-band calibration')
parser.add_argument('--stack_confidence_band_fg_calibration_v1_high', type=float, default=0.75,
                    help='upper foreground probability bound for stack confidence-band calibration')
parser.add_argument('--stack_confidence_band_fg_calibration_v1_agree_margin', type=float, default=0.20,
                    help='maximum peer foreground probability gap for stack confidence-band calibration')
parser.add_argument('--stack_confidence_band_fg_calibration_v1_log_interval', type=int, default=200,
                    help='iteration interval for stack confidence-band foreground calibration logging')
parser.add_argument('--use_stack_early_uncertainty_balanced_peer_consistency_v1', action='store_true',
                    help='enable early uncertainty-balanced peer consistency for Mask Token + AdaMix stack')
parser.add_argument('--stack_early_uncertainty_balanced_peer_consistency_v1_weight', type=float, default=0.003,
                    help='weight for early uncertainty-balanced peer consistency loss')
parser.add_argument('--stack_early_uncertainty_balanced_peer_consistency_v1_start_iter', type=int, default=0,
                    help='self-train iteration where early uncertainty-balanced peer consistency starts')
parser.add_argument('--stack_early_uncertainty_balanced_peer_consistency_v1_ramp_iters', type=int, default=2000,
                    help='number of iterations used to ramp early uncertainty-balanced peer consistency')
parser.add_argument('--stack_early_uncertainty_balanced_peer_consistency_v1_entropy_weight', type=float, default=0.50,
                    help='blend weight for normalized entropy in early uncertainty-balanced peer consistency')
parser.add_argument('--stack_early_uncertainty_balanced_peer_consistency_v1_min_pixel_weight', type=float, default=0.25,
                    help='minimum pixel weight for early uncertainty-balanced peer consistency')
parser.add_argument('--stack_early_uncertainty_balanced_peer_consistency_v1_log_interval', type=int, default=200,
                    help='iteration interval for early uncertainty-balanced peer consistency logging')
parser.add_argument('--use_stack_early_fg_area_range_calibration_v1', action='store_true',
                    help='enable early foreground-area range calibration for Mask Token + AdaMix stack')
parser.add_argument('--stack_early_fg_area_range_calibration_v1_weight', type=float, default=0.002,
                    help='weight for early foreground-area range calibration loss')
parser.add_argument('--stack_early_fg_area_range_calibration_v1_start_iter', type=int, default=0,
                    help='self-train iteration where early foreground-area range calibration starts')
parser.add_argument('--stack_early_fg_area_range_calibration_v1_ramp_iters', type=int, default=2000,
                    help='number of iterations used to ramp early foreground-area range calibration')
parser.add_argument('--stack_early_fg_area_range_calibration_v1_margin', type=float, default=0.05,
                    help='foreground area tolerance around labeled-batch foreground ratio')
parser.add_argument('--stack_early_fg_area_range_calibration_v1_min_area', type=float, default=0.001,
                    help='minimum allowed foreground area lower bound')
parser.add_argument('--stack_early_fg_area_range_calibration_v1_max_area', type=float, default=0.35,
                    help='maximum allowed foreground area upper bound')
parser.add_argument('--stack_early_fg_area_range_calibration_v1_log_interval', type=int, default=200,
                    help='iteration interval for early foreground-area range calibration logging')
parser.add_argument('--use_stack_early_class_confidence_anchor_v1', action='store_true',
                    help='enable early labeled class-confidence anchor calibration for Mask Token + AdaMix stack')
parser.add_argument('--stack_early_class_confidence_anchor_v1_weight', type=float, default=0.003,
                    help='weight for early class-confidence anchor loss')
parser.add_argument('--stack_early_class_confidence_anchor_v1_start_iter', type=int, default=0,
                    help='self-train iteration where early class-confidence anchor starts')
parser.add_argument('--stack_early_class_confidence_anchor_v1_ramp_iters', type=int, default=2000,
                    help='number of iterations used to ramp early class-confidence anchor')
parser.add_argument('--stack_early_class_confidence_anchor_v1_margin', type=float, default=0.05,
                    help='confidence margin around labeled foreground/background anchors')
parser.add_argument('--stack_early_class_confidence_anchor_v1_min_region_ratio', type=float, default=0.01,
                    help='minimum peer-agreement region ratio required for class-confidence anchor loss')
parser.add_argument('--stack_early_class_confidence_anchor_v1_log_interval', type=int, default=200,
                    help='iteration interval for early class-confidence anchor logging')
parser.add_argument('--use_stack_early_soft_boundary_peer_consistency_v1', action='store_true',
                    help='enable early soft-boundary peer consistency for Mask Token + AdaMix stack')
parser.add_argument('--stack_early_soft_boundary_peer_consistency_v1_weight', type=float, default=0.002,
                    help='weight for early soft-boundary peer consistency loss')
parser.add_argument('--stack_early_soft_boundary_peer_consistency_v1_start_iter', type=int, default=0,
                    help='self-train iteration where early soft-boundary peer consistency starts')
parser.add_argument('--stack_early_soft_boundary_peer_consistency_v1_ramp_iters', type=int, default=2000,
                    help='number of iterations used to ramp early soft-boundary peer consistency')
parser.add_argument('--stack_early_soft_boundary_peer_consistency_v1_confidence_floor', type=float, default=0.55,
                    help='minimum peer confidence for early soft-boundary peer consistency weighting')
parser.add_argument('--stack_early_soft_boundary_peer_consistency_v1_log_interval', type=int, default=200,
                    help='iteration interval for early soft-boundary peer consistency logging')
parser.add_argument('--use_stack_early_reliable_prototype_consistency_v1', action='store_true',
                    help='enable early reliable prototype consistency for Mask Token + AdaMix stack')
parser.add_argument('--stack_early_reliable_prototype_consistency_v1_weight', type=float, default=0.001,
                    help='weight for early reliable prototype consistency loss')
parser.add_argument('--stack_early_reliable_prototype_consistency_v1_start_iter', type=int, default=0,
                    help='self-train iteration where early reliable prototype consistency starts')
parser.add_argument('--stack_early_reliable_prototype_consistency_v1_ramp_iters', type=int, default=2000,
                    help='number of iterations used to ramp early reliable prototype consistency')
parser.add_argument('--stack_early_reliable_prototype_consistency_v1_confidence_floor', type=float, default=0.55,
                    help='minimum peer confidence for early reliable prototype weighting')
parser.add_argument('--stack_early_reliable_prototype_consistency_v1_entropy_weight', type=float, default=0.50,
                    help='weight of normalized entropy in early reliable prototype reliability')
parser.add_argument('--stack_early_reliable_prototype_consistency_v1_separation_weight', type=float, default=0.10,
                    help='weight for foreground/background prototype separation term')
parser.add_argument('--stack_early_reliable_prototype_consistency_v1_log_interval', type=int, default=200,
                    help='iteration interval for early reliable prototype consistency logging')
parser.add_argument('--use_stack_early_entropy_margin_sharpening_v1', action='store_true',
                    help='enable early entropy-margin sharpening for Mask Token + AdaMix stack')
parser.add_argument('--stack_early_entropy_margin_sharpening_v1_weight', type=float, default=0.001,
                    help='weight for early entropy-margin sharpening loss')
parser.add_argument('--stack_early_entropy_margin_sharpening_v1_start_iter', type=int, default=0,
                    help='self-train iteration where early entropy-margin sharpening starts')
parser.add_argument('--stack_early_entropy_margin_sharpening_v1_ramp_iters', type=int, default=2000,
                    help='number of iterations used to ramp early entropy-margin sharpening')
parser.add_argument('--stack_early_entropy_margin_sharpening_v1_confidence_floor', type=float, default=0.60,
                    help='minimum peer confidence for early entropy-margin sharpening')
parser.add_argument('--stack_early_entropy_margin_sharpening_v1_margin_floor', type=float, default=0.20,
                    help='minimum top-2 probability margin for early entropy-margin sharpening')
parser.add_argument('--stack_early_entropy_margin_sharpening_v1_kl_weight', type=float, default=0.50,
                    help='weight for peer KL alignment term in early entropy-margin sharpening')
parser.add_argument('--stack_early_entropy_margin_sharpening_v1_log_interval', type=int, default=200,
                    help='iteration interval for early entropy-margin sharpening logging')
parser.add_argument('--use_evidential_cps_lite_v1', action='store_true',
                    help='enable Evidential CPS Lite V1 reliability consistency')
parser.add_argument('--evidential_cps_lite_v1_weight', type=float, default=0.01,
                    help='weight for Evidential CPS Lite V1 auxiliary consistency')
parser.add_argument('--evidential_cps_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for Evidential CPS Lite V1 endpoint logging')
parser.add_argument('--use_hfc_fp_lite_v1', action='store_true',
                    help='enable high-frequency foreground consistency prompt lite auxiliary')
parser.add_argument('--hfc_fp_lite_v1_weight', type=float, default=0.01,
                    help='weight for HFC-FP Lite V1 auxiliary consistency')
parser.add_argument('--hfc_fp_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for HFC-FP Lite V1 endpoint logging')
parser.add_argument('--use_rncl_residual_lite_v1', action='store_true',
                    help='enable RNCL-inspired residual refinement lite auxiliary')
parser.add_argument('--rncl_residual_lite_v1_weight', type=float, default=0.01,
                    help='weight for RNCL residual lite V1 auxiliary consistency')
parser.add_argument('--rncl_residual_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for RNCL residual lite V1 endpoint logging')
parser.add_argument('--use_ocpr_lite_v1', action='store_true',
                    help='enable OCPR Lite V1 disagreement-conditioned overconfidence penalty')
parser.add_argument('--ocpr_lite_v1_weight', type=float, default=0.01,
                    help='weight for OCPR Lite V1 auxiliary overconfidence penalty')
parser.add_argument('--ocpr_lite_v1_confidence_ceiling', type=float, default=0.85,
                    help='confidence ceiling used on OCPR Lite V1 disagreement pixels')
parser.add_argument('--ocpr_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for OCPR Lite V1 endpoint logging')
parser.add_argument('--use_sra_seb_lite_v1', action='store_true',
                    help='enable SRA-SEB Lite V1 soft-edge BCP seam supervision')
parser.add_argument('--sra_seb_lite_v1_weight', type=float, default=0.05,
                    help='weight for SRA-SEB Lite V1 seam soft-label loss')
parser.add_argument('--sra_seb_lite_v1_blur_kernel', type=int, default=9,
                    help='average-pool kernel used to soften BCP hard mask edges')
parser.add_argument('--sra_seb_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for SRA-SEB Lite V1 endpoint logging')
parser.add_argument('--use_adamix_bcp_lite_v1', action='store_true',
                    help='enable AdaMix-inspired adaptive BCP patch sizing')
parser.add_argument('--adamix_bcp_lite_v1_min_side', type=float, default=0.18,
                    help='minimum adaptive BCP patch side fraction')
parser.add_argument('--adamix_bcp_lite_v1_max_side', type=float, default=0.38,
                    help='maximum adaptive BCP patch side fraction')
parser.add_argument('--adamix_bcp_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for AdaMix-BCP Lite V1 endpoint logging')
parser.add_argument('--use_plf_cip_lite_v1', action='store_true',
                    help='enable pseudo-label fusion lite inspired by cross-image perturbation')
parser.add_argument('--plf_cip_lite_v1_weight', type=float, default=0.03,
                    help='weight for PLF-CIP Lite V1 auxiliary consistency')
parser.add_argument('--plf_cip_lite_v1_temperature', type=float, default=0.75,
                    help='temperature used to sharpen fused soft pseudo labels')
parser.add_argument('--plf_cip_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for PLF-CIP Lite V1 endpoint logging')
parser.add_argument('--use_pem_lite_v1', action='store_true',
                    help='enable conservative entropy minimization on unlabeled self-train logits')
parser.add_argument('--pem_lite_v1_weight', type=float, default=0.005,
                    help='weight for PEM Lite V1 unlabeled entropy minimization')
parser.add_argument('--pem_lite_v1_start_iter', type=int, default=1000,
                    help='self-train iteration where PEM Lite V1 starts ramping in')
parser.add_argument('--pem_lite_v1_ramp_iters', type=int, default=2000,
                    help='number of self-train iterations used to ramp PEM Lite V1 weight')
parser.add_argument('--pem_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for PEM Lite V1 endpoint logging')
parser.add_argument('--use_ema_bcp_anchor_lite_v1', action='store_true',
                    help='enable EMA teacher soft anchor on BCP mixed unlabeled views')
parser.add_argument('--ema_bcp_anchor_lite_v1_weight', type=float, default=0.02,
                    help='weight for EMA BCP Anchor Lite V1 auxiliary consistency')
parser.add_argument('--ema_bcp_anchor_lite_v1_start_iter', type=int, default=1000,
                    help='self-train iteration where EMA BCP Anchor Lite V1 starts ramping in')
parser.add_argument('--ema_bcp_anchor_lite_v1_ramp_iters', type=int, default=2000,
                    help='number of self-train iterations used to ramp EMA BCP Anchor Lite V1 weight')
parser.add_argument('--ema_bcp_anchor_lite_v1_conf_floor', type=float, default=0.25,
                    help='minimum confidence weight for EMA BCP Anchor Lite V1')
parser.add_argument('--ema_bcp_anchor_lite_v1_log_interval', type=int, default=200,
                    help='iteration interval for EMA BCP Anchor Lite V1 endpoint logging')
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

def log_aupp_prehead_lite_v1_stats(model, model_name, iter_num):
    decoder = getattr(model, "decoder", None)
    aupp = getattr(decoder, "aupp_prehead", None)
    if aupp is None:
        logging.info(
            "iteration %d : %s_aupp_prehead_lite_v1_enabled : False" %
            (iter_num, model_name)
        )
        return
    stats = getattr(aupp, "last_stats", {}) or {}
    logging.info(
        "iteration %d : %s_aupp_prehead_lite_v1_enabled : True, "
        "active : %s, strength : %s, uncertainty_mean : %s, "
        "noise_std : %s, perturb_to_feature_ratio : %s, finite : %s" %
        (
            iter_num,
            model_name,
            stats.get("active", "NA"),
            stats.get("strength", "NA"),
            stats.get("uncertainty_mean", "NA"),
            stats.get("noise_std", "NA"),
            stats.get("perturb_to_feature_ratio", "NA"),
            stats.get("finite", "NA"),
        )
    )

def sgrs_lite_v1_weighted_pseudo_loss(student_soft, teacher_soft, student_pseudo, teacher_pseudo):
    teacher_soft = teacher_soft.detach()
    teacher_pseudo = teacher_pseudo.detach()
    student_pseudo = student_pseudo.detach()
    top2 = torch.topk(teacher_soft, k=2, dim=1).values
    teacher_conf = top2[:, 0]
    teacher_margin = top2[:, 0] - top2[:, 1]
    agreement = (student_pseudo == teacher_pseudo)
    confident = (
        (teacher_conf >= args.sgrs_lite_v1_conf_threshold)
        & (teacher_margin >= args.sgrs_lite_v1_margin_threshold)
    )
    consistent = agreement & confident
    low_conf = ~confident
    regional_weight = torch.ones_like(teacher_conf)
    regional_weight = torch.where(
        consistent,
        regional_weight,
        regional_weight * args.sgrs_lite_v1_min_weight,
    )
    target = F.one_hot(
        teacher_pseudo.long(),
        num_classes=student_soft.shape[1],
    ).permute(0, 3, 1, 2).float()
    weight = regional_weight.unsqueeze(1)
    numerator = 2.0 * torch.sum(student_soft * target * weight, dim=(0, 2, 3))
    denominator = torch.sum((student_soft + target) * weight, dim=(0, 2, 3)).clamp_min(1e-6)
    weighted_loss = torch.mean(1.0 - numerator / denominator)
    stats = {
        "weight_mean": regional_weight.mean().detach(),
        "consistent_ratio": consistent.float().mean().detach(),
        "disagreement_ratio": (~agreement).float().mean().detach(),
        "low_conf_ratio": low_conf.float().mean().detach(),
        "finite": torch.isfinite(weighted_loss).detach(),
    }
    return weighted_loss, stats

def log_sgrs_lite_v1_stats(iter_num, stats):
    logging.info(
        "iteration %d : sgrs_lite_v1_enabled : True, sgrs_lite_v1_loss_delta : %f, "
        "sgrs_lite_v1_weight_mean : %f, sgrs_lite_v1_consistent_ratio : %f, "
        "sgrs_lite_v1_disagreement_ratio : %f, sgrs_lite_v1_low_conf_ratio : %f, "
        "sgrs_lite_v1_finite : %s" %
        (
            iter_num,
            stats["loss_delta"].item(),
            stats["weight_mean"].item(),
            stats["consistent_ratio"].item(),
            stats["disagreement_ratio"].item(),
            stats["low_conf_ratio"].item(),
            stats["finite"],
        )
    )

def stack_hard_case_late_reweight_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return max(0.0, min(1.0, float(iter_num - start_iter) / float(ramp_iters)))

def stack_hard_case_late_reweight_v1_loss(logits, labels, num_classes):
    labels = labels.long()
    per_pixel_ce = F.cross_entropy(logits, labels, reduction='none')
    per_case_ce = per_pixel_ce.flatten(1).mean(dim=1)

    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    per_case_dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean(dim=1)

    difficulty = (0.5 * (per_case_ce + per_case_dice)).detach()
    centered = difficulty - difficulty.mean()
    scale = centered.abs().mean().clamp_min(1e-6)
    weights = (1.0 + centered / scale).clamp(0.5, 2.0)
    weights = weights / weights.mean().clamp_min(1e-6)

    weighted_ce = (weights * per_case_ce).mean()
    weighted_dice = (weights * per_case_dice).mean()
    value = 0.5 * (weighted_ce + weighted_dice)
    with torch.no_grad():
        stats = {
            "weight_mean": float(weights.mean().detach().cpu()),
            "weight_min": float(weights.min().detach().cpu()),
            "weight_max": float(weights.max().detach().cpu()),
            "difficulty_mean": float(difficulty.mean().detach().cpu()),
            "difficulty_max": float(difficulty.max().detach().cpu()),
            "finite": bool(torch.isfinite(value).all().detach().cpu()),
        }
    return value, stats

def log_stack_hard_case_late_reweight_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).all().detach().cpu().item())
    logging.info(
        "iteration %d : stack_hard_case_late_reweight_v1_enabled : True, "
        "stack_hard_case_late_reweight_v1_loss : %s, "
        "stack_hard_case_late_reweight_v1_ramp : %s, "
        "stack_hard_case_late_reweight_v1_weight_mean : %s, "
        "stack_hard_case_late_reweight_v1_weight_min : %s, "
        "stack_hard_case_late_reweight_v1_weight_max : %s, "
        "stack_hard_case_late_reweight_v1_difficulty_mean : %s, "
        "stack_hard_case_late_reweight_v1_difficulty_max : %s, "
        "stack_hard_case_late_reweight_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("weight_mean", "NA"),
            stats.get("weight_min", "NA"),
            stats.get("weight_max", "NA"),
            stats.get("difficulty_mean", "NA"),
            stats.get("difficulty_max", "NA"),
            stats.get("finite", "NA"),
            finite,
        )
    )

def stack_confidence_band_fg_calibration_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return max(0.0, min(1.0, float(iter_num - start_iter) / float(ramp_iters)))

def stack_confidence_band_fg_calibration_v1_loss(
    logits_1,
    logits_2,
    low,
    high,
    agree_margin,
):
    probs_1 = torch.softmax(logits_1, dim=1)
    probs_2 = torch.softmax(logits_2, dim=1)
    if probs_1.shape[1] > 1:
        fg_1 = probs_1[:, 1:].sum(dim=1)
        fg_2 = probs_2[:, 1:].sum(dim=1)
    else:
        fg_1 = probs_1[:, 0]
        fg_2 = probs_2[:, 0]

    fg_mean = 0.5 * (fg_1 + fg_2)
    band = (fg_mean >= low) & (fg_mean <= high)
    agree = (fg_1 - fg_2).abs() <= agree_margin
    selected = (band & agree).float()

    fg_mean_clamped = fg_mean.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(
        fg_mean_clamped * torch.log(fg_mean_clamped)
        + (1.0 - fg_mean_clamped) * torch.log(1.0 - fg_mean_clamped)
    )
    denom = selected.sum().clamp_min(1.0)
    value = (entropy * selected).sum() / denom

    with torch.no_grad():
        stats = {
            "band_ratio": float(band.float().mean().detach().cpu()),
            "agree_ratio": float(agree.float().mean().detach().cpu()),
            "selected_ratio": float(selected.mean().detach().cpu()),
            "fg_mean": float(fg_mean.mean().detach().cpu()),
            "fg_selected_mean": float(
                ((fg_mean * selected).sum() / denom).detach().cpu()
            ),
            "finite": bool(torch.isfinite(value).all().detach().cpu()),
        }
    return value, stats

def log_stack_confidence_band_fg_calibration_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).all().detach().cpu().item())
    logging.info(
        "iteration %d : stack_confidence_band_fg_calibration_v1_enabled : True, "
        "stack_confidence_band_fg_calibration_v1_loss : %s, "
        "stack_confidence_band_fg_calibration_v1_ramp : %s, "
        "stack_confidence_band_fg_calibration_v1_band_ratio : %s, "
        "stack_confidence_band_fg_calibration_v1_agree_ratio : %s, "
        "stack_confidence_band_fg_calibration_v1_selected_ratio : %s, "
        "stack_confidence_band_fg_calibration_v1_fg_mean : %s, "
        "stack_confidence_band_fg_calibration_v1_fg_selected_mean : %s, "
        "stack_confidence_band_fg_calibration_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("band_ratio", "NA"),
            stats.get("agree_ratio", "NA"),
            stats.get("selected_ratio", "NA"),
            stats.get("fg_mean", "NA"),
            stats.get("fg_selected_mean", "NA"),
            stats.get("finite", "NA"),
            finite,
        )
    )

def stack_early_uncertainty_balanced_peer_consistency_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return max(0.0, min(1.0, float(iter_num - start_iter) / float(ramp_iters)))

def stack_early_uncertainty_balanced_peer_consistency_v1_loss(
    logits_1,
    logits_2,
    entropy_weight,
    min_pixel_weight,
):
    probs_1 = torch.softmax(logits_1, dim=1)
    probs_2 = torch.softmax(logits_2, dim=1)
    eps = 1e-6
    num_classes = max(int(probs_1.shape[1]), 2)
    entropy_weight = min(max(float(entropy_weight), 0.0), 1.0)
    min_pixel_weight = min(max(float(min_pixel_weight), 0.0), 1.0)

    entropy_1 = -(probs_1.clamp_min(eps) * probs_1.clamp_min(eps).log()).sum(dim=1)
    entropy_2 = -(probs_2.clamp_min(eps) * probs_2.clamp_min(eps).log()).sum(dim=1)
    entropy_norm = (0.5 * (entropy_1 + entropy_2) / math.log(num_classes)).clamp(0.0, 1.0)
    disagreement = (probs_1.detach() - probs_2.detach()).abs().mean(dim=1).clamp(0.0, 1.0)
    agreement = (1.0 - disagreement).clamp(0.0, 1.0)
    reliability = ((1.0 - entropy_weight) * agreement + entropy_weight * entropy_norm).clamp(0.0, 1.0)
    pixel_weight = min_pixel_weight + (1.0 - min_pixel_weight) * reliability

    diff_12 = (probs_1 - probs_2.detach()).pow(2).mean(dim=1)
    diff_21 = (probs_2 - probs_1.detach()).pow(2).mean(dim=1)
    value = 0.5 * (
        (diff_12 * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
        + (diff_21 * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
    )

    with torch.no_grad():
        stats = {
            "loss": float(value.detach().cpu()),
            "entropy_norm": float(entropy_norm.mean().detach().cpu()),
            "agreement": float(agreement.mean().detach().cpu()),
            "pixel_weight": float(pixel_weight.mean().detach().cpu()),
            "disagreement": float(disagreement.mean().detach().cpu()),
            "finite": bool(torch.isfinite(value).all().detach().cpu().item()),
        }
    return value, stats

def log_stack_early_uncertainty_balanced_peer_consistency_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).all().detach().cpu().item())
    logging.info(
        "iteration %d : stack_early_uncertainty_balanced_peer_consistency_v1_enabled : True, "
        "stack_early_uncertainty_balanced_peer_consistency_v1_loss : %s, "
        "stack_early_uncertainty_balanced_peer_consistency_v1_ramp : %s, "
        "stack_early_uncertainty_balanced_peer_consistency_v1_entropy_norm : %s, "
        "stack_early_uncertainty_balanced_peer_consistency_v1_agreement : %s, "
        "stack_early_uncertainty_balanced_peer_consistency_v1_pixel_weight : %s, "
        "stack_early_uncertainty_balanced_peer_consistency_v1_disagreement : %s, "
        "stack_early_uncertainty_balanced_peer_consistency_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("entropy_norm", "NA"),
            stats.get("agreement", "NA"),
            stats.get("pixel_weight", "NA"),
            stats.get("disagreement", "NA"),
            stats.get("finite", "NA"),
            finite,
        )
    )

def stack_early_fg_area_range_calibration_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return max(0.0, min(1.0, float(iter_num - start_iter) / float(ramp_iters)))

def stack_early_fg_area_range_calibration_v1_loss(
    out_unl_1,
    out_unl_2,
    label_batch,
    labeled_bs,
    margin,
    min_area,
    max_area,
):
    eps = 1e-6
    margin = max(float(margin), 0.0)
    min_area = min(max(float(min_area), 0.0), 1.0)
    max_area = min(max(float(max_area), min_area + eps), 1.0)
    labeled_fg = (label_batch[:labeled_bs].float() > 0).float()
    with torch.no_grad():
        labeled_area = labeled_fg.mean(dim=(1, 2)).mean()
        lower = (labeled_area - margin).clamp(min_area, max_area)
        upper = (labeled_area + margin).clamp(lower + eps, max_area)
    prob_fg_1 = torch.softmax(out_unl_1, dim=1)[:, 1]
    prob_fg_2 = torch.softmax(out_unl_2, dim=1)[:, 1]
    pred_area_1 = prob_fg_1.mean(dim=(1, 2))
    pred_area_2 = prob_fg_2.mean(dim=(1, 2))
    target_1 = pred_area_1.detach().clamp(lower, upper)
    target_2 = pred_area_2.detach().clamp(lower, upper)
    loss_1 = F.smooth_l1_loss(pred_area_1, target_1)
    loss_2 = F.smooth_l1_loss(pred_area_2, target_2)
    loss = 0.5 * (loss_1 + loss_2)
    with torch.no_grad():
        below_1 = (pred_area_1 < lower).float().mean()
        above_1 = (pred_area_1 > upper).float().mean()
        below_2 = (pred_area_2 < lower).float().mean()
        above_2 = (pred_area_2 > upper).float().mean()
        stats = {
            "loss": float(loss.detach().cpu()),
            "labeled_area": float(labeled_area.detach().cpu()),
            "lower": float(lower.detach().cpu()),
            "upper": float(upper.detach().cpu()),
            "pred_area_model1": float(pred_area_1.mean().detach().cpu()),
            "pred_area_model2": float(pred_area_2.mean().detach().cpu()),
            "outside_ratio_model1": float((below_1 + above_1).detach().cpu()),
            "outside_ratio_model2": float((below_2 + above_2).detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_stack_early_fg_area_range_calibration_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).all().detach().cpu().item())
    logging.info(
        "iteration %d : stack_early_fg_area_range_calibration_v1_enabled : True, "
        "stack_early_fg_area_range_calibration_v1_loss : %s, "
        "stack_early_fg_area_range_calibration_v1_ramp : %s, "
        "stack_early_fg_area_range_calibration_v1_labeled_area : %s, "
        "stack_early_fg_area_range_calibration_v1_lower : %s, "
        "stack_early_fg_area_range_calibration_v1_upper : %s, "
        "stack_early_fg_area_range_calibration_v1_pred_area_model1 : %s, "
        "stack_early_fg_area_range_calibration_v1_pred_area_model2 : %s, "
        "stack_early_fg_area_range_calibration_v1_outside_ratio_model1 : %s, "
        "stack_early_fg_area_range_calibration_v1_outside_ratio_model2 : %s, "
        "stack_early_fg_area_range_calibration_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("labeled_area", "NA"),
            stats.get("lower", "NA"),
            stats.get("upper", "NA"),
            stats.get("pred_area_model1", "NA"),
            stats.get("pred_area_model2", "NA"),
            stats.get("outside_ratio_model1", "NA"),
            stats.get("outside_ratio_model2", "NA"),
            stats.get("finite", "NA"),
            finite,
        )
    )

def stack_early_class_confidence_anchor_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return float(min(1.0, (iter_num - start_iter) / float(ramp_iters)))

def stack_early_class_confidence_anchor_v1_loss(
    out_unl_1,
    out_unl_2,
    labeled_logits,
    label_batch,
    labeled_bs,
    margin,
    min_region_ratio,
    eps=1e-6,
):
    prob_unl_1 = torch.softmax(out_unl_1, dim=1)[:, 1]
    prob_unl_2 = torch.softmax(out_unl_2, dim=1)[:, 1]
    prob_labeled_fg = torch.softmax(labeled_logits[:labeled_bs], dim=1)[:, 1]
    labeled = label_batch[:labeled_bs].float()
    fg_mask = (labeled > 0).float()
    bg_mask = (labeled <= 0).float()
    fg_count = fg_mask.sum()
    bg_count = bg_mask.sum()
    if fg_count <= 0 or bg_count <= 0:
        zero = out_unl_1.new_tensor(0.0)
        return zero, {"finite": True, "skipped": True, "reason": "missing_labeled_class"}

    with torch.no_grad():
        fg_anchor = ((prob_labeled_fg * fg_mask).sum() / (fg_count + eps)).clamp(0.0, 1.0)
        bg_anchor = ((prob_labeled_fg * bg_mask).sum() / (bg_count + eps)).clamp(0.0, 1.0)
        pred_1 = (prob_unl_1 >= 0.5).float()
        pred_2 = (prob_unl_2 >= 0.5).float()
        agree_fg = (pred_1 * pred_2).float()
        agree_bg = ((1.0 - pred_1) * (1.0 - pred_2)).float()
        agree_fg_ratio = agree_fg.mean()
        agree_bg_ratio = agree_bg.mean()
        fg_target = (fg_anchor - margin).clamp(0.0, 1.0)
        bg_target = (bg_anchor + margin).clamp(0.0, 1.0)

    losses = []
    if float(agree_fg_ratio.detach().cpu()) >= min_region_ratio:
        fg_loss_1 = (F.relu(fg_target - prob_unl_1) * agree_fg).sum() / (agree_fg.sum() + eps)
        fg_loss_2 = (F.relu(fg_target - prob_unl_2) * agree_fg).sum() / (agree_fg.sum() + eps)
        losses.append(0.5 * (fg_loss_1 + fg_loss_2))
    if float(agree_bg_ratio.detach().cpu()) >= min_region_ratio:
        bg_loss_1 = (F.relu(prob_unl_1 - bg_target) * agree_bg).sum() / (agree_bg.sum() + eps)
        bg_loss_2 = (F.relu(prob_unl_2 - bg_target) * agree_bg).sum() / (agree_bg.sum() + eps)
        losses.append(0.5 * (bg_loss_1 + bg_loss_2))

    if not losses:
        loss = out_unl_1.new_tensor(0.0)
    else:
        loss = torch.stack(losses).mean()

    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "fg_anchor": float(fg_anchor.detach().cpu()),
            "bg_anchor": float(bg_anchor.detach().cpu()),
            "fg_target": float(fg_target.detach().cpu()),
            "bg_target": float(bg_target.detach().cpu()),
            "agree_fg_ratio": float(agree_fg_ratio.detach().cpu()),
            "agree_bg_ratio": float(agree_bg_ratio.detach().cpu()),
            "pred_fg_model1": float(prob_unl_1.mean().detach().cpu()),
            "pred_fg_model2": float(prob_unl_2.mean().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
            "skipped": False,
        }
    return loss, stats

def log_stack_early_class_confidence_anchor_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).all().detach().cpu().item())
    logging.info(
        "iteration %d : stack_early_class_confidence_anchor_v1_enabled : True, "
        "stack_early_class_confidence_anchor_v1_loss : %s, "
        "stack_early_class_confidence_anchor_v1_ramp : %s, "
        "stack_early_class_confidence_anchor_v1_fg_anchor : %s, "
        "stack_early_class_confidence_anchor_v1_bg_anchor : %s, "
        "stack_early_class_confidence_anchor_v1_fg_target : %s, "
        "stack_early_class_confidence_anchor_v1_bg_target : %s, "
        "stack_early_class_confidence_anchor_v1_agree_fg_ratio : %s, "
        "stack_early_class_confidence_anchor_v1_agree_bg_ratio : %s, "
        "stack_early_class_confidence_anchor_v1_pred_fg_model1 : %s, "
        "stack_early_class_confidence_anchor_v1_pred_fg_model2 : %s, "
        "stack_early_class_confidence_anchor_v1_skipped : %s, "
        "stack_early_class_confidence_anchor_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("fg_anchor", "NA"),
            stats.get("bg_anchor", "NA"),
            stats.get("fg_target", "NA"),
            stats.get("bg_target", "NA"),
            stats.get("agree_fg_ratio", "NA"),
            stats.get("agree_bg_ratio", "NA"),
            stats.get("pred_fg_model1", "NA"),
            stats.get("pred_fg_model2", "NA"),
            stats.get("skipped", "NA"),
            stats.get("finite", "NA"),
            finite,
        )
    )

def stack_early_soft_boundary_peer_consistency_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return float(min(1.0, (iter_num - start_iter) / float(ramp_iters)))

def stack_early_soft_boundary_peer_consistency_v1_sobel(prob, eps=1e-6):
    prob = prob.unsqueeze(1)
    kernel_x = prob.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = prob.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    grad_x = F.conv2d(prob, kernel_x, padding=1)
    grad_y = F.conv2d(prob, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps).squeeze(1)

def stack_early_soft_boundary_peer_consistency_v1_loss(
    out_unl_1,
    out_unl_2,
    confidence_floor,
    eps=1e-6,
):
    soft_1 = torch.softmax(out_unl_1, dim=1)
    soft_2 = torch.softmax(out_unl_2, dim=1)
    prob_1 = soft_1[:, 1]
    prob_2 = soft_2[:, 1]
    edge_1 = stack_early_soft_boundary_peer_consistency_v1_sobel(prob_1, eps)
    edge_2 = stack_early_soft_boundary_peer_consistency_v1_sobel(prob_2, eps)

    with torch.no_grad():
        pred_1 = torch.argmax(soft_1, dim=1)
        pred_2 = torch.argmax(soft_2, dim=1)
        conf_1 = torch.max(soft_1, dim=1)[0]
        conf_2 = torch.max(soft_2, dim=1)[0]
        agreement = (pred_1 == pred_2).float()
        confidence = (0.5 * (conf_1 + conf_2)).clamp(0.0, 1.0)
        confidence_gate = (confidence >= confidence_floor).float()
        peer_edge = torch.maximum(edge_1.detach(), edge_2.detach())
        peer_edge_max = peer_edge.amax(dim=(1, 2), keepdim=True).clamp_min(eps)
        boundary_weight = (peer_edge / peer_edge_max).clamp(0.0, 1.0)
        pixel_weight = agreement * confidence_gate * (0.25 + 0.75 * boundary_weight)
        active_ratio = (pixel_weight > 0).float().mean()
        boundary_weight_mean = boundary_weight.mean()
        confidence_mean = confidence.mean()
        agreement_ratio = agreement.mean()

    denom = pixel_weight.sum().clamp_min(eps)
    loss_1 = ((edge_1 - edge_2.detach()).abs() * pixel_weight).sum() / denom
    loss_2 = ((edge_2 - edge_1.detach()).abs() * pixel_weight).sum() / denom
    loss = 0.5 * (loss_1 + loss_2)

    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "edge_model1": float(edge_1.mean().detach().cpu()),
            "edge_model2": float(edge_2.mean().detach().cpu()),
            "agreement": float(agreement_ratio.detach().cpu()),
            "confidence": float(confidence_mean.detach().cpu()),
            "active_ratio": float(active_ratio.detach().cpu()),
            "boundary_weight": float(boundary_weight_mean.detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_stack_early_soft_boundary_peer_consistency_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).all().detach().cpu().item())
    logging.info(
        "iteration %d : stack_early_soft_boundary_peer_consistency_v1_enabled : True, "
        "stack_early_soft_boundary_peer_consistency_v1_loss : %s, "
        "stack_early_soft_boundary_peer_consistency_v1_ramp : %s, "
        "stack_early_soft_boundary_peer_consistency_v1_edge_model1 : %s, "
        "stack_early_soft_boundary_peer_consistency_v1_edge_model2 : %s, "
        "stack_early_soft_boundary_peer_consistency_v1_agreement : %s, "
        "stack_early_soft_boundary_peer_consistency_v1_confidence : %s, "
        "stack_early_soft_boundary_peer_consistency_v1_active_ratio : %s, "
        "stack_early_soft_boundary_peer_consistency_v1_boundary_weight : %s, "
        "stack_early_soft_boundary_peer_consistency_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("edge_model1", "NA"),
            stats.get("edge_model2", "NA"),
            stats.get("agreement", "NA"),
            stats.get("confidence", "NA"),
            stats.get("active_ratio", "NA"),
            stats.get("boundary_weight", "NA"),
            stats.get("finite", "NA"),
            finite,
        )
    )

def stack_early_reliable_prototype_consistency_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return float(min(1.0, (iter_num - start_iter) / float(ramp_iters)))

def stack_early_reliable_prototype_consistency_v1_weighted_proto(logits, weight, eps=1e-6):
    denom = weight.sum(dim=(0, 2, 3)).clamp_min(eps)
    return (logits * weight).sum(dim=(0, 2, 3)) / denom

def stack_early_reliable_prototype_consistency_v1_loss(
    out_unl_1,
    out_unl_2,
    confidence_floor,
    entropy_weight,
    separation_weight,
    eps=1e-6,
):
    soft_1 = torch.softmax(out_unl_1, dim=1)
    soft_2 = torch.softmax(out_unl_2, dim=1)
    with torch.no_grad():
        pred_1 = torch.argmax(soft_1, dim=1, keepdim=True)
        pred_2 = torch.argmax(soft_2, dim=1, keepdim=True)
        conf_1 = torch.max(soft_1, dim=1, keepdim=True)[0]
        conf_2 = torch.max(soft_2, dim=1, keepdim=True)[0]
        agreement = (pred_1 == pred_2).float()
        confidence = (0.5 * (conf_1 + conf_2)).clamp(0.0, 1.0)
        confidence_gate = (confidence >= confidence_floor).float()
        entropy_1 = -(soft_1 * torch.log(soft_1.clamp_min(eps))).sum(dim=1, keepdim=True) / math.log(args.num_classes)
        entropy_2 = -(soft_2 * torch.log(soft_2.clamp_min(eps))).sum(dim=1, keepdim=True) / math.log(args.num_classes)
        low_entropy = (1.0 - 0.5 * (entropy_1 + entropy_2)).clamp(0.0, 1.0)
        reliability = agreement * confidence_gate * (
            (1.0 - entropy_weight) * confidence + entropy_weight * low_entropy
        ).clamp(0.0, 1.0)

    align_losses = []
    sep_losses = []
    class_weights = []
    for class_id in range(args.num_classes):
        class_prob = (0.5 * (soft_1[:, class_id:class_id + 1] + soft_2[:, class_id:class_id + 1])).detach()
        weight = (reliability * class_prob).detach()
        class_weights.append(weight.mean())
        proto_1 = stack_early_reliable_prototype_consistency_v1_weighted_proto(out_unl_1, weight, eps)
        proto_2 = stack_early_reliable_prototype_consistency_v1_weighted_proto(out_unl_2, weight, eps)
        align_losses.append(1.0 - F.cosine_similarity(proto_1.unsqueeze(0), proto_2.unsqueeze(0), dim=1).mean())

    if args.num_classes >= 2:
        bg_weight = (reliability * 0.5 * (soft_1[:, 0:1] + soft_2[:, 0:1])).detach()
        fg_weight = (reliability * 0.5 * (soft_1[:, 1:2] + soft_2[:, 1:2])).detach()
        bg_proto_1 = stack_early_reliable_prototype_consistency_v1_weighted_proto(out_unl_1, bg_weight, eps)
        fg_proto_1 = stack_early_reliable_prototype_consistency_v1_weighted_proto(out_unl_1, fg_weight, eps)
        bg_proto_2 = stack_early_reliable_prototype_consistency_v1_weighted_proto(out_unl_2, bg_weight, eps)
        fg_proto_2 = stack_early_reliable_prototype_consistency_v1_weighted_proto(out_unl_2, fg_weight, eps)
        sep_losses.append(F.relu(F.cosine_similarity(bg_proto_1.unsqueeze(0), fg_proto_1.unsqueeze(0), dim=1).mean()))
        sep_losses.append(F.relu(F.cosine_similarity(bg_proto_2.unsqueeze(0), fg_proto_2.unsqueeze(0), dim=1).mean()))

    align_loss = torch.stack(align_losses).mean()
    sep_loss = torch.stack(sep_losses).mean() if sep_losses else out_unl_1.new_tensor(0.0)
    loss = align_loss + separation_weight * sep_loss

    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "align_loss": float(align_loss.detach().cpu()),
            "separation_loss": float(sep_loss.detach().cpu()),
            "agreement": float(agreement.mean().detach().cpu()),
            "confidence": float(confidence.mean().detach().cpu()),
            "low_entropy": float(low_entropy.mean().detach().cpu()),
            "reliability": float(reliability.mean().detach().cpu()),
            "active_ratio": float((reliability > 0).float().mean().detach().cpu()),
            "fg_weight": float(class_weights[1].detach().cpu()) if len(class_weights) > 1 else 0.0,
            "bg_weight": float(class_weights[0].detach().cpu()) if class_weights else 0.0,
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_stack_early_reliable_prototype_consistency_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).all().detach().cpu().item())
    logging.info(
        "iteration %d : stack_early_reliable_prototype_consistency_v1_enabled : True, "
        "stack_early_reliable_prototype_consistency_v1_loss : %s, "
        "stack_early_reliable_prototype_consistency_v1_ramp : %s, "
        "stack_early_reliable_prototype_consistency_v1_align_loss : %s, "
        "stack_early_reliable_prototype_consistency_v1_separation_loss : %s, "
        "stack_early_reliable_prototype_consistency_v1_agreement : %s, "
        "stack_early_reliable_prototype_consistency_v1_confidence : %s, "
        "stack_early_reliable_prototype_consistency_v1_low_entropy : %s, "
        "stack_early_reliable_prototype_consistency_v1_reliability : %s, "
        "stack_early_reliable_prototype_consistency_v1_active_ratio : %s, "
        "stack_early_reliable_prototype_consistency_v1_fg_weight : %s, "
        "stack_early_reliable_prototype_consistency_v1_bg_weight : %s, "
        "stack_early_reliable_prototype_consistency_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("align_loss", "NA"),
            stats.get("separation_loss", "NA"),
            stats.get("agreement", "NA"),
            stats.get("confidence", "NA"),
            stats.get("low_entropy", "NA"),
            stats.get("reliability", "NA"),
            stats.get("active_ratio", "NA"),
            stats.get("fg_weight", "NA"),
            stats.get("bg_weight", "NA"),
            stats.get("finite", "NA"),
            finite,
        )
    )

def stack_early_entropy_margin_sharpening_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return float(min(1.0, (iter_num - start_iter) / float(ramp_iters)))

def stack_early_entropy_margin_sharpening_v1_loss(
    out_unl_1,
    out_unl_2,
    confidence_floor,
    margin_floor,
    kl_weight,
    eps=1e-6,
):
    soft_1 = torch.softmax(out_unl_1, dim=1)
    soft_2 = torch.softmax(out_unl_2, dim=1)
    log_soft_1 = torch.log(soft_1.clamp_min(eps))
    log_soft_2 = torch.log(soft_2.clamp_min(eps))

    with torch.no_grad():
        top2_1 = torch.topk(soft_1, k=min(2, args.num_classes), dim=1).values
        top2_2 = torch.topk(soft_2, k=min(2, args.num_classes), dim=1).values
        conf_1 = top2_1[:, 0:1]
        conf_2 = top2_2[:, 0:1]
        if args.num_classes > 1:
            margin_1 = top2_1[:, 0:1] - top2_1[:, 1:2]
            margin_2 = top2_2[:, 0:1] - top2_2[:, 1:2]
        else:
            margin_1 = conf_1
            margin_2 = conf_2
        pred_1 = torch.argmax(soft_1, dim=1, keepdim=True)
        pred_2 = torch.argmax(soft_2, dim=1, keepdim=True)
        agreement = (pred_1 == pred_2).float()
        confidence = (0.5 * (conf_1 + conf_2)).clamp(0.0, 1.0)
        margin = (0.5 * (margin_1 + margin_2)).clamp(0.0, 1.0)
        pixel_weight = (
            agreement
            * (confidence >= confidence_floor).float()
            * (margin >= margin_floor).float()
            * confidence
            * margin
        ).detach()

    entropy_1 = -(soft_1 * log_soft_1).sum(dim=1, keepdim=True) / math.log(args.num_classes)
    entropy_2 = -(soft_2 * log_soft_2).sum(dim=1, keepdim=True) / math.log(args.num_classes)
    entropy_loss = ((entropy_1 + entropy_2) * 0.5 * pixel_weight).sum() / pixel_weight.sum().clamp_min(eps)

    kl_12 = (soft_1 * (log_soft_1 - log_soft_2.detach())).sum(dim=1, keepdim=True)
    kl_21 = (soft_2 * (log_soft_2 - log_soft_1.detach())).sum(dim=1, keepdim=True)
    kl_loss = ((kl_12 + kl_21) * 0.5 * pixel_weight).sum() / pixel_weight.sum().clamp_min(eps)
    loss = entropy_loss + kl_weight * kl_loss

    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "entropy_loss": float(entropy_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "agreement": float(agreement.mean().detach().cpu()),
            "confidence": float(confidence.mean().detach().cpu()),
            "margin": float(margin.mean().detach().cpu()),
            "active_ratio": float((pixel_weight > 0).float().mean().detach().cpu()),
            "pixel_weight": float(pixel_weight.mean().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_stack_early_entropy_margin_sharpening_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).all().detach().cpu().item())
    logging.info(
        "iteration %d : stack_early_entropy_margin_sharpening_v1_enabled : True, "
        "stack_early_entropy_margin_sharpening_v1_loss : %s, "
        "stack_early_entropy_margin_sharpening_v1_ramp : %s, "
        "stack_early_entropy_margin_sharpening_v1_entropy_loss : %s, "
        "stack_early_entropy_margin_sharpening_v1_kl_loss : %s, "
        "stack_early_entropy_margin_sharpening_v1_agreement : %s, "
        "stack_early_entropy_margin_sharpening_v1_confidence : %s, "
        "stack_early_entropy_margin_sharpening_v1_margin : %s, "
        "stack_early_entropy_margin_sharpening_v1_active_ratio : %s, "
        "stack_early_entropy_margin_sharpening_v1_pixel_weight : %s, "
        "stack_early_entropy_margin_sharpening_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("entropy_loss", "NA"),
            stats.get("kl_loss", "NA"),
            stats.get("agreement", "NA"),
            stats.get("confidence", "NA"),
            stats.get("margin", "NA"),
            stats.get("active_ratio", "NA"),
            stats.get("pixel_weight", "NA"),
            stats.get("finite", "NA"),
            finite,
        )
    )

def dirichlet_evidence_from_logits(logits):
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
    prob = alpha / strength
    uncertainty = (logits.shape[1] / strength).clamp(0.0, 1.0)
    reliability = (1.0 - uncertainty).clamp(0.0, 1.0)
    return prob, uncertainty, reliability, strength

def evidential_cps_lite_v1_loss(logits1, logits2, fft_logits1, fft_logits2):
    prob1, unc1, rel1, strength1 = dirichlet_evidence_from_logits(logits1)
    prob2, unc2, rel2, strength2 = dirichlet_evidence_from_logits(logits2)
    rel = (rel1 * rel2).detach()
    branch_loss = 0.5 * (
        (((prob1 - prob2.detach()) ** 2) * rel).mean()
        + (((prob2 - prob1.detach()) ** 2) * rel).mean()
    )

    fft_prob1, fft_unc1, fft_rel1, fft_strength1 = dirichlet_evidence_from_logits(fft_logits1)
    fft_prob2, fft_unc2, fft_rel2, fft_strength2 = dirichlet_evidence_from_logits(fft_logits2)
    fft_rel = (fft_rel1 * fft_rel2).detach()
    fft_loss = 0.5 * (
        (((fft_prob1 - fft_prob2.detach()) ** 2) * fft_rel).mean()
        + (((fft_prob2 - fft_prob1.detach()) ** 2) * fft_rel).mean()
    )
    loss = branch_loss + fft_loss

    with torch.no_grad():
        stats = {
            "branch_loss": float(branch_loss.detach().cpu()),
            "fft_loss": float(fft_loss.detach().cpu()),
            "unc_model1_mean": float(unc1.mean().detach().cpu()),
            "unc_model2_mean": float(unc2.mean().detach().cpu()),
            "rel_pair_mean": float(rel.mean().detach().cpu()),
            "strength_model1_mean": float(strength1.mean().detach().cpu()),
            "strength_model2_mean": float(strength2.mean().detach().cpu()),
            "fft_unc_model1_mean": float(fft_unc1.mean().detach().cpu()),
            "fft_unc_model2_mean": float(fft_unc2.mean().detach().cpu()),
            "fft_rel_pair_mean": float(fft_rel.mean().detach().cpu()),
            "fft_strength_model1_mean": float(fft_strength1.mean().detach().cpu()),
            "fft_strength_model2_mean": float(fft_strength2.mean().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu()),
        }
    return loss, stats

def log_evidential_cps_lite_v1_stats(iter_num, loss_value, stats):
    logging.info(
        "iteration %d : evidential_cps_lite_v1_enabled : True, evidential_cps_lite_v1_loss : %s, "
        "branch_loss : %s, fft_loss : %s, unc_model1_mean : %s, unc_model2_mean : %s, "
        "rel_pair_mean : %s, strength_model1_mean : %s, strength_model2_mean : %s, "
        "fft_unc_model1_mean : %s, fft_unc_model2_mean : %s, fft_rel_pair_mean : %s, "
        "fft_strength_model1_mean : %s, fft_strength_model2_mean : %s, evidential_cps_lite_v1_finite : %s" %
        (
            iter_num,
            loss_value,
            stats.get("branch_loss", "NA"),
            stats.get("fft_loss", "NA"),
            stats.get("unc_model1_mean", "NA"),
            stats.get("unc_model2_mean", "NA"),
            stats.get("rel_pair_mean", "NA"),
            stats.get("strength_model1_mean", "NA"),
            stats.get("strength_model2_mean", "NA"),
            stats.get("fft_unc_model1_mean", "NA"),
            stats.get("fft_unc_model2_mean", "NA"),
            stats.get("fft_rel_pair_mean", "NA"),
            stats.get("fft_strength_model1_mean", "NA"),
            stats.get("fft_strength_model2_mean", "NA"),
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

def cmsc_lite_v1_loss(output_soft, target, eps=1e-6):
    if output_soft.shape[1] < 2:
        zero = output_soft.sum() * 0.0
        return zero, {"finite": False, "valid_count": 0}

    prob = output_soft[:, 1]
    gt = (target > 0).float()
    batch_size, height, width = prob.shape
    device = prob.device
    dtype = prob.dtype
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, height, 1)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, width)

    pred_mass = prob.sum(dim=(1, 2)) + eps
    gt_mass = gt.sum(dim=(1, 2)) + eps
    valid = (gt_mass > 16.0).float()
    if valid.sum() < 0.5:
        zero = output_soft.sum() * 0.0
        return zero, {
            "finite": True,
            "valid_count": 0,
            "center_loss": 0.0,
            "scale_loss": 0.0,
            "area_loss": 0.0,
        }

    pred_cx = (prob * xs).sum(dim=(1, 2)) / pred_mass
    pred_cy = (prob * ys).sum(dim=(1, 2)) / pred_mass
    gt_cx = (gt * xs).sum(dim=(1, 2)) / gt_mass
    gt_cy = (gt * ys).sum(dim=(1, 2)) / gt_mass

    pred_var_x = (prob * (xs - pred_cx.view(batch_size, 1, 1)).pow(2)).sum(dim=(1, 2)) / pred_mass
    pred_var_y = (prob * (ys - pred_cy.view(batch_size, 1, 1)).pow(2)).sum(dim=(1, 2)) / pred_mass
    gt_var_x = (gt * (xs - gt_cx.view(batch_size, 1, 1)).pow(2)).sum(dim=(1, 2)) / gt_mass
    gt_var_y = (gt * (ys - gt_cy.view(batch_size, 1, 1)).pow(2)).sum(dim=(1, 2)) / gt_mass

    center_loss = (
        F.smooth_l1_loss(pred_cx, gt_cx, reduction='none') +
        F.smooth_l1_loss(pred_cy, gt_cy, reduction='none')
    )
    scale_loss = (
        F.smooth_l1_loss(torch.sqrt(pred_var_x + eps), torch.sqrt(gt_var_x + eps), reduction='none') +
        F.smooth_l1_loss(torch.sqrt(pred_var_y + eps), torch.sqrt(gt_var_y + eps), reduction='none')
    )
    area_loss = F.smooth_l1_loss(
        pred_mass / float(height * width),
        gt_mass / float(height * width),
        reduction='none',
    )
    loss = ((center_loss + scale_loss + area_loss) * valid).sum() / (valid.sum() + eps)
    stats = {
        "finite": bool(torch.isfinite(loss).detach().cpu().item()),
        "valid_count": int(valid.sum().detach().cpu().item()),
        "center_loss": float(((center_loss * valid).sum() / (valid.sum() + eps)).detach().cpu()),
        "scale_loss": float(((scale_loss * valid).sum() / (valid.sum() + eps)).detach().cpu()),
        "area_loss": float(((area_loss * valid).sum() / (valid.sum() + eps)).detach().cpu()),
    }
    return loss, stats

def log_cmsc_lite_v1_stats(iter_num, loss_value, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).detach().cpu().item())
    logging.info(
        "iteration %d : cmsc_lite_v1_enabled : True, "
        "cmsc_lite_v1_loss : %f, cmsc_center_loss : %s, cmsc_scale_loss : %s, "
        "cmsc_area_loss : %s, cmsc_valid_count : %s, cmsc_lite_v1_finite : %s" %
        (
            iter_num,
            loss_value.detach().item(),
            stats.get("center_loss", "NA"),
            stats.get("scale_loss", "NA"),
            stats.get("area_loss", "NA"),
            stats.get("valid_count", "NA"),
            str(finite),
        )
    )

def bcsi_csi_lite_v1_loss(feat1_unl, feat1_l, feat2_unl, feat2_l, top_ratio=0.25):
    feats = [feat1_unl, feat1_l, feat2_unl, feat2_l]
    ref = next((feat for feat in feats if feat is not None), None)
    if ref is None:
        zero = torch.tensor(0.0).cuda()
        return zero, {"finite": False, "reason": "missing_all_features"}
    if any(feat is None for feat in feats):
        zero = ref.sum() * 0.0
        return zero, {"finite": False, "reason": "missing_feature"}

    feat1 = torch.cat([feat1_unl, feat1_l], dim=0)
    feat2 = torch.cat([feat2_unl, feat2_l], dim=0)
    if feat1.dim() != 4 or feat2.dim() != 4:
        zero = ref.sum() * 0.0
        return zero, {"finite": False, "reason": "non_4d_feature"}

    height = min(feat1.shape[2], feat2.shape[2])
    width = min(feat1.shape[3], feat2.shape[3])
    if feat1.shape[2:] != (height, width):
        feat1 = F.interpolate(feat1, size=(height, width), mode="bilinear", align_corners=False)
    if feat2.shape[2:] != (height, width):
        feat2 = F.interpolate(feat2, size=(height, width), mode="bilinear", align_corners=False)

    channels = min(feat1.shape[1], feat2.shape[1])
    if channels <= 0:
        zero = ref.sum() * 0.0
        return zero, {"finite": False, "reason": "no_shared_channels"}
    feat1 = feat1[:, :channels]
    feat2 = feat2[:, :channels]

    ratio = max(0.0, min(float(top_ratio), 1.0))
    selected_channels = max(1, int(round(channels * ratio)))
    salience = 0.5 * (
        feat1.detach().abs().mean(dim=(2, 3)) +
        feat2.detach().abs().mean(dim=(2, 3))
    )
    selected = torch.topk(salience, k=selected_channels, dim=1).indices
    gather_index = selected.view(selected.shape[0], selected_channels, 1, 1).expand(
        selected.shape[0], selected_channels, height, width
    )
    feat1_selected = torch.gather(feat1, dim=1, index=gather_index)
    feat2_selected = torch.gather(feat2, dim=1, index=gather_index)

    def normalize_feature(feat):
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        return (feat - mean) / std

    norm1 = normalize_feature(feat1_selected)
    norm2 = normalize_feature(feat2_selected)
    loss = 0.5 * (
        F.mse_loss(norm1, norm2.detach()) +
        F.mse_loss(norm2, norm1.detach())
    )
    stats = {
        "finite": bool(torch.isfinite(loss).detach().cpu().item()),
        "selected_channel_ratio": float(selected_channels) / float(channels),
        "selected_channels": int(selected_channels),
        "shared_channels": int(channels),
        "height": int(height),
        "width": int(width),
        "feature_gap": float(torch.mean(torch.abs(norm1.detach() - norm2.detach())).detach().cpu()),
    }
    return loss, stats

def log_bcsi_csi_lite_v1_stats(iter_num, loss_value, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).detach().cpu().item())
    logging.info(
        "iteration %d : bcsi_csi_lite_v1_enabled : True, "
        "bcsi_csi_lite_v1_loss : %f, selected_channel_ratio : %s, "
        "selected_channels : %s, shared_channels : %s, feature_hw : %sx%s, "
        "feature_gap : %s, bcsi_csi_lite_v1_finite : %s" %
        (
            iter_num,
            loss_value.detach().item(),
            stats.get("selected_channel_ratio", "NA"),
            stats.get("selected_channels", "NA"),
            stats.get("shared_channels", "NA"),
            stats.get("height", "NA"),
            stats.get("width", "NA"),
            stats.get("feature_gap", "NA"),
            str(finite),
        )
    )

def hfc_fp_lite_v1_high_frequency_residual(prob, kernel_size=9):
    fg = prob[:, 1:2]
    smooth = F.avg_pool2d(
        fg,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return fg - smooth

def hfc_fp_lite_v1_pair_loss(prob_a, prob_b):
    if prob_a.shape[0] != prob_b.shape[0]:
        n = min(prob_a.shape[0], prob_b.shape[0])
        prob_a = prob_a[:n]
        prob_b = prob_b[:n]
    hf_a = hfc_fp_lite_v1_high_frequency_residual(prob_a)
    hf_b = hfc_fp_lite_v1_high_frequency_residual(prob_b).detach()
    fg_a = prob_a[:, 1:2]
    fg_b = prob_b[:, 1:2].detach()
    agreement = (1.0 - torch.abs(fg_a.detach() - fg_b.detach())).clamp(0.0, 1.0)
    return (agreement * (hf_a - hf_b).pow(2)).mean(), agreement

def hfc_fp_lite_v1_loss(logits1, logits2, fft_logits1, fft_logits2):
    prob1 = torch.softmax(logits1, dim=1)
    prob2 = torch.softmax(logits2, dim=1)
    fft_prob1 = torch.softmax(fft_logits1, dim=1)
    fft_prob2 = torch.softmax(fft_logits2, dim=1)

    branch_loss_12, q_branch_12 = hfc_fp_lite_v1_pair_loss(prob1, prob2)
    branch_loss_21, q_branch_21 = hfc_fp_lite_v1_pair_loss(prob2, prob1)
    fft_loss_12, q_fft_12 = hfc_fp_lite_v1_pair_loss(fft_prob1, fft_prob2)
    fft_loss_21, q_fft_21 = hfc_fp_lite_v1_pair_loss(fft_prob2, fft_prob1)
    cross_loss_11, q_cross_11 = hfc_fp_lite_v1_pair_loss(prob1, fft_prob1)
    cross_loss_22, q_cross_22 = hfc_fp_lite_v1_pair_loss(prob2, fft_prob2)

    branch_loss = 0.5 * (branch_loss_12 + branch_loss_21)
    fft_loss = 0.5 * (fft_loss_12 + fft_loss_21)
    cross_stream_loss = 0.5 * (cross_loss_11 + cross_loss_22)
    value = branch_loss + fft_loss + 0.5 * cross_stream_loss

    qualities = torch.cat([
        q_branch_12.detach().flatten(), q_branch_21.detach().flatten(),
        q_fft_12.detach().flatten(), q_fft_21.detach().flatten(),
        q_cross_11.detach().flatten(), q_cross_22.detach().flatten(),
    ])
    stats = {
        "branch_hf_loss": float(branch_loss.detach().cpu()),
        "fft_hf_loss": float(fft_loss.detach().cpu()),
        "cross_stream_hf_loss": float(cross_stream_loss.detach().cpu()),
        "agreement_mean": float(qualities.mean().detach().cpu()),
        "agreement_min": float(qualities.min().detach().cpu()),
        "agreement_max": float(qualities.max().detach().cpu()),
        "finite": bool(torch.isfinite(value).all().detach().cpu()),
    }
    return value, stats

def log_hfc_fp_lite_v1_stats(iter_num, loss_value, stats):
    logging.info(
        "iteration %d : hfc_fp_lite_v1_enabled : True, hfc_fp_lite_v1_loss : %s, "
        "branch_hf_loss : %s, fft_hf_loss : %s, cross_stream_hf_loss : %s, "
        "agreement_mean : %s, agreement_min : %s, agreement_max : %s, "
        "hfc_fp_lite_v1_finite : %s" %
        (
            iter_num,
            loss_value,
            stats.get("branch_hf_loss", "NA"),
            stats.get("fft_hf_loss", "NA"),
            stats.get("cross_stream_hf_loss", "NA"),
            stats.get("agreement_mean", "NA"),
            stats.get("agreement_min", "NA"),
            stats.get("agreement_max", "NA"),
            stats.get("finite", "NA"),
        )
    )

def rncl_residual_lite_v1_mixed_target_fg(first_label, second_label, mask):
    mixed = first_label.float() * mask.float() + second_label.float() * (1.0 - mask.float())
    return mixed.unsqueeze(1).clamp(0.0, 1.0)

def rncl_residual_lite_v1_loss(out_1, out_2, target_1, target_2):
    prob_1 = torch.softmax(out_1, dim=1)[:, 1:2]
    prob_2 = torch.softmax(out_2, dim=1)[:, 1:2]
    if prob_1.shape[0] != prob_2.shape[0]:
        n = min(prob_1.shape[0], prob_2.shape[0])
        prob_1 = prob_1[:n]
        prob_2 = prob_2[:n]
        target_1 = target_1[:n]
        target_2 = target_2[:n]
    residual_1 = prob_1 - target_1.detach()
    residual_2 = prob_2 - target_2.detach()
    noise_1 = residual_1.detach().abs()
    noise_2 = residual_2.detach().abs()
    reliability = (1.0 - 0.5 * (noise_1 + noise_2)).clamp(0.0, 1.0)
    residual_consistency = (reliability * (residual_1 - residual_2.detach()).pow(2)).mean()
    refinement_magnitude = 0.5 * (noise_1.mean() + noise_2.mean())
    with torch.no_grad():
        stats = {
            "residual_consistency": float(residual_consistency.detach().cpu()),
            "refinement_magnitude": float(refinement_magnitude.detach().cpu()),
            "reliability_mean": float(reliability.mean().cpu()),
            "reliability_min": float(reliability.min().cpu()),
            "reliability_max": float(reliability.max().cpu()),
            "finite": bool(torch.isfinite(residual_consistency).all().detach().cpu()),
        }
    return residual_consistency, stats

def log_rncl_residual_lite_v1_stats(iter_num, loss_value, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).detach().cpu().item())
    logging.info(
        "iteration %d : rncl_residual_lite_v1_enabled : True, "
        "rncl_residual_lite_v1_loss : %s, residual_consistency : %s, "
        "refinement_magnitude : %s, reliability_mean : %s, reliability_min : %s, "
        "reliability_max : %s, rncl_residual_lite_v1_finite : %s" %
        (
            iter_num,
            loss_value,
            stats.get("residual_consistency", "NA"),
            stats.get("refinement_magnitude", "NA"),
            stats.get("reliability_mean", "NA"),
            stats.get("reliability_min", "NA"),
            stats.get("reliability_max", "NA"),
            str(finite),
        )
    )

def ocpr_lite_v1_loss(out_soft_1, out_soft_2, confidence_ceiling):
    conf_1, pred_1 = torch.max(out_soft_1, dim=1)
    conf_2, pred_2 = torch.max(out_soft_2, dim=1)
    disagreement = (pred_1 != pred_2).float()
    excess_1 = F.relu(conf_1 - confidence_ceiling)
    excess_2 = F.relu(conf_2 - confidence_ceiling)
    penalty_map = 0.5 * (excess_1.pow(2) + excess_2.pow(2)) * disagreement
    loss = penalty_map.sum() / (disagreement.sum() + 1e-6)
    with torch.no_grad():
        excess_mean = 0.5 * (excess_1 + excess_2)
        conf_mean = 0.5 * (conf_1 + conf_2)
        stats = {
            "disagreement_ratio": float(disagreement.mean().detach().cpu()),
            "excess_confidence_mean": float(excess_mean.mean().detach().cpu()),
            "penalized_ratio": float((penalty_map.detach() > 0).float().mean().cpu()),
            "confidence_mean": float(conf_mean.mean().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu()),
        }
    return loss, stats

def log_ocpr_lite_v1_stats(iter_num, loss_value, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).detach().cpu().item())
    logging.info(
        "iteration %d : ocpr_lite_v1_enabled : True, "
        "ocpr_lite_v1_loss : %s, disagreement_ratio : %s, "
        "excess_confidence_mean : %s, penalized_ratio : %s, "
        "confidence_mean : %s, ocpr_lite_v1_finite : %s" %
        (
            iter_num,
            loss_value,
            stats.get("disagreement_ratio", "NA"),
            stats.get("excess_confidence_mean", "NA"),
            stats.get("penalized_ratio", "NA"),
            stats.get("confidence_mean", "NA"),
            str(finite),
        )
    )

def sra_seb_lite_v1_soft_target(img_label, patch_label, mask, blur_kernel):
    kernel = int(blur_kernel)
    if kernel % 2 == 0:
        kernel += 1
    hard = mask.float().unsqueeze(1)
    soft = F.avg_pool2d(hard, kernel_size=kernel, stride=1, padding=kernel // 2)
    soft = soft.clamp(0.0, 1.0)
    img_fg = img_label.float().unsqueeze(1)
    patch_fg = patch_label.float().unsqueeze(1)
    target = soft * img_fg + (1.0 - soft) * patch_fg
    seam = ((soft > 1e-4) & (soft < 1.0 - 1e-4)).float()
    return target, seam

def sra_seb_lite_v1_branch_loss(output, img_label, patch_label, mask, blur_kernel):
    out_soft = torch.softmax(output, dim=1)[:, 1:2]
    target, seam = sra_seb_lite_v1_soft_target(img_label, patch_label, mask, blur_kernel)
    denom = seam.sum().clamp_min(1.0)
    target_detached = target.detach()
    mse = (((out_soft - target_detached) ** 2) * seam).sum() / denom
    intersection = (out_soft * target_detached * seam).sum()
    dice_den = ((out_soft + target_detached) * seam).sum().clamp_min(1e-6)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (dice_den + 1e-6)
    loss = 0.5 * (mse + dice)
    with torch.no_grad():
        stats = {
            "seam_ratio": float(seam.detach().mean().cpu()),
            "soft_target_mean": float((target_detached * seam).sum().cpu() / denom.detach().cpu()),
            "pred_fg_seam_mean": float((out_soft.detach() * seam).sum().cpu() / denom.detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu()),
        }
    return loss, stats

def sra_seb_lite_v1_loss(out_unl_1, out_l_1, out_unl_2, out_l_2,
                         plab_a, lab_a, lab_b, plab_b,
                         plab_a_s, lab_a_s, lab_b_s, plab_b_s,
                         loss_mask, blur_kernel):
    branches = [
        sra_seb_lite_v1_branch_loss(out_unl_1, plab_a, lab_a, loss_mask, blur_kernel),
        sra_seb_lite_v1_branch_loss(out_l_1, lab_b, plab_b, loss_mask, blur_kernel),
        sra_seb_lite_v1_branch_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, blur_kernel),
        sra_seb_lite_v1_branch_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, blur_kernel),
    ]
    values = [item[0] for item in branches]
    stats_list = [item[1] for item in branches]
    value = 0.25 * sum(values)
    stats = {
        "seam_ratio": sum(s["seam_ratio"] for s in stats_list) / len(stats_list),
        "pred_fg_seam_mean": sum(s["pred_fg_seam_mean"] for s in stats_list) / len(stats_list),
        "soft_target_mean": sum(s["soft_target_mean"] for s in stats_list) / len(stats_list),
        "finite": all(s["finite"] for s in stats_list) and bool(torch.isfinite(value).all().detach().cpu()),
    }
    return value, stats

def log_sra_seb_lite_v1_stats(iter_num, loss_value, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).detach().cpu().item())
    logging.info(
        "iteration %d : sra_seb_lite_v1_enabled : True, "
        "sra_seb_lite_v1_loss : %s, seam_ratio : %s, "
        "pred_fg_seam_mean : %s, soft_target_mean : %s, sra_seb_lite_v1_finite : %s" %
        (
            iter_num,
            loss_value,
            stats.get("seam_ratio", "NA"),
            stats.get("pred_fg_seam_mean", "NA"),
            stats.get("soft_target_mean", "NA"),
            str(finite),
        )
    )

def plf_cip_lite_v1_fused_target(logits_1, logits_2, temperature):
    prob_1 = torch.softmax(logits_1, dim=1)
    prob_2 = torch.softmax(logits_2, dim=1)
    conf_1 = prob_1.detach().max(dim=1, keepdim=True).values
    conf_2 = prob_2.detach().max(dim=1, keepdim=True).values
    denom = (conf_1 + conf_2).clamp_min(1e-6)
    fused = (prob_1.detach() * conf_1 + prob_2.detach() * conf_2) / denom
    temp = max(float(temperature), 1e-3)
    if abs(temp - 1.0) > 1e-6:
        fused = torch.pow(fused.clamp_min(1e-6), 1.0 / temp)
        fused = fused / fused.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return fused, prob_1, prob_2, conf_1, conf_2

def plf_cip_lite_v1_pair_loss(logits_1, logits_2, temperature):
    fused, prob_1, prob_2, conf_1, conf_2 = plf_cip_lite_v1_fused_target(
        logits_1,
        logits_2,
        temperature,
    )
    log_prob_1 = torch.log_softmax(logits_1, dim=1)
    log_prob_2 = torch.log_softmax(logits_2, dim=1)
    target = fused.detach()
    loss_1 = F.kl_div(log_prob_1, target, reduction='batchmean')
    loss_2 = F.kl_div(log_prob_2, target, reduction='batchmean')
    value = 0.5 * (loss_1 + loss_2)
    with torch.no_grad():
        pred_1 = prob_1.argmax(dim=1)
        pred_2 = prob_2.argmax(dim=1)
        entropy = -(target * torch.log(target.clamp_min(1e-6))).sum(dim=1).mean()
        stats = {
            "loss": float(value.detach().cpu()),
            "agreement": float((pred_1 == pred_2).float().mean().detach().cpu()),
            "conf_model1": float(conf_1.mean().detach().cpu()),
            "conf_model2": float(conf_2.mean().detach().cpu()),
            "fused_entropy": float(entropy.detach().cpu()),
            "finite": bool(torch.isfinite(value).all().detach().cpu()),
        }
    return value, stats

def plf_cip_lite_v1_loss(logits_1, logits_2, fft_logits_1, fft_logits_2, temperature):
    main_loss, main_stats = plf_cip_lite_v1_pair_loss(logits_1, logits_2, temperature)
    fft_loss, fft_stats = plf_cip_lite_v1_pair_loss(fft_logits_1, fft_logits_2, temperature)
    value = main_loss + fft_loss
    stats = {
        "main_loss": main_stats["loss"],
        "fft_loss": fft_stats["loss"],
        "main_agreement": main_stats["agreement"],
        "fft_agreement": fft_stats["agreement"],
        "main_conf_model1": main_stats["conf_model1"],
        "main_conf_model2": main_stats["conf_model2"],
        "fft_conf_model1": fft_stats["conf_model1"],
        "fft_conf_model2": fft_stats["conf_model2"],
        "main_fused_entropy": main_stats["fused_entropy"],
        "fft_fused_entropy": fft_stats["fused_entropy"],
        "finite": (
            main_stats["finite"]
            and fft_stats["finite"]
            and bool(torch.isfinite(value).all().detach().cpu())
        ),
    }
    return value, stats

def log_plf_cip_lite_v1_stats(iter_num, loss_value, stats, total_loss):
    finite = stats.get("finite", False) and bool(torch.isfinite(total_loss).detach().cpu().item())
    logging.info(
        "iteration %d : plf_cip_lite_v1_enabled : True, "
        "plf_cip_lite_v1_loss : %s, main_loss : %s, fft_loss : %s, "
        "main_agreement : %s, fft_agreement : %s, "
        "main_conf_model1 : %s, main_conf_model2 : %s, "
        "fft_conf_model1 : %s, fft_conf_model2 : %s, "
        "main_fused_entropy : %s, fft_fused_entropy : %s, "
        "plf_cip_lite_v1_finite : %s" %
        (
            iter_num,
            loss_value,
            stats.get("main_loss", "NA"),
            stats.get("fft_loss", "NA"),
            stats.get("main_agreement", "NA"),
            stats.get("fft_agreement", "NA"),
            stats.get("main_conf_model1", "NA"),
            stats.get("main_conf_model2", "NA"),
            stats.get("fft_conf_model1", "NA"),
            stats.get("fft_conf_model2", "NA"),
            stats.get("main_fused_entropy", "NA"),
            stats.get("fft_fused_entropy", "NA"),
            str(finite),
        )
    )

def pem_lite_v1_entropy_loss(prob_tensors, eps=1e-6):
    losses = []
    entropies = []
    confidences = []
    for prob in prob_tensors:
        prob_safe = prob.clamp_min(eps)
        entropy = -(prob_safe * prob_safe.log()).sum(dim=1)
        losses.append(entropy.mean())
        entropies.append(entropy.detach().mean())
        confidences.append(prob.detach().max(dim=1)[0].mean())
    loss = torch.stack(losses).mean()
    entropy_mean = torch.stack(entropies).mean()
    confidence_mean = torch.stack(confidences).mean()
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "entropy": float(entropy_mean.detach().cpu()),
            "confidence": float(confidence_mean.detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def pem_lite_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return max(0.0, min(1.0, float(iter_num - start_iter) / float(ramp_iters)))

def log_pem_lite_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    logging.info(
        "iteration %d : pem_lite_v1_enabled : True, "
        "pem_lite_v1_loss : %s, pem_lite_v1_ramp : %s, "
        "pem_lite_v1_entropy : %s, pem_lite_v1_confidence : %s, "
        "pem_lite_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("entropy", "NA"),
            stats.get("confidence", "NA"),
            stats.get("finite", "NA"),
            bool(torch.isfinite(total_loss).all().detach().cpu().item()),
        )
    )

def ema_bcp_anchor_lite_v1_ramp(iter_num, start_iter, ramp_iters):
    if iter_num < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return max(0.0, min(1.0, float(iter_num - start_iter) / float(ramp_iters)))

def ema_bcp_anchor_lite_v1_pair_loss(student_soft, teacher_soft, conf_floor):
    conf_floor = min(max(float(conf_floor), 0.0), 1.0)
    teacher_soft = teacher_soft.detach()
    teacher_conf = teacher_soft.max(dim=1, keepdim=True)[0]
    weight = conf_floor + (1.0 - conf_floor) * teacher_conf
    diff = (student_soft - teacher_soft).pow(2).mean(dim=1, keepdim=True)
    loss = (diff * weight).sum() / weight.sum().clamp_min(1.0)
    return loss, teacher_conf, weight

def ema_bcp_anchor_lite_v1_loss(
    model_ema,
    net_input_unl_1,
    net_input_unl_2,
    out_soft_unl_1,
    out_soft_unl_2,
    conf_floor,
):
    with torch.no_grad():
        teacher_soft_1 = torch.softmax(
            model_ema(prepare_25d_input_for_model(net_input_unl_1, apply_neighbor_policy=True)),
            dim=1,
        )
        teacher_soft_2 = torch.softmax(
            model_ema(prepare_25d_input_for_model(net_input_unl_2, apply_neighbor_policy=True)),
            dim=1,
        )
    loss_1, conf_1, weight_1 = ema_bcp_anchor_lite_v1_pair_loss(
        out_soft_unl_1,
        teacher_soft_1,
        conf_floor,
    )
    loss_2, conf_2, weight_2 = ema_bcp_anchor_lite_v1_pair_loss(
        out_soft_unl_2,
        teacher_soft_2,
        conf_floor,
    )
    value = 0.5 * (loss_1 + loss_2)
    with torch.no_grad():
        stats = {
            "loss": float(value.detach().cpu()),
            "loss_model1": float(loss_1.detach().cpu()),
            "loss_model2": float(loss_2.detach().cpu()),
            "teacher_conf_model1": float(conf_1.mean().detach().cpu()),
            "teacher_conf_model2": float(conf_2.mean().detach().cpu()),
            "weight_model1": float(weight_1.mean().detach().cpu()),
            "weight_model2": float(weight_2.mean().detach().cpu()),
            "finite": bool(torch.isfinite(value).all().detach().cpu().item()),
        }
    return value, stats

def log_ema_bcp_anchor_lite_v1_stats(iter_num, loss_value, ramp, stats, total_loss):
    logging.info(
        "iteration %d : ema_bcp_anchor_lite_v1_enabled : True, "
        "ema_bcp_anchor_lite_v1_loss : %s, ema_bcp_anchor_lite_v1_ramp : %s, "
        "ema_bcp_anchor_lite_v1_loss_model1 : %s, ema_bcp_anchor_lite_v1_loss_model2 : %s, "
        "ema_bcp_anchor_lite_v1_teacher_conf_model1 : %s, "
        "ema_bcp_anchor_lite_v1_teacher_conf_model2 : %s, "
        "ema_bcp_anchor_lite_v1_weight_model1 : %s, "
        "ema_bcp_anchor_lite_v1_weight_model2 : %s, "
        "ema_bcp_anchor_lite_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            loss_value,
            ramp,
            stats.get("loss_model1", "NA"),
            stats.get("loss_model2", "NA"),
            stats.get("teacher_conf_model1", "NA"),
            stats.get("teacher_conf_model2", "NA"),
            stats.get("weight_model1", "NA"),
            stats.get("weight_model2", "NA"),
            stats.get("finite", "NA"),
            bool(torch.isfinite(total_loss).all().detach().cpu().item()),
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

def generate_adamix_bcp_lite_v1_mask(img, logits_a, logits_b, min_side=0.18, max_side=0.38):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    min_side = max(0.05, min(0.75, float(min_side)))
    max_side = max(min_side, min(0.90, float(max_side)))
    with torch.no_grad():
        conf_a = torch.softmax(logits_a.detach(), dim=1).max(dim=1)[0].mean()
        conf_b = torch.softmax(logits_b.detach(), dim=1).max(dim=1)[0].mean()
        teacher_conf = torch.clamp(0.5 * (conf_a + conf_b), 0.0, 1.0)
        side_fraction = min_side + (max_side - min_side) * teacher_conf.item()
    patch_x = max(1, min(img_x - 1, int(round(img_x * side_fraction))))
    patch_y = max(1, min(img_y - 1, int(round(img_y * side_fraction))))
    w = np.random.randint(0, max(1, img_x - patch_x))
    h = np.random.randint(0, max(1, img_y - patch_y))
    mask = torch.ones(img_x, img_y, device=img.device)
    loss_mask = torch.ones(batch_size, img_x, img_y, device=img.device)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    stats = {
        "teacher_conf": teacher_conf.detach(),
        "side_fraction": img.new_tensor(side_fraction),
        "patch_area_ratio": img.new_tensor((patch_x * patch_y) / float(img_x * img_y)),
        "finite": torch.isfinite(teacher_conf) and bool(np.isfinite(side_fraction)),
    }
    return mask.long(), loss_mask.long(), stats

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

def sba_bdl_boundary_band_from_binary(mask, kernel_size=3):
    mask = mask.float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.clamp(dilated - eroded, min=0.0, max=1.0)

def sba_bdl_v1_loss(logits, labels, eps=1e-6):
    prob_fg = torch.softmax(logits, dim=1)[:, 1:2]
    gt_boundary = sba_bdl_boundary_band_from_binary(labels > 0)
    pred_boundary = sba_bdl_boundary_band_from_binary(prob_fg)
    intersection = (pred_boundary * gt_boundary).sum(dim=(1, 2, 3))
    denominator = pred_boundary.sum(dim=(1, 2, 3)) + gt_boundary.sum(dim=(1, 2, 3)) + eps
    loss = (1.0 - (2.0 * intersection + eps) / denominator).mean()
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "gt_boundary_ratio": float(gt_boundary.mean().detach().cpu()),
            "pred_boundary_ratio": float(pred_boundary.mean().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_sba_bdl_v1_stats(iter_num, stats, total_loss):
    logging.info(
        "iteration %d : sba_bdl_v1_enabled : True, "
        "sba_bdl_v1_loss : %s, sba_gt_boundary_ratio : %s, "
        "sba_pred_boundary_ratio : %s, sba_bdl_v1_finite : %s, total_finite : %s" %
        (
            iter_num,
            stats.get("loss", "NA"),
            stats.get("gt_boundary_ratio", "NA"),
            stats.get("pred_boundary_ratio", "NA"),
            stats.get("finite", "NA"),
            bool(torch.isfinite(total_loss).all().detach().cpu().item()),
        )
    )

def astc_csab_sobel_xy(x):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3) / 8.0
    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    return grad_x, grad_y

def astc_csab_lite_v1_loss(image, pred_soft, eps=1e-6):
    image_gray = image[:, :1]
    fg_prob = pred_soft[:, 1:2]
    img_gx, img_gy = astc_csab_sobel_xy(image_gray)
    prob_gx, prob_gy = astc_csab_sobel_xy(fg_prob)
    img_mag = torch.sqrt(img_gx * img_gx + img_gy * img_gy + eps)
    prob_mag = torch.sqrt(prob_gx * prob_gx + prob_gy * prob_gy + eps)
    cosine = (img_gx * prob_gx + img_gy * prob_gy) / (img_mag * prob_mag + eps)
    align_loss = 1.0 - cosine.pow(2).clamp(0.0, 1.0)
    boundary_weight = (
        prob_mag.detach() / (prob_mag.detach().mean(dim=(2, 3), keepdim=True) + eps)
    ).clamp(0.0, 3.0)
    image_weight = (
        img_mag.detach() / (img_mag.detach().mean(dim=(2, 3), keepdim=True) + eps)
    ).clamp(0.0, 3.0)
    weight = boundary_weight * image_weight
    loss = (align_loss * weight).sum() / (weight.sum() + eps)
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "prob_grad": float(prob_mag.mean().detach().cpu()),
            "image_grad": float(img_mag.mean().detach().cpu()),
            "finite": bool(torch.isfinite(loss).all().detach().cpu().item()),
        }
    return loss, stats

def log_astc_csab_lite_v1_stats(iter_num, stats_1, stats_2, total_loss):
    loss_mean = 0.5 * (stats_1.get("loss", 0.0) + stats_2.get("loss", 0.0))
    prob_grad = 0.5 * (stats_1.get("prob_grad", 0.0) + stats_2.get("prob_grad", 0.0))
    image_grad = 0.5 * (stats_1.get("image_grad", 0.0) + stats_2.get("image_grad", 0.0))
    finite = (
        bool(stats_1.get("finite", False))
        and bool(stats_2.get("finite", False))
        and bool(torch.isfinite(total_loss).all().detach().cpu().item())
    )
    logging.info(
        "iteration %d : astc_csab_lite_v1_enabled : True, "
        "astc_csab_loss : %s, astc_prob_grad : %s, astc_img_grad : %s, "
        "astc_csab_finite : %s" %
        (iter_num, loss_mean, prob_grad, image_grad, str(finite))
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
        use_aupp_prehead_lite=args.use_aupp_prehead_lite_v1,
        aupp_prehead_strength=args.aupp_prehead_lite_v1_strength,
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
                
                adamix_bcp_lite_v1_stats = None
                if args.use_adamix_bcp_lite_v1:
                    img_mask, loss_mask, adamix_bcp_lite_v1_stats = generate_adamix_bcp_lite_v1_mask(
                        img_a,
                        pre_a,
                        pre_b,
                        args.adamix_bcp_lite_v1_min_side,
                        args.adamix_bcp_lite_v1_max_side,
                    )
                else:
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
            feat_unl_1 = getattr(model1, "last_decoder_feature", None)
            prediction_disagreement_neighbor_loss_1, prediction_disagreement_neighbor_stats_1 = prediction_disagreement_neighbor_regularization_loss(
                model1, net_input_unl_1, out_unl_1
            )
            out_l_1 = model1(prepare_25d_input_for_model(net_input_l_1, apply_neighbor_policy=False))
            feat_l_1 = getattr(model1, "last_decoder_feature", None)
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
            feat_unl_2 = getattr(model2, "last_decoder_feature", None)
            prediction_disagreement_neighbor_loss_2, prediction_disagreement_neighbor_stats_2 = prediction_disagreement_neighbor_regularization_loss(
                model2, net_input_unl_2, out_unl_2
            )
            out_l_2 = model2(prepare_25d_input_for_model(net_input_l_2, apply_neighbor_policy=False))
            feat_l_2 = getattr(model2, "last_decoder_feature", None)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_soft_2 = torch.softmax(out_2, dim=1)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False) 
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            # Model1 & Model2 Cross Pseudo Supervision
            pseudo_supervision1_base = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))  
            pseudo_supervision2_base = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1))  
            pseudo_supervision1 = pseudo_supervision1_base
            pseudo_supervision2 = pseudo_supervision2_base
            sgrs_lite_v1_stats = None
            if args.use_sgrs_lite_region_weighting_v1:
                pseudo_supervision1_weighted, sgrs_lite_v1_stats_1 = sgrs_lite_v1_weighted_pseudo_loss(
                    out_soft_1,
                    out_soft_2,
                    out_pseudo_1,
                    out_pseudo_2,
                )
                pseudo_supervision2_weighted, sgrs_lite_v1_stats_2 = sgrs_lite_v1_weighted_pseudo_loss(
                    out_soft_2,
                    out_soft_1,
                    out_pseudo_2,
                    out_pseudo_1,
                )
                sgrs_alpha = max(0.0, min(1.0, args.sgrs_lite_v1_weight))
                pseudo_supervision1 = (1.0 - sgrs_alpha) * pseudo_supervision1_base + sgrs_alpha * pseudo_supervision1_weighted
                pseudo_supervision2 = (1.0 - sgrs_alpha) * pseudo_supervision2_base + sgrs_alpha * pseudo_supervision2_weighted
                sgrs_lite_v1_stats = {
                    "loss_delta": (
                        (pseudo_supervision1 + pseudo_supervision2)
                        - (pseudo_supervision1_base + pseudo_supervision2_base)
                    ).detach(),
                    "weight_mean": 0.5 * (sgrs_lite_v1_stats_1["weight_mean"] + sgrs_lite_v1_stats_2["weight_mean"]),
                    "consistent_ratio": 0.5 * (sgrs_lite_v1_stats_1["consistent_ratio"] + sgrs_lite_v1_stats_2["consistent_ratio"]),
                    "disagreement_ratio": 0.5 * (sgrs_lite_v1_stats_1["disagreement_ratio"] + sgrs_lite_v1_stats_2["disagreement_ratio"]),
                    "low_conf_ratio": 0.5 * (sgrs_lite_v1_stats_1["low_conf_ratio"] + sgrs_lite_v1_stats_2["low_conf_ratio"]),
                    "finite": bool(sgrs_lite_v1_stats_1["finite"].item() and sgrs_lite_v1_stats_2["finite"].item()),
                }
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
            stack_hard_case_late_reweight_v1_value = outputs.new_tensor(0.0)
            stack_hard_case_late_reweight_v1_stats = {}
            stack_hard_case_late_reweight_v1_alpha = 0.0
            if args.use_stack_hard_case_late_reweight_v1:
                stack_hard_case_late_reweight_v1_alpha = stack_hard_case_late_reweight_v1_ramp(
                    iter_num,
                    args.stack_hard_case_late_reweight_v1_start_iter,
                    args.stack_hard_case_late_reweight_v1_ramp_iters,
                )
                (
                    stack_hard_case_late_reweight_v1_value,
                    stack_hard_case_late_reweight_v1_stats,
                ) = stack_hard_case_late_reweight_v1_loss(
                    outputs[:args.labeled_bs],
                    label_batch[:args.labeled_bs],
                    num_classes,
                )
                supervised_loss_ict = supervised_loss_ict + (
                    args.stack_hard_case_late_reweight_v1_weight
                    * stack_hard_case_late_reweight_v1_alpha
                    * stack_hard_case_late_reweight_v1_value
                )
            sba_bdl_loss = outputs.new_tensor(0.0)
            sba_bdl_stats = {}
            if args.use_sba_bdl_v1:
                sba_bdl_loss, sba_bdl_stats = sba_bdl_v1_loss(
                    outputs[:args.labeled_bs],
                    label_batch[:args.labeled_bs],
                )
                
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
            if args.use_sba_bdl_v1:
                loss = loss + args.sba_bdl_v1_weight * sba_bdl_loss
            cmsc_lite_v1_value = loss.new_tensor(0.0)
            cmsc_lite_v1_stats = {}
            if args.use_cmsc_lite_v1:
                cmsc_lite_v1_value, cmsc_lite_v1_stats = cmsc_lite_v1_loss(
                    outputs_soft[:args.labeled_bs],
                    label_batch[:args.labeled_bs],
                )
                loss = loss + args.cmsc_lite_v1_weight * cmsc_lite_v1_value
            bcsi_csi_lite_v1_value = loss.new_tensor(0.0)
            bcsi_csi_lite_v1_stats = {}
            if args.use_bcsi_csi_lite_v1:
                bcsi_csi_lite_v1_value, bcsi_csi_lite_v1_stats = bcsi_csi_lite_v1_loss(
                    feat_unl_1,
                    feat_l_1,
                    feat_unl_2,
                    feat_l_2,
                    args.bcsi_csi_lite_v1_top_ratio,
                )
                loss = loss + args.bcsi_csi_lite_v1_weight * bcsi_csi_lite_v1_value
            astc_csab_loss = loss.new_tensor(0.0)
            astc_csab_stats_1 = {}
            astc_csab_stats_2 = {}
            if args.use_astc_csab_lite_v1:
                astc_loss_1, astc_csab_stats_1 = astc_csab_lite_v1_loss(net_input_1, out_soft_1)
                astc_loss_2, astc_csab_stats_2 = astc_csab_lite_v1_loss(net_input_2, out_soft_2)
                astc_csab_loss = 0.5 * (astc_loss_1 + astc_loss_2)
                loss = loss + args.astc_csab_lite_weight * astc_csab_loss
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

            evidential_cps_lite_v1_value = loss.new_tensor(0.0)
            evidential_cps_lite_v1_stats = {}
            if args.use_evidential_cps_lite_v1:
                evidential_cps_lite_v1_value, evidential_cps_lite_v1_stats = evidential_cps_lite_v1_loss(
                    out_1,
                    out_2,
                    image_output_1,
                    image_output_2,
                )
                loss = loss + args.evidential_cps_lite_v1_weight * evidential_cps_lite_v1_value
            hfc_fp_lite_v1_value = loss.new_tensor(0.0)
            hfc_fp_lite_v1_stats = {}
            if args.use_hfc_fp_lite_v1:
                hfc_fp_lite_v1_value, hfc_fp_lite_v1_stats = hfc_fp_lite_v1_loss(
                    out_1,
                    out_2,
                    image_output_1,
                    image_output_2,
                )
                loss = loss + args.hfc_fp_lite_v1_weight * hfc_fp_lite_v1_value
            rncl_residual_lite_v1_value = loss.new_tensor(0.0)
            rncl_residual_lite_v1_stats = {}
            if args.use_rncl_residual_lite_v1:
                target_unl_1 = rncl_residual_lite_v1_mixed_target_fg(plab_a, lab_a, loss_mask)
                target_l_1 = rncl_residual_lite_v1_mixed_target_fg(lab_b, plab_b, loss_mask)
                target_1 = torch.cat([target_unl_1, target_l_1], dim=0)
                target_unl_2 = rncl_residual_lite_v1_mixed_target_fg(plab_a_s, lab_a_s, loss_mask)
                target_l_2 = rncl_residual_lite_v1_mixed_target_fg(lab_b_s, plab_b_s, loss_mask)
                target_2 = torch.cat([target_unl_2, target_l_2], dim=0)
                rncl_residual_lite_v1_value, rncl_residual_lite_v1_stats = rncl_residual_lite_v1_loss(
                    out_1,
                    out_2,
                    target_1,
                    target_2,
                )
                loss = loss + args.rncl_residual_lite_v1_weight * rncl_residual_lite_v1_value
            ocpr_lite_v1_value = loss.new_tensor(0.0)
            ocpr_lite_v1_stats = {}
            if args.use_ocpr_lite_v1:
                ocpr_lite_v1_value, ocpr_lite_v1_stats = ocpr_lite_v1_loss(
                    out_soft_1,
                    out_soft_2,
                    args.ocpr_lite_v1_confidence_ceiling,
                )
                loss = loss + args.ocpr_lite_v1_weight * ocpr_lite_v1_value
            sra_seb_lite_v1_value = loss.new_tensor(0.0)
            sra_seb_lite_v1_stats = {}
            if args.use_sra_seb_lite_v1:
                sra_seb_lite_v1_value, sra_seb_lite_v1_stats = sra_seb_lite_v1_loss(
                    out_unl_1,
                    out_l_1,
                    out_unl_2,
                    out_l_2,
                    plab_a,
                    lab_a,
                    lab_b,
                    plab_b,
                    plab_a_s,
                    lab_a_s,
                    lab_b_s,
                    plab_b_s,
                    loss_mask,
                    args.sra_seb_lite_v1_blur_kernel,
                )
                loss = loss + args.sra_seb_lite_v1_weight * sra_seb_lite_v1_value
            plf_cip_lite_v1_value = loss.new_tensor(0.0)
            plf_cip_lite_v1_stats = {}
            if args.use_plf_cip_lite_v1:
                plf_cip_lite_v1_value, plf_cip_lite_v1_stats = plf_cip_lite_v1_loss(
                    out_1,
                    out_2,
                    image_output_1,
                    image_output_2,
                    args.plf_cip_lite_v1_temperature,
                )
                loss = loss + args.plf_cip_lite_v1_weight * plf_cip_lite_v1_value

            pem_lite_v1_value = loss.new_tensor(0.0)
            pem_lite_v1_stats = {}
            pem_lite_v1_alpha = 0.0
            if args.use_pem_lite_v1:
                pem_lite_v1_alpha = pem_lite_v1_ramp(
                    iter_num,
                    args.pem_lite_v1_start_iter,
                    args.pem_lite_v1_ramp_iters,
                )
                pem_lite_v1_value, pem_lite_v1_stats = pem_lite_v1_entropy_loss(
                    [
                        torch.softmax(out_unl_1, dim=1),
                        torch.softmax(out_unl_2, dim=1),
                        outputs_soft[args.labeled_bs:],
                    ]
                )
                loss = loss + args.pem_lite_v1_weight * pem_lite_v1_alpha * pem_lite_v1_value

            ema_bcp_anchor_lite_v1_value = loss.new_tensor(0.0)
            ema_bcp_anchor_lite_v1_stats = {}
            ema_bcp_anchor_lite_v1_alpha = 0.0
            if args.use_ema_bcp_anchor_lite_v1:
                ema_bcp_anchor_lite_v1_alpha = ema_bcp_anchor_lite_v1_ramp(
                    iter_num,
                    args.ema_bcp_anchor_lite_v1_start_iter,
                    args.ema_bcp_anchor_lite_v1_ramp_iters,
                )
                ema_bcp_anchor_lite_v1_value, ema_bcp_anchor_lite_v1_stats = ema_bcp_anchor_lite_v1_loss(
                    ema_model,
                    net_input_unl_1,
                    net_input_unl_2,
                    torch.softmax(out_unl_1, dim=1),
                    torch.softmax(out_unl_2, dim=1),
                    args.ema_bcp_anchor_lite_v1_conf_floor,
                )
                loss = loss + (
                    args.ema_bcp_anchor_lite_v1_weight
                    * ema_bcp_anchor_lite_v1_alpha
                    * ema_bcp_anchor_lite_v1_value
                )

            stack_confidence_band_fg_calibration_v1_value = loss.new_tensor(0.0)
            stack_confidence_band_fg_calibration_v1_stats = {}
            stack_confidence_band_fg_calibration_v1_alpha = 0.0
            if args.use_stack_confidence_band_fg_calibration_v1:
                stack_confidence_band_fg_calibration_v1_alpha = stack_confidence_band_fg_calibration_v1_ramp(
                    iter_num,
                    args.stack_confidence_band_fg_calibration_v1_start_iter,
                    args.stack_confidence_band_fg_calibration_v1_ramp_iters,
                )
                (
                    stack_confidence_band_fg_calibration_v1_value,
                    stack_confidence_band_fg_calibration_v1_stats,
                ) = stack_confidence_band_fg_calibration_v1_loss(
                    out_unl_1,
                    out_unl_2,
                    args.stack_confidence_band_fg_calibration_v1_low,
                    args.stack_confidence_band_fg_calibration_v1_high,
                    args.stack_confidence_band_fg_calibration_v1_agree_margin,
                )
                loss = loss + (
                    args.stack_confidence_band_fg_calibration_v1_weight
                    * stack_confidence_band_fg_calibration_v1_alpha
                    * stack_confidence_band_fg_calibration_v1_value
                )

            stack_early_uncertainty_balanced_peer_consistency_v1_value = loss.new_tensor(0.0)
            stack_early_uncertainty_balanced_peer_consistency_v1_stats = {}
            stack_early_uncertainty_balanced_peer_consistency_v1_alpha = 0.0
            if args.use_stack_early_uncertainty_balanced_peer_consistency_v1:
                stack_early_uncertainty_balanced_peer_consistency_v1_alpha = (
                    stack_early_uncertainty_balanced_peer_consistency_v1_ramp(
                        iter_num,
                        args.stack_early_uncertainty_balanced_peer_consistency_v1_start_iter,
                        args.stack_early_uncertainty_balanced_peer_consistency_v1_ramp_iters,
                    )
                )
                (
                    stack_early_uncertainty_balanced_peer_consistency_v1_value,
                    stack_early_uncertainty_balanced_peer_consistency_v1_stats,
                ) = stack_early_uncertainty_balanced_peer_consistency_v1_loss(
                    out_unl_1,
                    out_unl_2,
                    args.stack_early_uncertainty_balanced_peer_consistency_v1_entropy_weight,
                    args.stack_early_uncertainty_balanced_peer_consistency_v1_min_pixel_weight,
                )
                loss = loss + (
                    args.stack_early_uncertainty_balanced_peer_consistency_v1_weight
                    * stack_early_uncertainty_balanced_peer_consistency_v1_alpha
                    * stack_early_uncertainty_balanced_peer_consistency_v1_value
                )

            stack_early_fg_area_range_calibration_v1_value = loss.new_tensor(0.0)
            stack_early_fg_area_range_calibration_v1_stats = {}
            stack_early_fg_area_range_calibration_v1_alpha = 0.0
            if args.use_stack_early_fg_area_range_calibration_v1:
                stack_early_fg_area_range_calibration_v1_alpha = (
                    stack_early_fg_area_range_calibration_v1_ramp(
                        iter_num,
                        args.stack_early_fg_area_range_calibration_v1_start_iter,
                        args.stack_early_fg_area_range_calibration_v1_ramp_iters,
                    )
                )
                (
                    stack_early_fg_area_range_calibration_v1_value,
                    stack_early_fg_area_range_calibration_v1_stats,
                ) = stack_early_fg_area_range_calibration_v1_loss(
                    out_unl_1,
                    out_unl_2,
                    label_batch,
                    args.labeled_bs,
                    args.stack_early_fg_area_range_calibration_v1_margin,
                    args.stack_early_fg_area_range_calibration_v1_min_area,
                    args.stack_early_fg_area_range_calibration_v1_max_area,
                )
                loss = loss + (
                    args.stack_early_fg_area_range_calibration_v1_weight
                    * stack_early_fg_area_range_calibration_v1_alpha
                    * stack_early_fg_area_range_calibration_v1_value
                )

            stack_early_class_confidence_anchor_v1_value = loss.new_tensor(0.0)
            stack_early_class_confidence_anchor_v1_stats = {}
            stack_early_class_confidence_anchor_v1_alpha = 0.0
            if args.use_stack_early_class_confidence_anchor_v1:
                stack_early_class_confidence_anchor_v1_alpha = (
                    stack_early_class_confidence_anchor_v1_ramp(
                        iter_num,
                        args.stack_early_class_confidence_anchor_v1_start_iter,
                        args.stack_early_class_confidence_anchor_v1_ramp_iters,
                    )
                )
                (
                    stack_early_class_confidence_anchor_v1_value,
                    stack_early_class_confidence_anchor_v1_stats,
                ) = stack_early_class_confidence_anchor_v1_loss(
                    out_unl_1,
                    out_unl_2,
                    outputs,
                    label_batch,
                    args.labeled_bs,
                    args.stack_early_class_confidence_anchor_v1_margin,
                    args.stack_early_class_confidence_anchor_v1_min_region_ratio,
                )
                loss = loss + (
                    args.stack_early_class_confidence_anchor_v1_weight
                    * stack_early_class_confidence_anchor_v1_alpha
                    * stack_early_class_confidence_anchor_v1_value
                )

            stack_early_soft_boundary_peer_consistency_v1_value = loss.new_tensor(0.0)
            stack_early_soft_boundary_peer_consistency_v1_stats = {}
            stack_early_soft_boundary_peer_consistency_v1_alpha = 0.0
            if args.use_stack_early_soft_boundary_peer_consistency_v1:
                stack_early_soft_boundary_peer_consistency_v1_alpha = (
                    stack_early_soft_boundary_peer_consistency_v1_ramp(
                        iter_num,
                        args.stack_early_soft_boundary_peer_consistency_v1_start_iter,
                        args.stack_early_soft_boundary_peer_consistency_v1_ramp_iters,
                    )
                )
                (
                    stack_early_soft_boundary_peer_consistency_v1_value,
                    stack_early_soft_boundary_peer_consistency_v1_stats,
                ) = stack_early_soft_boundary_peer_consistency_v1_loss(
                    out_unl_1,
                    out_unl_2,
                    args.stack_early_soft_boundary_peer_consistency_v1_confidence_floor,
                )
                loss = loss + (
                    args.stack_early_soft_boundary_peer_consistency_v1_weight
                    * stack_early_soft_boundary_peer_consistency_v1_alpha
                    * stack_early_soft_boundary_peer_consistency_v1_value
                )

            stack_early_reliable_prototype_consistency_v1_value = loss.new_tensor(0.0)
            stack_early_reliable_prototype_consistency_v1_stats = {}
            stack_early_reliable_prototype_consistency_v1_alpha = 0.0
            if args.use_stack_early_reliable_prototype_consistency_v1:
                stack_early_reliable_prototype_consistency_v1_alpha = (
                    stack_early_reliable_prototype_consistency_v1_ramp(
                        iter_num,
                        args.stack_early_reliable_prototype_consistency_v1_start_iter,
                        args.stack_early_reliable_prototype_consistency_v1_ramp_iters,
                    )
                )
                (
                    stack_early_reliable_prototype_consistency_v1_value,
                    stack_early_reliable_prototype_consistency_v1_stats,
                ) = stack_early_reliable_prototype_consistency_v1_loss(
                    out_unl_1,
                    out_unl_2,
                    args.stack_early_reliable_prototype_consistency_v1_confidence_floor,
                    args.stack_early_reliable_prototype_consistency_v1_entropy_weight,
                    args.stack_early_reliable_prototype_consistency_v1_separation_weight,
                )
                loss = loss + (
                    args.stack_early_reliable_prototype_consistency_v1_weight
                    * stack_early_reliable_prototype_consistency_v1_alpha
                    * stack_early_reliable_prototype_consistency_v1_value
                )

            stack_early_entropy_margin_sharpening_v1_value = loss.new_tensor(0.0)
            stack_early_entropy_margin_sharpening_v1_stats = {}
            stack_early_entropy_margin_sharpening_v1_alpha = 0.0
            if args.use_stack_early_entropy_margin_sharpening_v1:
                stack_early_entropy_margin_sharpening_v1_alpha = (
                    stack_early_entropy_margin_sharpening_v1_ramp(
                        iter_num,
                        args.stack_early_entropy_margin_sharpening_v1_start_iter,
                        args.stack_early_entropy_margin_sharpening_v1_ramp_iters,
                    )
                )
                (
                    stack_early_entropy_margin_sharpening_v1_value,
                    stack_early_entropy_margin_sharpening_v1_stats,
                ) = stack_early_entropy_margin_sharpening_v1_loss(
                    out_unl_1,
                    out_unl_2,
                    args.stack_early_entropy_margin_sharpening_v1_confidence_floor,
                    args.stack_early_entropy_margin_sharpening_v1_margin_floor,
                    args.stack_early_entropy_margin_sharpening_v1_kl_weight,
                )
                loss = loss + (
                    args.stack_early_entropy_margin_sharpening_v1_weight
                    * stack_early_entropy_margin_sharpening_v1_alpha
                    * stack_early_entropy_margin_sharpening_v1_value
                )

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
            if args.use_sba_bdl_v1 and iter_num > 0 and iter_num % args.sba_bdl_v1_log_interval == 0:
                log_sba_bdl_v1_stats(iter_num, sba_bdl_stats, loss)
            if args.use_astc_csab_lite_v1 and iter_num > 0 and iter_num % args.astc_csab_log_interval == 0:
                log_astc_csab_lite_v1_stats(iter_num, astc_csab_stats_1, astc_csab_stats_2, loss)
            if args.use_cmsc_lite_v1 and iter_num > 0 and iter_num % args.cmsc_lite_v1_log_interval == 0:
                log_cmsc_lite_v1_stats(iter_num, cmsc_lite_v1_value, cmsc_lite_v1_stats, loss)
            if args.use_bcsi_csi_lite_v1 and iter_num > 0 and iter_num % args.bcsi_csi_lite_v1_log_interval == 0:
                log_bcsi_csi_lite_v1_stats(iter_num, bcsi_csi_lite_v1_value, bcsi_csi_lite_v1_stats, loss)
            if (
                args.use_stack_hard_case_late_reweight_v1
                and iter_num > 0
                and iter_num % args.stack_hard_case_late_reweight_v1_log_interval == 0
            ):
                log_stack_hard_case_late_reweight_v1_stats(
                    iter_num,
                    float(stack_hard_case_late_reweight_v1_value.detach().cpu()),
                    stack_hard_case_late_reweight_v1_alpha,
                    stack_hard_case_late_reweight_v1_stats,
                    loss,
                )
            if (
                args.use_stack_confidence_band_fg_calibration_v1
                and iter_num > 0
                and iter_num % args.stack_confidence_band_fg_calibration_v1_log_interval == 0
            ):
                log_stack_confidence_band_fg_calibration_v1_stats(
                    iter_num,
                    float(stack_confidence_band_fg_calibration_v1_value.detach().cpu()),
                    stack_confidence_band_fg_calibration_v1_alpha,
                    stack_confidence_band_fg_calibration_v1_stats,
                    loss,
                )
            if (
                args.use_stack_early_uncertainty_balanced_peer_consistency_v1
                and iter_num > 0
                and iter_num % args.stack_early_uncertainty_balanced_peer_consistency_v1_log_interval == 0
            ):
                log_stack_early_uncertainty_balanced_peer_consistency_v1_stats(
                    iter_num,
                    float(stack_early_uncertainty_balanced_peer_consistency_v1_value.detach().cpu()),
                    stack_early_uncertainty_balanced_peer_consistency_v1_alpha,
                    stack_early_uncertainty_balanced_peer_consistency_v1_stats,
                    loss,
                )
            if (
                args.use_stack_early_fg_area_range_calibration_v1
                and iter_num > 0
                and iter_num % args.stack_early_fg_area_range_calibration_v1_log_interval == 0
            ):
                log_stack_early_fg_area_range_calibration_v1_stats(
                    iter_num,
                    float(stack_early_fg_area_range_calibration_v1_value.detach().cpu()),
                    stack_early_fg_area_range_calibration_v1_alpha,
                    stack_early_fg_area_range_calibration_v1_stats,
                    loss,
                )
            if (
                args.use_stack_early_class_confidence_anchor_v1
                and iter_num > 0
                and iter_num % args.stack_early_class_confidence_anchor_v1_log_interval == 0
            ):
                log_stack_early_class_confidence_anchor_v1_stats(
                    iter_num,
                    float(stack_early_class_confidence_anchor_v1_value.detach().cpu()),
                    stack_early_class_confidence_anchor_v1_alpha,
                    stack_early_class_confidence_anchor_v1_stats,
                    loss,
                )
            if (
                args.use_stack_early_soft_boundary_peer_consistency_v1
                and iter_num > 0
                and iter_num % args.stack_early_soft_boundary_peer_consistency_v1_log_interval == 0
            ):
                log_stack_early_soft_boundary_peer_consistency_v1_stats(
                    iter_num,
                    float(stack_early_soft_boundary_peer_consistency_v1_value.detach().cpu()),
                    stack_early_soft_boundary_peer_consistency_v1_alpha,
                    stack_early_soft_boundary_peer_consistency_v1_stats,
                    loss,
                )
            if (
                args.use_stack_early_reliable_prototype_consistency_v1
                and iter_num > 0
                and iter_num % args.stack_early_reliable_prototype_consistency_v1_log_interval == 0
            ):
                log_stack_early_reliable_prototype_consistency_v1_stats(
                    iter_num,
                    float(stack_early_reliable_prototype_consistency_v1_value.detach().cpu()),
                    stack_early_reliable_prototype_consistency_v1_alpha,
                    stack_early_reliable_prototype_consistency_v1_stats,
                    loss,
                )
            if (
                args.use_stack_early_entropy_margin_sharpening_v1
                and iter_num > 0
                and iter_num % args.stack_early_entropy_margin_sharpening_v1_log_interval == 0
            ):
                log_stack_early_entropy_margin_sharpening_v1_stats(
                    iter_num,
                    float(stack_early_entropy_margin_sharpening_v1_value.detach().cpu()),
                    stack_early_entropy_margin_sharpening_v1_alpha,
                    stack_early_entropy_margin_sharpening_v1_stats,
                    loss,
                )
            if (
                args.use_evidential_cps_lite_v1
                and iter_num > 0
                and iter_num % args.evidential_cps_lite_v1_log_interval == 0
            ):
                log_evidential_cps_lite_v1_stats(
                    iter_num,
                    float(evidential_cps_lite_v1_value.detach().cpu()),
                    evidential_cps_lite_v1_stats,
                )
            if (
                args.use_hfc_fp_lite_v1
                and iter_num > 0
                and iter_num % args.hfc_fp_lite_v1_log_interval == 0
            ):
                log_hfc_fp_lite_v1_stats(
                    iter_num,
                    float(hfc_fp_lite_v1_value.detach().cpu()),
                    hfc_fp_lite_v1_stats,
                )
            if (
                args.use_rncl_residual_lite_v1
                and iter_num > 0
                and iter_num % args.rncl_residual_lite_v1_log_interval == 0
            ):
                log_rncl_residual_lite_v1_stats(
                    iter_num,
                    float(rncl_residual_lite_v1_value.detach().cpu()),
                    rncl_residual_lite_v1_stats,
                    loss,
                )
            if (
                args.use_ocpr_lite_v1
                and iter_num > 0
                and iter_num % args.ocpr_lite_v1_log_interval == 0
            ):
                log_ocpr_lite_v1_stats(
                    iter_num,
                    float(ocpr_lite_v1_value.detach().cpu()),
                    ocpr_lite_v1_stats,
                    loss,
                )
            if (
                args.use_sra_seb_lite_v1
                and iter_num > 0
                and iter_num % args.sra_seb_lite_v1_log_interval == 0
            ):
                log_sra_seb_lite_v1_stats(
                    iter_num,
                    float(sra_seb_lite_v1_value.detach().cpu()),
                    sra_seb_lite_v1_stats,
                    loss,
                )
            if (
                args.use_plf_cip_lite_v1
                and iter_num > 0
                and iter_num % args.plf_cip_lite_v1_log_interval == 0
            ):
                log_plf_cip_lite_v1_stats(
                    iter_num,
                    float(plf_cip_lite_v1_value.detach().cpu()),
                    plf_cip_lite_v1_stats,
                    loss,
                )
            if (
                args.use_pem_lite_v1
                and iter_num > 0
                and iter_num % args.pem_lite_v1_log_interval == 0
            ):
                log_pem_lite_v1_stats(
                    iter_num,
                    float(pem_lite_v1_value.detach().cpu()),
                    pem_lite_v1_alpha,
                    pem_lite_v1_stats,
                    loss,
                )
            if (
                args.use_ema_bcp_anchor_lite_v1
                and iter_num > 0
                and iter_num % args.ema_bcp_anchor_lite_v1_log_interval == 0
            ):
                log_ema_bcp_anchor_lite_v1_stats(
                    iter_num,
                    float(ema_bcp_anchor_lite_v1_value.detach().cpu()),
                    ema_bcp_anchor_lite_v1_alpha,
                    ema_bcp_anchor_lite_v1_stats,
                    loss,
                )

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
            if (
                args.use_adamix_bcp_lite_v1
                and adamix_bcp_lite_v1_stats is not None
                and iter_num % args.adamix_bcp_lite_v1_log_interval == 0
            ):
                logging.info(
                    'iteration %d : adamix_bcp_lite_v1_enabled : True, '
                    'teacher_conf : %f, side_fraction : %f, patch_area_ratio : %f, '
                    'adamix_bcp_lite_v1_finite : %s' %
                    (
                        iter_num,
                        adamix_bcp_lite_v1_stats["teacher_conf"].detach().item(),
                        adamix_bcp_lite_v1_stats["side_fraction"].detach().item(),
                        adamix_bcp_lite_v1_stats["patch_area_ratio"].detach().item(),
                        str(adamix_bcp_lite_v1_stats["finite"]),
                    )
                )
            if args.use_aupp_prehead_lite_v1 and iter_num % args.aupp_prehead_lite_v1_log_interval == 0:
                log_aupp_prehead_lite_v1_stats(model1, "model1", iter_num)
                log_aupp_prehead_lite_v1_stats(model2, "model2", iter_num)
            if (
                args.use_sgrs_lite_region_weighting_v1
                and iter_num % args.sgrs_lite_v1_log_interval == 0
                and sgrs_lite_v1_stats is not None
            ):
                log_sgrs_lite_v1_stats(iter_num, sgrs_lite_v1_stats)
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
