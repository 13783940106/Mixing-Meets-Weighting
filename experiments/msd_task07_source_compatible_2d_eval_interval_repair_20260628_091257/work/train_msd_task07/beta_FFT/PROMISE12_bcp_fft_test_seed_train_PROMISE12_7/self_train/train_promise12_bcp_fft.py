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
from dataloaders.promise12 import Promise12
from dataloaders.msd_task07 import MSDTask07
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.config import get_config
from utils import losses, ramps
from val_2D import test_single_volume_promise, test_single_volume_msd
from utils.displacement import process_image_batches
import torch.nn as nn
import torch.nn.functional as F

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
parser.add_argument('--dataset', type=str, default='promise12', choices=['promise12', 'msd_task07'],
                    help='dataset adapter')
parser.add_argument('--train_split', type=str, default=None, help='MSD train split tsv')
parser.add_argument('--val_split', type=str, default=None, help='MSD validation split tsv')
parser.add_argument('--test_split', type=str, default=None, help='MSD internal-test split tsv')
parser.add_argument('--label_budget', type=str, default=None, help='MSD train label budget tsv')
parser.add_argument('--run_root', type=str, default=None, help='absolute run root for source-compatible execution')
parser.add_argument('--pretrain_max_iterations_override', type=int, default=None,
                    help='engineering smoke only; leave unset for full')
parser.add_argument('--self_train_max_iterations_override', type=int, default=None,
                    help='engineering smoke only; leave unset for full')
parser.add_argument('--loader_smoke_only', action='store_true',
                    help='audit dataloader contract and exit without training')
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
args = parser.parse_args()
config = get_config(args)
dice_loss = losses.DiceLoss(args.num_classes)

def build_dataset(mode):
    if args.dataset == 'msd_task07':
        return MSDTask07(
            args.root_path,
            mode=mode,
            out_size=args.image_size[0],
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            label_budget=args.label_budget,
        )
    return Promise12(args.root_path, mode=mode, out_size=args.image_size[0])

def evaluate_volume(sampled_batch, model, num_classes):
    if args.dataset == 'msd_task07':
        return test_single_volume_msd(
            sampled_batch["image"],
            sampled_batch["mask"],
            model,
            classes=num_classes,
            patch_size=tuple(args.image_size),
            spacing=sampled_batch.get("spacing"),
        )
    return test_single_volume_promise(sampled_batch["image"], sampled_batch["mask"], model, classes=num_classes)

def metric_primary(metric_list):
    arr = np.array(metric_list, dtype=np.float64)
    if args.dataset == 'msd_task07':
        dice_values = arr[:, 0] if arr.ndim == 2 else arr
        return float(np.nanmean(dice_values))
    return float(np.mean(arr, axis=0))

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

    db_train = build_dataset('train')
    db_val = build_dataset('val')
    
    total_slices = len(db_train)
    labeled_slice = getattr(db_train, 'labeled_slice_count', 202) # labeled slice count
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
            out_mixl = model(net_input)
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



            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = evaluate_volume(sampled_batch, model, num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)



                performance = metric_primary(metric_list)

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
    writer.close()

def create_model(ema=False):
    # Network definition
    model = net_factory(net_type=args.model_1, in_chns=1, class_num=args.num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model
def self_train(args, pre_snapshot_path,snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model_1))


    model1 = create_model()
    model2 = create_model()
    ema_model = create_model()

    # model2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()
    # model2.load_from(config)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = build_dataset('train')
    db_val = build_dataset('val')

    total_slices = len(db_train)
    labeled_slice = getattr(db_train, 'labeled_slice_count', 202)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)  # args.labeled_bs=8
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
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
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                pre_a_s = ema_model(uimg_a_s)
                pre_b_s = ema_model(uimg_b_s)
                
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
            out_unl_1 = model1(net_input_unl_1)
            out_l_1 = model1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            out_soft_1 = torch.softmax(out_1, dim=1)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2 = model2(net_input_unl_2)
            out_l_2 = model2(net_input_l_2)
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
            outputs = model1(input_volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(
                    ema_model(unlabeled_volume_batch_0), dim=1)
                ema_output_ux1 = torch.softmax(
                    ema_model(unlabeled_volume_batch_1), dim=1)
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

            image_patch_last = process_image_batches(net_input_1, net_input_2,30)
            image_output_1 = model1(image_patch_last.unsqueeze(1))  
            image_output_soft_1 = torch.softmax(image_output_1, dim=1)
            pseudo_image_output_1 = torch.argmax(image_output_soft_1.detach(), dim=1, keepdim=False)
            image_output_2 = model2(image_patch_last.unsqueeze(1))
            image_output_soft_2 = torch.softmax(image_output_2, dim=1)
            pseudo_image_output_2 = torch.argmax(image_output_soft_2.detach(), dim=1, keepdim=False)
            # Model1 & Model2 Second Step Cross Pseudo Supervision
            pseudo_supervision3 = dice_loss(image_output_soft_1, pseudo_image_output_2.unsqueeze(1))
            pseudo_supervision4 = dice_loss(image_output_soft_2, pseudo_image_output_1.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision3
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision4
            # loss = loss_1 + loss_2
            loss = loss_1 + loss_2 + consistency_loss +supervised_loss_ict 

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
            
            # logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
             
            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(loader):
                    metric_i = evaluate_volume(sampled_batch, model1, num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance1 = metric_primary(metric_list)
                writer.add_scalar('info/model1_val_primary_dice', performance1, iter_num)
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model_1))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info('iteration %d : model1_mean_dice : %f' % (iter_num, performance1))
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
    if args.run_root:
        pre_snapshot_path = os.path.join(args.run_root, "pre_train")
        snapshot_path = os.path.join(args.run_root, "self_train")
    else:
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
    if args.loader_smoke_only:
        for mode in ['train', 'val', 'test']:
            dataset = build_dataset(mode)
            print('LOADER_SMOKE', mode, 'len', len(dataset), 'labeled_slice_count', getattr(dataset, 'labeled_slice_count', 'NA'))
            if args.dataset == 'msd_task07' and mode == 'test':
                print(' ', 'internal_test_label_access', 'NO')
                print(' ', 'first_case_id', dataset.case_ids[0] if dataset.case_ids else 'NA')
                continue
            sample = dataset[0]
            for key, value in sample.items():
                if torch.is_tensor(value):
                    print(' ', key, tuple(value.shape), value.dtype, 'min', value.min().item() if value.numel() else 'NA', 'max', value.max().item() if value.numel() else 'NA')
                else:
                    print(' ', key, value)
        raise SystemExit(0)

    pre_train(args, pre_snapshot_path)


    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args,pre_snapshot_path, snapshot_path)
