import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

def calculate_metric_percase_promise(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        iou = _compute_iou(pred, gt)
        return dice, hd95, iou
    else:
        return 0, 0, 0


def _compute_iou(pred, gt):
    """Compute Jaccard / IoU from binary prediction and ground truth.
    Returns 0.0 when union is zero.
    """
    intersection = float((pred & gt).sum())
    union = float((pred | gt).sum())
    if union == 0:
        return 0.0
    return intersection / union
    
def test_single_volume_promise(image, label, net, classes):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_promise(prediction == i, label == i))
    return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256], input_mode="2d"):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    D = image.shape[0]
    prediction = np.zeros_like(label)
    for ind in range(D):
        center_slice = image[ind, :, :]
        x, y = center_slice.shape[0], center_slice.shape[1]
        center_slice = zoom(center_slice, (patch_size[0] / x, patch_size[1] / y), order=0)

        if input_mode == "25d_3ch":
            prev_idx = ind - 1 if ind > 0 else ind
            next_idx = ind + 1 if ind < D - 1 else ind
            prev_slice = zoom(image[prev_idx, :, :], (patch_size[0] / x, patch_size[1] / y), order=0)
            next_slice = zoom(image[next_idx, :, :], (patch_size[0] / x, patch_size[1] / y), order=0)
            input_t = torch.from_numpy(np.stack([prev_slice, center_slice, next_slice], axis=0)).unsqueeze(0).float().cuda()
            if input_mode == "25d_neighbor_relation":
                prev_valid = 0.0 if prev_idx == ind else 1.0
                next_valid = 0.0 if next_idx == ind else 1.0
                validity = torch.tensor([[[[prev_valid]], [[next_valid]]]], dtype=torch.float32).cuda()
                validity = validity.expand(-1, -1, patch_size[0], patch_size[1])
            else:
                validity = None
        else:
            input_t = torch.from_numpy(center_slice).unsqueeze(0).unsqueeze(0).float().cuda()
            validity = None

        net.eval()
        with torch.no_grad():
            out = net(input_t, neighbor_validity=validity) if validity is not None else net(input_t)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
