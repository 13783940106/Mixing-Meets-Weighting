from train_MSCMRSEG.networks.unet import UNet
def make_model(device):
    return UNet(in_chns=3, class_num=2).to(device)
