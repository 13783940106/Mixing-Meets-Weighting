from networks.unet import UNet, UNet_2d
import torch.nn as nn

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0,
                use_mask_token_prehead_router=False, mask_token_num=4, mask_token_dim=16,
                use_aupp_prehead_lite=False, aupp_prehead_strength=0.05):
    if net_type == "unet" and mode == "train":
        net = UNet(
            in_chns=in_chns,
            class_num=class_num,
            use_mask_token_prehead_router=use_mask_token_prehead_router,
            mask_token_num=mask_token_num,
            mask_token_dim=mask_token_dim,
            use_aupp_prehead_lite=use_aupp_prehead_lite,
            aupp_prehead_strength=aupp_prehead_strength,
        ).cuda()
    return net

def BCP_net(in_chns=1, class_num=4, ema=False, use_mask_token_prehead_router=False,
            mask_token_num=4, mask_token_dim=16,
            use_aupp_prehead_lite=False, aupp_prehead_strength=0.05):
    net = UNet_2d(
        in_chns=in_chns,
        class_num=class_num,
        use_mask_token_prehead_router=use_mask_token_prehead_router,
        mask_token_num=mask_token_num,
        mask_token_dim=mask_token_dim,
        use_aupp_prehead_lite=use_aupp_prehead_lite,
        aupp_prehead_strength=aupp_prehead_strength,
    ).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net
