from __future__ import annotations
import hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def canonical_sha(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def beta_fft_rgb(x):
    # Per-channel spatial beta-FFT low-frequency exchange is performed in train core.
    return torch.fft.ifft2(torch.fft.fft2(x,dim=(-2,-1)),dim=(-2,-1)).real
def bcp_rgb(x, other, mask):
    """RGB broadcast helper retained for legacy/basic shape tests."""
    return torch.where(mask.bool(), other, x)
def fft_swap(a,b,radius=30):
    h,w=a.shape[-2:]; r=min(radius,h//2-1,w//2-1); mask=torch.zeros((h,w),device=a.device); mask[h//2-r:h//2+r,w//2-r:w//2+r]=1
    def low(x):
        s=torch.fft.fftshift(torch.fft.fft2(x,dim=(-2,-1)),dim=(-2,-1))
        return torch.fft.ifft2(torch.fft.ifftshift(s*mask,dim=(-2,-1)),dim=(-2,-1)).abs()
    la,lb=low(a),low(b); return torch.cat((a-la+lb,b-lb+la))
def dice_loss(logits,target):
    p=torch.softmax(logits,1); t=F.one_hot(target.long(),2).permute(0,3,1,2).float(); return 1-(2*(p*t).sum((0,2,3))+1e-6).div((p.square()+t.square()).sum((0,2,3))+1e-6).mean()
def deterministic_cross_entropy(logits, target, *, weight=None, ignore_index=-100, reduction="mean"):
    """CrossEntropyLoss-compatible hard-target backend without CUDA NLLLoss2d.

    This is deliberately limited to the hard-label semantics used by the
    frozen ISIC implementation.  In particular, the mean denominator follows
    PyTorch: it is the sum of selected class weights (or valid-pixel count),
    rather than the total number of pixels.  Keeping this primitive here makes
    the strict-determinism path identical for Baseline and Candidate.
    """
    if logits.ndim < 2 or target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("cross entropy expects logits [N,C,...] and target [N,...]")
    target = target.long()
    valid = target.ne(ignore_index)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    if safe_target.numel() and (safe_target.min() < 0 or safe_target.max() >= logits.shape[1]):
        raise ValueError("cross entropy target outside class range")
    negative_log_probability = -F.log_softmax(logits, dim=1).gather(1, safe_target.unsqueeze(1)).squeeze(1)
    if weight is None:
        pixel_weight = torch.ones_like(negative_log_probability)
    else:
        pixel_weight = weight.to(device=logits.device, dtype=logits.dtype).gather(0, safe_target.reshape(-1)).reshape_as(safe_target)
    terms = negative_log_probability * pixel_weight * valid.to(logits.dtype)
    if reduction == "none": return terms
    total = terms.sum()
    if reduction == "sum": return total
    if reduction != "mean": raise ValueError("unsupported cross entropy reduction")
    denominator = (pixel_weight * valid.to(logits.dtype)).sum()
    # Matches PyTorch's finite, zero-gradient all-ignore behavior.
    return total * 0.0 if denominator.item() == 0 else total / denominator
def supervised_loss(logits,target): return .5*(deterministic_cross_entropy(logits,target)+dice_loss(logits,target))
def mixed_supervised_loss(logits, target_a, target_b, keep_mask, weight_a=1.0, weight_b=1.0):
    """Parent BCP mixed-loss algebra, with the deterministic CE backend."""
    keep_mask = keep_mask.to(device=logits.device, dtype=logits.dtype)
    patch_mask = 1.0 - keep_mask
    ce_map_a = deterministic_cross_entropy(logits, target_a, reduction="none")
    ce_map_b = deterministic_cross_entropy(logits, target_b, reduction="none")
    ce = weight_a * (ce_map_a * keep_mask).sum() / keep_mask.sum().clamp_min(1)
    ce = ce + weight_b * (ce_map_b * patch_mask).sum() / patch_mask.sum().clamp_min(1)
    mixed_target = torch.where(keep_mask.bool(), target_a.long(), target_b.long())
    return .5 * (ce + dice_loss(logits, mixed_target)), ce, dice_loss(logits, mixed_target)
def parent_bcp_plan(labeled, unlabeled, rng):
    """Exact V15.1 BCP branch construction, RGB-broadcast only.

    Parent order is `[la, lb, ua, ub]`; there is no permutation.  `keep` is
    one outside and zero inside the same rectangular patch for every sample.
    Returned target roles are deliberately explicit for parity tests.
    """
    if labeled.ndim != 4 or unlabeled.ndim != 4 or labeled.shape != unlabeled.shape:
        raise ValueError('parent BCP requires matched BCHW paired streams')
    b, c, h, w = labeled.shape
    if b < 2 or b % 2: raise ValueError('parent BCP requires even half-batch')
    patch_h, patch_w = int(h * (2/3)), int(w * (2/3))
    top, left = int(rng.randint(0, max(1,h-patch_h))), int(rng.randint(0,max(1,w-patch_w)))
    keep = torch.ones((b//2,1,h,w), dtype=labeled.dtype, device=labeled.device)
    keep[:,:,top:top+patch_h,left:left+patch_w] = 0
    lh=b//2; la,lb=labeled[:lh],labeled[lh:];ua,ub=unlabeled[:lh],unlabeled[lh:]
    return {'pair_indices': {'la':[0,lh], 'lb':[lh,b], 'ua':[0,lh], 'ub':[lh,b]}, 'permutation': None,
            'keep':keep, 'input_u1':ua*keep+la*(1-keep), 'input_l1':lb*keep+ub*(1-keep),
            'input_u2':None, 'input_l2':None, 'rectangle':{'top':top,'left':left,'height':patch_h,'width':patch_w}}
def capture_rng(): return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if s.get('cuda') is not None: torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
def atomic_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.tmp.{os.getpid()}');torch.save(payload,tmp); loaded=torch.load(tmp,map_location='cpu',weights_only=False)
    if not isinstance(loaded,dict) or 'model1' not in loaded: raise RuntimeError('checkpoint integrity failure')
    os.replace(tmp,path); return sha256_file(path)
