"""No-training ISIC design and access-guard preflight."""
from pathlib import Path
import sys, torch
sys.path.insert(0,str(Path(__file__).parent.parent))
from train_ISIC2018.isic2018_dataset import ISICDataset, counter_snapshot, reset_counters
from train_ISIC2018.isic2018_models import make_model
from train_ISIC2018.isic2018_common import beta_fft_rgb
root=Path('/home/renguanming/datasets/ISIC2018');split=root/'splits/isic2018_official_5p10p_seed1337_v1'; reset_counters()
l=ISICDataset(split/'isic2018_split_manifest.csv',split/'train_5p_labeled_ids.txt','labeled');u=ISICDataset(split/'isic2018_split_manifest.csv',split/'train_5p_unlabeled_ids.txt','unlabeled')
a,b=l[0],u[0];model=make_model(torch.device('cpu')).eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
with torch.no_grad(): logits=model(beta_fft_rgb(a['image'][None]));assert logits.shape==(1,2,256,256) and torch.isfinite(logits).all()
assert a['target'].dtype==torch.long and set(a['target'].unique().tolist())<=set([0,1]) and 'target' not in b
assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
assert all(v==0 for v in counter_snapshot().values())
print('S05_STATELESS_PREFLIGHT_PASS',counter_snapshot())
