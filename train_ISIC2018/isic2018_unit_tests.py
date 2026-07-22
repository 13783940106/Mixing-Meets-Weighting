import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from isic2018_eval import mean_metrics
from isic2018_common import bcp_rgb,beta_fft_rgb,deterministic_cross_entropy
x=torch.zeros(2,3,8,8);y=torch.ones_like(x);m=torch.zeros(2,1,8,8,dtype=torch.bool);m[:,:,2:,:]=1
z=bcp_rgb(x,y,m);assert (z[:,:,2:,:]==1).all() and (z[:,:,:2,:]==0).all() and beta_fft_rgb(x).shape==x.shape
p=torch.tensor([[[1,0],[0,1]],[[0,0],[0,0]]]);t=p.clone();d,i=mean_metrics(p,t);assert d==1 and i==1
print('UNIT_TESTS_PASS')
