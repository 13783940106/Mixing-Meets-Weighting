#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, sys
try: import torch
except Exception as e: torch=None; print('torch import failed',e)
root=Path(__file__).resolve().parents[2]
inv=root/'migration_full_beta_fft'/'CHECKPOINT_INVENTORY.tsv'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
checked=0; bad=[]
for r in csv.DictReader(inv.open(),delimiter='\t'):
 p=root/r['checkpoint_path']; checked+=1
 if not p.exists(): bad.append((str(p),'missing')); continue
 if r['sha256'] and not r['sha256'].startswith('SHA_ERROR') and sha(p)!=r['sha256']: bad.append((str(p),'sha mismatch')); continue
 if torch is not None and checked<=20:
  try:
   z=torch.load(p,map_location='cpu',weights_only=False)
   required=['model1','optimizer1','scheduler1','global_optimizer_update']
   print(p, 'load_ok', isinstance(z,dict), {k:(k in z) for k in required})
  except Exception as e: bad.append((str(p),'torch_load_error:'+repr(e)))
print({'checked':checked,'bad_count':len(bad),'first_bad':bad[:10]})
sys.exit(1 if bad else 0)
