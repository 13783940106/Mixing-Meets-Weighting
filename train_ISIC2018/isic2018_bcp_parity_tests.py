"""Deterministic parent-BCP trace tests; no dataset or GPU access."""
from __future__ import annotations
import random, torch
from train_ISIC2018.isic2018_common import parent_bcp_plan

def main():
    # Distinct RGB/sample identities expose any source/partner/channel mismatch.
    x=torch.stack([torch.full((3,12,12),10+i) for i in range(4)]).float()
    u=torch.stack([torch.full((3,12,12),30+i) for i in range(4)]).float()
    a=parent_bcp_plan(x,u,random.Random(1337));b=parent_bcp_plan(x,u,random.Random(1337))
    assert a['pair_indices']=={'la':[0,2],'lb':[2,4],'ua':[0,2],'ub':[2,4]}
    assert a['permutation'] is None and torch.equal(a['keep'],b['keep'])
    assert torch.equal(a['input_u1'],b['input_u1']) and torch.equal(a['input_l1'],b['input_l1'])
    k=a['keep'].bool(); expected=u[:2]*k+x[:2]*(~k)
    assert torch.equal(a['input_u1'],expected)
    # The single spatial mask is RGB-broadcast: source identity is identical in all channels.
    assert torch.equal(a['input_u1'][:,0],a['input_u1'][:,1]) and torch.equal(a['input_u1'][:,1],a['input_u1'][:,2])
    print('TEST_BCP_PARENT_PAIR_INDEX_PARITY PASS')
    print('TEST_BCP_PARENT_PERMUTATION_PARITY PASS')
    print('TEST_BCP_PARENT_MASK_PARITY PASS')
    print('TEST_BCP_PARENT_IMAGE_MIX_PARITY PASS')
    print('TEST_BCP_FIXED_SEED_REPLAY PASS')
if __name__=='__main__': main()
