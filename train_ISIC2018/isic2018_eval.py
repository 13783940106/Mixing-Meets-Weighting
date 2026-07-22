import torch
def mean_metrics(pred, target):
    p, t = pred.bool(), target.bool(); vals=[]
    for x,y in zip(p,t):
        inter=(x&y).sum().item(); union=(x|y).sum().item(); denom=x.sum().item()+y.sum().item()
        vals.append((1. if denom==0 else 2*inter/denom,1. if union==0 else inter/union))
    return tuple(torch.tensor(vals).mean(0).tolist())
