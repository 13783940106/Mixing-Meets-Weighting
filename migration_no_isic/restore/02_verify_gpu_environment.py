#!/usr/bin/env python3
import torch, platform
print('python', platform.python_version())
print('torch', torch.__version__, 'cuda_runtime', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
    x=torch.randn(2,3,device='cuda'); print('cuda_tensor_ok', float(x.sum().cpu()))
