import torch
print(torch.version.cuda)  # 检查 PyTorch 是否包含 CUDA 支持
print(torch.cuda.is_available())  # 检查 CUDA 是否可用