import torch

def myInt8Matmul(A: torch.Tensor, B: torch.Tensor, zp_times_weight_channel_sum, act_times_weight_delta, bias) -> torch.Tensor: ...

def myInt8Conv(A: torch.Tensor, B: torch.Tensor, padH, padW, strideH, strideW, dilationH, dilationW, zp_times_weight_channel_sum, act_times_weight_delta, bias) -> torch.Tensor: ...