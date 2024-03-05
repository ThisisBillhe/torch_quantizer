import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_quantizer
from torch_quantizer.src.quant_layer import qlinear_8bit_Linherit, qconv2d_8bit_Cinherit

def calculate_channelwise_quant_params(weight_matrix, n_bits=8):
    # Assuming each row is a channel
    min_vals = weight_matrix.min(dim=1).values
    max_vals = weight_matrix.max(dim=1).values

    # Calculate delta (scale) and zero-point for each channel
    deltas = (max_vals - min_vals) / (2 ** n_bits - 1)
    zero_points = -min_vals / deltas
    zero_points = zero_points.round().clamp(0, 2 ** n_bits - 1)

    return deltas, zero_points

def calculate_channelwise_symmetric_scale(weight_matrix, n_levels=256):
    # Assuming each row is a channel
    # Calculate the maximum absolute value in each channel
    max_abs_vals = weight_matrix.abs().max(dim=1).values

    # Calculate delta (scale) for each channel
    # For symmetric quantization, we use half the quantization levels for positive and half for negative
    half_range = (n_levels // 2) - 1
    deltas = max_abs_vals / half_range

    return deltas

def quantize(x, n_bits):
    xmax = torch.max(x)
    xmin = torch.min(x)
    delta = (xmax - xmin) / (2 ** n_bits - 1)
    zero_point = (-128 - xmin / delta).round()

    return delta, zero_point

def calculate_channelwise_symmetric_scale_4D(weights):
    """
    Calculate per-channel symmetric quantization scales for a convolution weight tensor.
    Weights tensor should have shape [Co, Cin, K, K].
    """
    Co, Cin, K, K = weights.shape
    half_range = 127  # for int8 quantization
    scale_factors = torch.zeros(Co).cuda()

    for i in range(Co):
        # Find the maximum absolute value in this output channel
        max_abs_val = torch.max(torch.abs(weights[i]))
        
        # Calculate the scale factor for this channel
        scale_factors[i] = max_abs_val / half_range if max_abs_val != 0 else 0

    return scale_factors

def benchmark_linearInheritance(bs=512, cin=960, cout=960):
    assert cin % 32 == 0 and cout % 32 == 0, 'cin and cout should be divisible by 32'

    x = 5 * torch.randn(bs,cin).cuda().to(torch.float16)

    my_quant_linear = qlinear_8bit_Linherit(cout, cin).cuda()
    linearfp = nn.Linear(cout,cin).cuda()

    bias = linearfp.bias.data.to(torch.float32)
    weight = linearfp.weight.data.to(torch.float32)

    my_bias = linearfp.bias.data.to(torch.float32)
    my_weight = linearfp.weight.data.to(torch.float32)

    ## start benchmark
    import time

    linearfp.float()
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(1000):
        out_fp = linearfp(x.to(torch.float32))
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for FP32: ', (end_time-start_time) / 100)

    linearfp.half()
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(1000):
        out_fp = linearfp(x)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for FP16: ', (end_time-start_time) / 100)


    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(1000):
        out_fp = my_quant_linear(x)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for INT8 (Quant+Dequant): ', (end_time-start_time) / 100)

def benchmark_linear(bs=512, cin=960, cout=960):
    assert cin % 32 == 0 and cout % 32 == 0, 'cin and cout should be divisible by 32'

    x = 5 * torch.randn(bs,cin).cuda().to(torch.float16)

    linearfp = nn.Linear(cout,cin).cuda().half()

    bias = linearfp.bias.data.to(torch.float32)
    weight = linearfp.weight.data.to(torch.float32)

    weight_delta = calculate_channelwise_symmetric_scale(weight).unsqueeze(-1)
    # weight_delta = weight_delta.transpose(1,0)
    int_weight = (linearfp.weight.data / weight_delta).round().to(torch.int8)

    act_delta, act_zp = quantize(x, n_bits=8)


    dequantized_weight = int_weight * weight_delta
    zp_times_weight_channel_sum = act_zp.to(torch.float32) * int_weight.sum(dim=1)
    act_times_weight_delta = act_delta.to(torch.float32) * weight_delta.to(torch.float32).squeeze(0)

    int_act = ((x / act_delta).round() + act_zp).to(torch.int8)
    dequantized_act = (int_act - act_zp) * act_delta

    out_int_fake = dequantized_act.to(torch.float32) @ dequantized_weight.T + bias

    ## warm up
    out_fp = linearfp(x)
    int_x = torch_quantizer.asymmetric.myQuantize(x, act_delta, act_zp)
    out_int8 = torch_quantizer.matmul.myInt8Matmul(int_x, int_weight, zp_times_weight_channel_sum, act_times_weight_delta, bias)

    ## start benchmark
    import time

    linearfp.float()
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(100):
        out_fp = linearfp(x.to(torch.float32))
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for FP32: ', (end_time-start_time) / 100)

    linearfp.half()
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(100):
        out_fp = linearfp(x)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for FP16: ', (end_time-start_time) / 100)

    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(100):
        # int_x = ((x / act_delta).round() + act_zp).to(torch.int8)
        int_x = torch_quantizer.asymmetric.myQuantize(x, act_delta, act_zp)
        out_int8 = torch_quantizer.matmul.myInt8Matmul(int_x, int_weight, zp_times_weight_channel_sum, act_times_weight_delta, bias)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for INT8 (Quant+Dequant): ', (end_time-start_time) / 100)

def benchmark_conv2d(bs=8, cin=384, h=32, w=32, cout=384, k=3, padding=0):
    x = (torch.randn((bs,cin,h,w)) + 3).half().cuda()
    conv1 = nn.Conv2d(cout,cin,k, padding=padding).half().cuda()

    conv1.weight.requires_grad = False
    conv1.bias.requires_grad = False

    weight = conv1.weight.data
    weight_delta = calculate_channelwise_symmetric_scale_4D(weight).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    int_weight = (weight / weight_delta).round().to(torch.int8)
    weight_nhwc = int_weight.permute(0,2,3,1).contiguous()
    dequantized_weight = int_weight * weight_delta

    act_delta, act_zp = quantize(x, n_bits=8)
    int_act = ((x / act_delta).round() + act_zp).to(torch.int8)
    dequantized_act = (int_act - act_zp) * act_delta

    bias = conv1.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    ## warm up
    out_fp = conv1(x)

    import time

    conv1.float()
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(100):
        out_fp = conv1(x.to(torch.float32))
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for FP32: ', (end_time-start_time) / 100)

    conv1.half()
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(100):
        out_fp = conv1(x)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for FP16: ', (end_time-start_time) / 100)

    ## int8Conv + fused dequantization
    zp_times_weight_channel_sum = act_zp.to(torch.float32) * int_weight.sum(dim=(1,2,3))
    act_times_weight_delta = act_delta.to(torch.float32) * weight_delta.reshape(-1,)
    bias = conv1.bias.to(torch.float32)
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(100):
        x_nhwc = torch_quantizer.asymmetric.myQuantizeNCHW(x, act_delta, act_zp)
        if padding != 0:
            x_nhwc = F.pad(x_nhwc, pad=(0,0,padding,padding,padding,padding), value=act_zp)
        out_int8_fused = torch_quantizer.matmul.myInt8Conv(x_nhwc, weight_nhwc, 0, 0, 1, 1, 1, 1, zp_times_weight_channel_sum, act_times_weight_delta, bias)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for INT8 (Quant+Dequant): ', (end_time-start_time) / 100)

def benchmark_conv2dInheritance(bs=8, cin=384, h=32, w=32, cout=384, k=3, padding=0):
    x = (torch.randn((bs,cin,h,w)) + 3).half().cuda()
    conv1 = nn.Conv2d(cout,cin,k, padding=padding).half().cuda()
    my_quant_conv = qconv2d_8bit_Cinherit(cout,cin,k, padding=padding).cuda()
    my_quant_conv.bias.requires_grad = False

    conv1.weight.requires_grad = False
    conv1.bias.requires_grad = False

    weight = conv1.weight.data
    weight_delta = calculate_channelwise_symmetric_scale_4D(weight).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    int_weight = (weight / weight_delta).round().to(torch.int8)
    weight_nhwc = int_weight.permute(0,2,3,1).contiguous()
    dequantized_weight = int_weight * weight_delta

    act_delta, act_zp = quantize(x, n_bits=8)
    int_act = ((x / act_delta).round() + act_zp).to(torch.int8)
    dequantized_act = (int_act - act_zp) * act_delta

    bias = conv1.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    ## warm up
    out_fp = conv1(x)

    import time

    conv1.float()
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(100):
        out_fp = conv1(x.to(torch.float32))
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for FP32: ', (end_time-start_time) / 100)

    conv1.half()
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(100):
        out_fp = conv1(x)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for FP16: ', (end_time-start_time) / 100)

    start_time = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(100):
        out_fp = my_quant_conv(x)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print('average time for INT8 (Quant+Dequant): ', (end_time-start_time) / 100)
