import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch_quantizer
from torch_quantizer.src.quant_layer import qlinear_8bit_Linherit, qconv2d_8bit_Cinherit, qconv2d_8bit

def main():



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

    # def benchmark_linearInherit(bs=512, cin=960, cout=960):
    #     assert cin % 32 == 0 and cout % 32 == 0, 'cin and cout should be divisible by 32'

    #     x = 5 * torch.randn(bs,cin).cuda().half()
    #     linearfp = qlinear_8bit_Linherit(cout, cin).cuda()

    #     bias = linearfp.bias.data.to(torch.float32)
    #     weight = linearfp.weight.data.to(torch.float32)

    #     output = linearfp(x)
    #     # print(linearfp.int_weight)
    #     print(linearfp.weight)
    
    def quantCoherence(bs=1, cin=32, h=8, w=8, cout=32, k=3, padding=0):
        # weight shape (Cout, Cin, k, k)
        x = (torch.randn((bs,cin,h,w)) + 2).half().cuda()
        conv1 = nn.Conv2d(cout,cin,k, padding=padding).half().cuda()
        conv1.weight.requires_grad = False
        conv1.bias.requires_grad = False
        relu_fushion = True
        weight = conv1.weight.data
        
        weight_delta = calculate_channelwise_symmetric_scale_4D(weight).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        int_weight = (weight / weight_delta).round().to(torch.int8)
        weight_nhwc = int_weight.permute(0,2,3,1).contiguous()
        dequantized_weight = int_weight * weight_delta

        act_delta, act_zp = quantize(x, n_bits=8)
        int_act = ((x / act_delta).round() + act_zp).to(torch.int8)
        dequantized_act = (int_act - act_zp) * act_delta
        
       
        bias = conv1.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
        out_fp = conv1(x)

        out_fp = F.relu(out_fp)
        zp_times_weight_channel_sum = act_zp.to(torch.float32) * int_weight.sum(dim=(1,2,3))
        act_times_weight_delta = act_delta.to(torch.float32) * weight_delta.to(torch.float32).squeeze(0)
        bias = conv1.bias.to(torch.float32)
        
        x_nhwc = torch_quantizer.asymmetric.myQuantizeNCHW(x, act_delta, act_zp)
        out_int8_fused = torch_quantizer.matmul.myInt8Conv(x_nhwc, weight_nhwc, 0, 0, 1, 1, 1, 1, zp_times_weight_channel_sum, act_times_weight_delta, bias, relu_fushion)
        
        print("Standard conv output: ")
        print(out_fp)
        print("Quant conv output")
        print(out_int8_fused)
        print("Diff")
        print(out_fp - out_int8_fused)
    # benchmark_linearInherit()
    quantCoherence()

if __name__ == '__main__':
    main()