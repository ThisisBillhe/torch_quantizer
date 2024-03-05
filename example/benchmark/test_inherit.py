import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch_quantizer
from torch_quantizer.src.quant_layer import qlinear_8bit_Linherit

def main():

    def benchmark_linearInherit(bs=512, cin=960, cout=960):
        assert cin % 32 == 0 and cout % 32 == 0, 'cin and cout should be divisible by 32'

        x = 5 * torch.randn(bs,cin).cuda().half()
        linearfp = qlinear_8bit_Linherit(cout, cin).cuda()

        bias = linearfp.bias.data.to(torch.float32)
        weight = linearfp.weight.data.to(torch.float32)

        output = linearfp(x)
        print(linearfp.in_features)
        
    benchmark_linearInherit()

if __name__ == '__main__':
    main()