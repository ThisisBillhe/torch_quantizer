import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch_quantizer
from torch_quantizer.src.quant_layer import qlinear_8bit_Linherit, qconv2d_8bit_Cinherit

def main():

    def benchmark_convInherit(bs=1, cin=512, h=32, w=32, cout=512, k=3, padding=0):
        x = (torch.randn((bs,cin,h,w)) + 3).half().cuda()
        conv1 = qconv2d_8bit_Cinherit(cout, cin, 3, padding = padding).cuda()

        output = conv1(x)
        print(output)
        # print(conv1.int_weight)
        # print('--------------------------------------')
        # print(conv1.int_weight)




    def benchmark_linearInherit(bs=512, cin=960, cout=960):
        assert cin % 32 == 0 and cout % 32 == 0, 'cin and cout should be divisible by 32'

        x = 5 * torch.randn(bs,cin).cuda().half()
        linearfp = qlinear_8bit_Linherit(cout, cin).cuda()

        bias = linearfp.bias.data.to(torch.float32)
        weight = linearfp.weight.data.to(torch.float32)

        output = linearfp(x)
        # print(linearfp.int_weight)
        print(linearfp.weight)
        
    # benchmark_linearInherit()
    benchmark_convInherit()

if __name__ == '__main__':
    main()