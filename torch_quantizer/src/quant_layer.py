import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import math
import torch_quantizer
from .quant_utils import SymmetricQuantizer, naiveTemporalQuantizer

class FakeQuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, num_steps=1):
        super(FakeQuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.dilation = org_module.dilation
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        # initialize quantizer
        self.weight_quantizer = SymmetricQuantizer(**weight_quant_params)

        self.act_quantizer = naiveTemporalQuantizer(**act_quant_params, num_steps=num_steps)

    def forward(self, input: torch.Tensor):
        weight = self.weight_quantizer(self.weight)
        bias = self.bias
        input = self.act_quantizer(input)
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        return out

class qlinear_8bit_Linherit(nn.Linear):
    """
    8-bit Linear Module. 
    """
    def __init__(self, in_features: int = 0, out_features: int = 0, org_module: nn.Linear = None, n_bits=8, num_steps=1):

        if org_module:
            ## Copy attributes from org_module
            self.in_features = org_module.in_features
            self.out_features = org_module.out_features
            self.ori_shape = org_module.weight.shape

        else:
            self.in_features = in_features
            self.out_features = out_features
            self.ori_shape = [out_features, in_features]
        

        super(qlinear_8bit_Linherit, self).__init__(self.in_features, self.out_features)

        self.fwd_kwargs = dict()
        # self.org_weight = org_module.weight.data.clone()

        
        
        # self.register_buffer('int_weight', torch.randint(-128, 127, (self.ori_shape[0], self.ori_shape[1]),
        #                                                      dtype=torch.int8, requires_grad=False))
        # self.register_buffer('int_weight', torch.zeros((self.ori_shape[0], self.ori_shape[1]), dtype=torch.int8))

         # Check if bias already exists in the parent class
        if not hasattr(self, 'bias'):
            if org_module and org_module.bias is not None:
                self.register_buffer('bias', org_module.bias.data)
            else:
                self.register_buffer('bias', torch.zeros(size=(self.ori_shape[0],)))

        self.n_bits = n_bits

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        self.ignore_reconstruction = False

        self.register_buffer('act_delta',torch.randn(size=(num_steps,), dtype=torch.float16)) ## should be float16
        self.register_buffer('act_zp',torch.randn(size=(num_steps,), dtype=torch.float16)) ## should be float16
        self.register_buffer('zp_times_weight_channel_sum',torch.randn(size=(num_steps, self.ori_shape[0]), dtype=torch.float32)) ## should be float32
        self.register_buffer('act_times_weight_delta',torch.randn(size=(num_steps, self.ori_shape[0]), dtype=torch.float32)) ## should be float32

        self.total_steps = num_steps
        self.current_step = self.total_steps - 1

    def forward(self, input: torch.Tensor):
        ## fetch quantization parameters
        act_delta = self.act_delta[self.current_step]
        act_zp = self.act_zp[self.current_step]
        zp_times_weight_channel_sum = self.zp_times_weight_channel_sum[self.current_step]
        act_times_weight_delta = self.act_times_weight_delta[self.current_step]

        self.current_step = self.total_steps - 1 if  self.current_step - 1 < 0 else self.current_step - 1

        ## perform linear operation
        if len(input.shape) != 2:
            original_shape = input.shape[:-1]
            input = input.view(-1, input.shape[-1])
        else:
            original_shape = None

        int_x = torch_quantizer.asymmetric.myQuantize(input, act_delta, act_zp)
        output = torch_quantizer.matmul.myInt8Matmul(int_x, self.weight.to(torch.int8), zp_times_weight_channel_sum, act_times_weight_delta, self.bias)
        
        if original_shape is not None:
            output = output.view(original_shape + (-1,)).to(torch.float16)
        else:
            output = output.to(torch.float16)
        
        return output

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

class qlinear_8bit(nn.Module):
    """
    8-bit Linear Module. 
    """
    def __init__(self, org_module: nn.Linear, n_bits=8, num_steps=1):
        super(qlinear_8bit, self).__init__()
        
        ## Copy attributes from org_module
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        
        self.fwd_kwargs = dict()
        # self.org_weight = org_module.weight.data.clone()

        self.ori_shape = org_module.weight.shape
        self.n_bits = n_bits
        # self.register_buffer('int_weight', torch.randint(-128, 127, (self.ori_shape[0], self.ori_shape[1]),
        #                                                      dtype=torch.int8, requires_grad=False))
        self.register_buffer('int_weight', torch.zeros((self.ori_shape[0], self.ori_shape[1]), dtype=torch.int8))

        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias.data)
        else:
            self.register_buffer('bias', torch.zeros(size=(self.ori_shape[0],)))

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        self.ignore_reconstruction = False

        self.register_buffer('act_delta',torch.randn(size=(num_steps,), dtype=torch.float16)) ## should be float16
        self.register_buffer('act_zp',torch.randn(size=(num_steps,), dtype=torch.float16)) ## should be float16
        self.register_buffer('zp_times_weight_channel_sum',torch.randn(size=(num_steps, self.ori_shape[0]), dtype=torch.float32)) ## should be float32
        self.register_buffer('act_times_weight_delta',torch.randn(size=(num_steps, self.ori_shape[0]), dtype=torch.float32)) ## should be float32

        self.total_steps = num_steps
        self.current_step = self.total_steps - 1

    def forward(self, input: torch.Tensor, scale = None):
        ## fetch quantization parameters
        act_delta = self.act_delta[self.current_step]
        act_zp = self.act_zp[self.current_step]
        zp_times_weight_channel_sum = self.zp_times_weight_channel_sum[self.current_step]
        act_times_weight_delta = self.act_times_weight_delta[self.current_step]

        self.current_step = self.total_steps - 1 if  self.current_step - 1 < 0 else self.current_step - 1

        ## perform linear operation
        if len(input.shape) != 2:
            original_shape = input.shape[:-1]
            input = input.view(-1, input.shape[-1])
        else:
            original_shape = None

        int_x = torch_quantizer.asymmetric.myQuantize(input, act_delta, act_zp)
        output = torch_quantizer.matmul.myInt8Matmul(int_x, self.int_weight, zp_times_weight_channel_sum, act_times_weight_delta, self.bias)
        
        if original_shape is not None:
            output = output.view(original_shape + (-1,)).to(torch.float16)
        else:
            output = output.to(torch.float16)
        
        return output

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

class qconv2d_8bit(nn.Module):
    """
    8-bit Conv2d Module. 
    """
    def __init__(self, org_module: nn.Conv2d, n_bits=8, num_steps=1):
        super(qconv2d_8bit, self).__init__()
        self.fwd_kwargs = dict(strideH=org_module.stride[0], strideW=org_module.stride[1], padH=org_module.padding[0], padW=org_module.padding[1],
                                dilationH=org_module.dilation[0], dilationW=org_module.dilation[1])
        self.ori_shape = org_module.weight.shape
        self.weight_nhwc_shape = [self.ori_shape[0], self.ori_shape[2], self.ori_shape[3], self.ori_shape[1]]

        self.n_bits = n_bits
        self.register_buffer('int_weight', torch.randint(-128, 127, self.weight_nhwc_shape,
                                                             dtype=torch.int8, requires_grad=False))

        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias.data)

        else:
            self.register_buffer('bias', torch.zeros(size=(self.ori_shape[0],)))

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        self.ignore_reconstruction = False

        self.register_buffer('act_delta',torch.randn(size=(num_steps,), dtype=torch.float16)) ## should be float16
        self.register_buffer('act_zp',torch.randn(size=(num_steps,), dtype=torch.float16)) ## should be float16
        self.register_buffer('zp_times_weight_channel_sum',torch.randn(size=(num_steps, self.ori_shape[0]), dtype=torch.float32)) ## should be float32
        self.register_buffer('act_times_weight_delta',torch.randn(size=(num_steps, self.ori_shape[0]), dtype=torch.float32)) ## should be float32

        self.total_steps = num_steps
        self.current_step = self.total_steps - 1

    def forward(self, input: torch.Tensor, scale = None):
        ## fetch quantization parameters
        act_delta = self.act_delta[self.current_step]
        act_zp = self.act_zp[self.current_step]
        zp_times_weight_channel_sum = self.zp_times_weight_channel_sum[self.current_step]
        act_times_weight_delta = self.act_times_weight_delta[self.current_step]

        self.current_step = self.total_steps - 1 if  self.current_step - 1 < 0 else self.current_step - 1

        ## perform conv operation
        if not input.is_contiguous():
            input = input.contiguous()
        x_nhwc = torch_quantizer.asymmetric.myQuantizeNCHW(input, act_delta, act_zp) ## why some input can be non-contiguous?
        if self.fwd_kwargs['padH'] > 0:
            x_nhwc = F.pad(x_nhwc, pad=(0,0,self.fwd_kwargs['padH'],self.fwd_kwargs['padH'],self.fwd_kwargs['padW'],self.fwd_kwargs['padW']), value=act_zp)
        output_nchw = torch_quantizer.matmul.myInt8Conv(x_nhwc, self.int_weight, 0,0, self.fwd_kwargs['strideH'],\
                                             self.fwd_kwargs['strideW'],self.fwd_kwargs['dilationH'],self.fwd_kwargs['dilationW'],\
                                             zp_times_weight_channel_sum, act_times_weight_delta, self.bias, False).to(torch.float16)

        return output_nchw

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

class qconv2d_8bit_Cinherit(nn.Conv2d):
    """
    8-bit Conv2d Module. 
    """
    def __init__(self, in_channels:int = 0, out_channels:int = 0, kernel_size: tuple[int, ...] = 1, stride:tuple[int, ...] = 1, 
                padding:tuple[int, ...] = 0, padding_mode:str = "zeros", dilation:tuple[int, ...] = 1, groups:int = 1, bias:bool = True,
                org_module: nn.Conv2d = None, n_bits=8, num_steps=1, relu_fushion:bool=False):
        if org_module:
            in_channels = org_module.in_channels
            out_channels = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = getattr(org_module, 'stride', 1)
            padding = getattr(org_module, 'padding', 0)
            padding_mode = getattr(org_module, 'padding_mode', 'zeros')
            dilation = getattr(org_module, 'dilation', 1)
            groups = getattr(org_module, 'groups', 1)
            bias = getattr(org_module, 'bias', True)
            self.ori_shape = org_module.weight.shape

        super(qconv2d_8bit_Cinherit, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            padding_mode = padding_mode,
            dilation = dilation,
            bias = bias
        )

        if isinstance(kernel_size, tuple) and len(kernel_size) > 1:
            self.ori_shape = [out_channels, in_channels, kernel_size[0], kernel_size[1]]
        else:
            self.ori_shape = [out_channels, in_channels, kernel_size, kernel_size]

        if isinstance(stride, tuple) and len(stride) > 1:
            self.strideH, self.strideW = stride[0], stride[1]
        else:
            self.strideH = self.strideW = stride
        
        if isinstance(padding, tuple) and len(padding) > 1:
            self.padH, self.padW = padding[0], padding[1]
        else:
            self.padH = self.padW = padding
        
        if isinstance(dilation, tuple) and len(dilation) > 1:
            self.dilationH, self.dilationW = dilation[0], dilation[1]
        else:
            self.dilationH = self.dilationW = dilation

        # self.fwd_kwargs = dict(strideH=org_module.stride[0], strideW=org_module.stride[1], padH=org_module.padding[0], padW=org_module.padding[1],
        #                         dilationH=org_module.dilation[0], dilationW=org_module.dilation[1])
        
        self.weight_nhwc_shape = [self.ori_shape[0], self.ori_shape[2], self.ori_shape[3], self.ori_shape[1]]
        self.relu_fushion = relu_fushion
        self.n_bits = n_bits
        # self.weight = torch.tensor(torch.randint(-128, 127, self.weight_nhwc_shape, dtype=torch.int8))
        self.register_buffer('int_weight', torch.randint(-128, 127, self.weight_nhwc_shape,
                                                             dtype=torch.int8, requires_grad=False))
        # if org_module and org_module.bias is not None:
        #     self.register_buffer('bias', org_module.bias.data)

        # else:
        #     self.register_buffer('bias', torch.zeros(size=(self.ori_shape[0],)))

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        self.ignore_reconstruction = False

        self.register_buffer('act_delta',torch.randn(size=(num_steps,), dtype=torch.float16)) ## should be float16
        self.register_buffer('act_zp',torch.randn(size=(num_steps,), dtype=torch.float16)) ## should be float16
        self.register_buffer('zp_times_weight_channel_sum',torch.randn(size=(num_steps, self.ori_shape[0]), dtype=torch.float32)) ## should be float32
        self.register_buffer('act_times_weight_delta',torch.randn(size=(num_steps, self.ori_shape[0]), dtype=torch.float32)) ## should be float32

        self.total_steps = num_steps
        self.current_step = self.total_steps - 1

    def forward(self, input: torch.Tensor):
        ## fetch quantization parameters
        act_delta = self.act_delta[self.current_step]
        act_zp = self.act_zp[self.current_step]
        zp_times_weight_channel_sum = self.zp_times_weight_channel_sum[self.current_step]
        act_times_weight_delta = self.act_times_weight_delta[self.current_step]

        self.current_step = self.total_steps - 1 if  self.current_step - 1 < 0 else self.current_step - 1

        ## perform conv operation
        if not input.is_contiguous():
            input = input.contiguous()
        x_nhwc = torch_quantizer.asymmetric.myQuantizeNCHW(input, act_delta, act_zp) ## why some input can be non-contiguous?
        if self.padH > 0:
            x_nhwc = F.pad(x_nhwc, pad=(0,0,self.padH,self.padH,self.padW,self.padW), value=act_zp)
        output_nchw = torch_quantizer.matmul.myInt8Conv(x_nhwc, self.int_weight, 0,0, self.strideH,
                                             self.strideW, self.dilationH ,self.dilationW,
                                             zp_times_weight_channel_sum, act_times_weight_delta, self.bias, self.relu_fushion).to(torch.float16)

        return output_nchw

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
