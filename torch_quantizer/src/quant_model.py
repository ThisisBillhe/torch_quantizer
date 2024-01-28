import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_layer import qconv2d_8bit, qlinear_8bit, FakeQuantModule

class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, n_bits=8, n_steps=1, skip_layer=None):
        super().__init__()
        self.model = model
        self.count = 0
        self.skip_layer = [] if skip_layer is None else skip_layer
        self.quant_module_refactor(self.model, n_bits=n_bits, n_steps=n_steps)
        
    def quant_module_refactor(self, module: nn.Module, n_bits=8, n_steps=1):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.Conv2d):
                self.count += 1
                if self.count not in self.skip_layer:
                    Cout, Cin = child_module.weight.shape[0], child_module.weight.shape[1]
                    if Cout % 32 == 0 and Cin % 32 ==0:
                        setattr(module, name, qconv2d_8bit(child_module, n_bits=n_bits, num_steps=n_steps))
                    else:
                        pass
                else:
                    pass
            elif isinstance(child_module, nn.Linear):
                self.count += 1
                if self.count not in self.skip_layer:
                    Cout, Cin = child_module.weight.shape[0], child_module.weight.shape[1]
                    if Cout % 32 == 0 and Cin % 32 ==0:                    
                        setattr(module, name, qlinear_8bit(child_module, n_bits=n_bits, num_steps=n_steps))
                    else:
                        pass
                else:
                    pass
            elif isinstance(child_module, FakeQuantModule):
                if child_module.fwd_func is F.linear:
                    setattr(module, name, qlinear_8bit(child_module, n_bits=n_bits, num_steps=n_steps))
                elif child_module.fwd_func is F.conv2d:
                    setattr(module, name, qconv2d_8bit(child_module, n_bits=n_bits, num_steps=n_steps))
            else:
                self.quant_module_refactor(child_module, n_bits=n_bits, n_steps=n_steps)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def half(self):
        for name, param in self.named_parameters():
            param.data = param.data.half()
        for name, buf in self.named_buffers():
            if 'zp_times_weight_channel_sum' in name or 'act_times_weight_delta' in name or 'bias' in name:
                buf.data = buf.data.float()
            elif 'int_weight' not in name:
                buf.data = buf.data.half() ## these data is required to be float32 for cuda kernel
                
class FakeQuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, num_steps=1):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.n_bits = weight_quant_params['n_bits']
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, num_steps=num_steps)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, num_steps=1):
        """
        Recursively replace the normal conv2d and Linear layer to FakeQuantModule
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                Cout, Cin = child_module.weight.shape[0], child_module.weight.shape[1]
                if Cout % 32 == 0 and Cin % 32 ==0:
                    setattr(module, name, FakeQuantModule(child_module, weight_quant_params, act_quant_params, num_steps=num_steps))
                else:
                    pass
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, num_steps=num_steps)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)