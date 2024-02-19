import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

class SymmetricQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, scale_method: str = 'max'):
        super(SymmetricQuantizer, self).__init__()
        assert n_bits == 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.inited = True ## modified here
        self.channel_wise = channel_wise
        self.scale_method = scale_method

        self.inited = False # use this when quantizing models
        self.register_buffer('delta', torch.tensor(0.005))
        
    def clipping(self, x, lower, upper):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x
    
    def forward(self, x: torch.Tensor):

        if self.inited is False:
            delta= self.init_quantization_scale(x, self.channel_wise)
            self.delta = torch.nn.Parameter(delta)

            self.inited = True

        # start quantization
        x_int = round_ste(x / self.delta)
        x_quant = self.clipping(x_int, -self.n_levels//2, self.n_levels//2 - 1)
        # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = x_quant * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta = None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()

            ## comment below for faster initialization in inference
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c]= self.init_quantization_scale(x_clone[c], channel_wise=False)

            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                ## symmetric
                x_absmax = max(abs(x_min), x_max)
                x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                # Initial factors
                start_factor = 0.95
                end_factor = 1.05

                # Calculate the step increase per iteration
                step = (end_factor - start_factor) / 80

                for i in range(80):
                    factor = start_factor + i * step
                    new_max = x_max * factor
                    new_min = x_min * factor
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
            else:
                raise NotImplementedError

        return delta

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int, -self.n_levels//2, self.n_levels//2 - 1)
        x_float_q = x_quant * delta
        return x_float_q

class naiveTemporalQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 num_steps = 1):
        super(naiveTemporalQuantizer, self).__init__()
        self.sym = symmetric
        assert n_bits == 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits

        self.total_steps = num_steps
        self.current_step = self.total_steps - 1

        self.register_buffer('delta_list', torch.tensor([torch.tensor(0.005) for _ in range(self.total_steps)]))
        self.register_buffer('zp_list', torch.tensor([torch.tensor(0.005) for _ in range(self.total_steps)]))

        self.inited = False 
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        
    def clipping(self, x, lower, upper):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x
    
    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.current_step == 0:
                self.inited = True
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.delta_list[self.current_step] = delta
            self.zp_list[self.current_step] = zero_point

            x_int = round_ste(x / self.delta_list[self.current_step]) + round_ste(self.zp_list[self.current_step])
            x_quant = self.clipping(x_int, -self.n_levels//2, self.n_levels//2 - 1) ## modified here to replace torch.clamp for gradient prop
            x_dequant = (x_quant - round_ste(self.zp_list[self.current_step])) * self.delta_list[self.current_step]
            self.current_step = self.total_steps - 1 if  self.current_step - 1 < 0 else self.current_step - 1
            return x_dequant
        else:
            x_int = round_ste(x / self.delta_list[self.current_step]) + round_ste(self.zp_list[self.current_step])
            x_quant = self.clipping(x_int, -self.n_levels//2, self.n_levels//2 - 1) ## modified here to replace torch.clamp for gradient prop
            x_dequant = (x_quant - round_ste(self.zp_list[self.current_step])) * self.delta_list[self.current_step]
            self.current_step = self.total_steps - 1 if  self.current_step - 1 < 0 else self.current_step - 1
            return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()

            ## comment below for faster initialization in inference
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)

            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) - self.n_levels // 2
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()

                best_score = 1e+10
                # Initial factors
                start_factor = 0.95
                end_factor = 1.05

                # Calculate the step increase per iteration
                step = (end_factor - start_factor) / 80

                for i in range(80):
                    factor = start_factor + i * step
                    new_max = x_max * factor
                    new_min = x_min * factor
                    
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    # score = lp_loss(x, x_q, p=torch.sqrt(x_max - x_min), reduction='all') ## adaptive p-norm
                    # score = lp_loss(x, x_q, p=torch.clamp(-1.22*torch.pow(torch.var(x),-1)+9.42,0.1,10.0), reduction='all')

                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round() - self.n_levels // 2

            else:
                raise NotImplementedError
        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},'
        return s.format(**self.__dict__)
