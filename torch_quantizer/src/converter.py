import os
import torch

from torch_quantizer.src.quant_model import QuantModel, FakeQuantModel

def fake_quant(model, weight_quant_params, act_quant_params, num_steps=1):
    '''
    Quantize a floating-point model to fake quantized model. The fakeq model will be used for calibration.

    Args:
        model (YourFloatingPointModelClass): The original floating-point model to be quantized.
        weight_quant_params: a dict specifies n_bits, channel_wise and scale_method. For example: {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'max'}.
        act_quant_params: same as weight_quant_params.
        num_steps (int, optional): The number of quantization steps. Default is 1.
    Returns:
        FakeQModelClass: The fake quantized model.
    '''

    fakeq_model = FakeQuantModel(model, weight_quant_params, act_quant_params, num_steps).model
    setattr(fakeq_model, 'num_steps', num_steps)
    setattr(fakeq_model, 'n_bits', 8)

    return fakeq_model

def real_quant(model, n_bits=8, num_steps=1, ckpt_path=None):
    '''
    Quantize a floating-point model to INT8.

    Args:
        model (YourFloatingPointModelClass): The original floating-point model to be quantized.
        num_steps (int, optional): The number of quantization steps. Default is 1.
        ckpt_path (str, optional): The path to a pre-trained INT8 model checkpoint. If not provided, 
            an INT8 model will be randomly initialized (which can be used to benchmark).
    Returns:
        YourINT8ModelClass: The quantized INT8 model.
    '''
    if ckpt_path is not None:
        print('Restoring INT8 models from {}'.format(ckpt_path))
    else:
        print('Get INT8 models without checkpoint...')
    realq_model = QuantModel(model, n_bits=n_bits, n_steps=num_steps)
    realq_model.half()
    realq_model = realq_model.model ## get rid of the wrapper

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        realq_model.load_state_dict(ckpt, strict=False)

    setattr(realq_model, 'num_steps', num_steps)

    return realq_model



def fake2real(fakeq_model, save_dir='.'):
    n_levels = 2**8
    ckpt = fakeq_model.state_dict()
    for k in list(ckpt.keys()):
        if 'weight_quantizer' in k and len(ckpt[k[:-16]].shape)==2: ## Linear layer
            prefix = k[:-22] ## with '.' in the end
            weight_k = k[:-16]
            weight_delta = ckpt[k].reshape(-1,1)
            int_weight = torch.clamp((ckpt[weight_k] / weight_delta).round(), -n_levels//2, n_levels//2 - 1).to(torch.int8)
            ckpt[prefix + 'int_weight'] = int_weight
            # ckpt[prefix + 'weight_delta'] = weight_delta.transpose(1,0)
            del ckpt[weight_k]
            del ckpt[k]

            act_delta_list_k = k[:-22]+'act_quantizer.delta_list'
            act_zp_list_k = k[:-22]+'act_quantizer.zp_list'
            act_zp = ckpt[act_zp_list_k].clone().detach().reshape(-1,)
            act_delta = ckpt[act_delta_list_k].clone().detach().reshape(-1,)
            ckpt[prefix + 'act_delta'] = act_delta
            ckpt[prefix + 'act_zp'] = act_zp
            ckpt[prefix + 'zp_times_weight_channel_sum'] = act_zp.unsqueeze(-1) * int_weight.sum(dim=1).unsqueeze(0).to(torch.float32)
            ckpt[prefix + 'act_times_weight_delta'] = act_delta.unsqueeze(-1) * weight_delta.reshape(1,-1)
            if ckpt[prefix + 'zp_times_weight_channel_sum'].isnan().any() or ckpt[prefix + 'zp_times_weight_channel_sum'].isinf().any():
                print('nan or inf!')
            del ckpt[act_delta_list_k]
            del ckpt[act_zp_list_k]
            # del ckpt[act_delta_list_k[:-5]]
            # del ckpt[act_delta_list_k[:-10]+'zero_point']

        elif 'weight_quantizer' in k and len(ckpt[k[:-16]].shape)==4: ## Conv layer
            prefix = k[:-22] ## with '.' in the end
            weight_k = k[:-16]
            weight_delta = ckpt[k]  ## (Co, 1, 1, 1)
            int_weight = torch.clamp((ckpt[weight_k] / weight_delta).round(), -n_levels//2, n_levels//2 - 1).to(torch.int8)
            weight_nhwc = int_weight.permute(0,2,3,1).contiguous()
            ckpt[prefix + 'int_weight'] = weight_nhwc
            del ckpt[weight_k]
            del ckpt[k]

            act_delta_list_k = k[:-22]+'act_quantizer.delta_list'
            act_zp_list_k = k[:-22]+'act_quantizer.zp_list'
            act_zp = ckpt[act_zp_list_k].clone().detach().reshape(-1,)
            act_delta = ckpt[act_delta_list_k].clone().detach().reshape(-1,)
            ckpt[prefix + 'act_delta'] = act_delta
            ckpt[prefix + 'act_zp'] = act_zp
            ckpt[prefix + 'zp_times_weight_channel_sum'] = act_zp.unsqueeze(-1) * int_weight.sum(dim=(1,2,3)).unsqueeze(0).to(torch.float32)
            ckpt[prefix + 'act_times_weight_delta'] = act_delta.unsqueeze(-1) * weight_delta.reshape(1,-1)
            if ckpt[prefix + 'zp_times_weight_channel_sum'].isnan().any() or ckpt[prefix + 'zp_times_weight_channel_sum'].isinf().any():
                print('nan or inf!')
            del ckpt[act_delta_list_k]
            del ckpt[act_zp_list_k]
            # del ckpt[act_delta_list_k[:-5]]
            # del ckpt[act_delta_list_k[:-10]+'zero_point']

    model_name = fakeq_model.__class__.__name__
    save_path = os.path.join(save_dir, '{}_8bits_{}steps.pth'.format(model_name, fakeq_model.num_steps))
    print('Saving quantized checkpoint to {}'.format(save_path))
    torch.save(ckpt, save_path)

    realq_model = QuantModel(fakeq_model, n_bits=fakeq_model.n_bits, n_steps=fakeq_model.num_steps).to(next(fakeq_model.parameters()).device)
    realq_model.half()
    realq_model = realq_model.model ## to get rid of redundent fakeq_model
    is_compatible = realq_model.load_state_dict(ckpt, strict=False) ## we assume there is bias for every layer, which may cause missing keys.

    return realq_model