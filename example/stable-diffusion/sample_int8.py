import torch
torch.manual_seed(3407)
from diffusers import StableDiffusionPipeline
import torch_quantizer as tq

model_id = "PATH TO stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)

wq_params = {'n_bits': 8, 'channel_wise': True, 'scale_method': 'max'}
aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'mse'}
ddim_steps = 50

## convert to INT8 model
pipe.unet = tq.real_quant(pipe.unet, n_bits=8, num_steps=ddim_steps, ckpt_path="PATH TO UNet2DConditionModel_8bits_{}steps.pth".format(ddim_steps))

## sampling
pipe = pipe.to(device)
prompt = "A cozy cabin nestled in a snowy forest with smoke rising from the chimney"
image = pipe(prompt, num_inference_steps=ddim_steps).images[0]  
image.save("cabin_8bit.png")
