import torch
from diffusers import StableDiffusionPipeline
import torch_quantizer as tq

model_id = "PATH TO stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)

wq_params = {'n_bits': 8, 'channel_wise': True, 'scale_method': 'max'}
aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'mse'}
ddim_steps = 50

## convert to fakeq model
pipe.unet = tq.fake_quant(pipe.unet, wq_params, aq_params, num_steps=ddim_steps)

## run fakeq model to do calibration
pipe = pipe.to(device)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=ddim_steps).images[0]  

## convert to realq model
pipe.unet = tq.fake2real(pipe.unet, save_dir='.')

## sampling with INT8 model
pipe = pipe.to(device)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=ddim_steps).images[0]  
image.save("astronaut_rides_horse_8bit.png")
