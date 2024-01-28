import torch
torch.manual_seed(3407)
from diffusers import StableDiffusionPipeline

model_id = "PATH TO stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)
ddim_steps = 50

## sampling
pipe = pipe.to(device)
prompt = "A cozy cabin nestled in a snowy forest with smoke rising from the chimney"
image = pipe(prompt, num_inference_steps=ddim_steps).images[0]  
image.save("cabin_8bit.png")
