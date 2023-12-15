from diffusers import StableDiffusionPipeline
import torch
from utils_sd import register_normal_pipeline, register_faster_forward, register_parallel_pipeline, seed_everything  # 1.import package

seed_everything(2023)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

#------------------------------
register_parallel_pipeline(pipe) # 2. enable parallel. If memory is limited, replace it with  `register_normal_pipeline(pipe)`
register_faster_forward(pipe.unet)  # 3. encoder propagation
#------------------------------
prompt = "a cat wearing sunglasses"
image = pipe.call(prompt).images[0]  
    
image.save("cat.png")
