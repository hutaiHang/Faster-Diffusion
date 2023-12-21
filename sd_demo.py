from diffusers import StableDiffusionPipeline
import torch
from utils_sd import register_normal_pipeline, register_faster_forward, register_parallel_pipeline, seed_everything  # 1.import package
import time

seed_everything(2023)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a cat wearing sunglasses"
generator = torch.Generator().manual_seed(2023)
#warmup gpu for testing time
print('Warm up of the gpu')
for i in range(2):
    image = pipe(prompt)
#-------------------

print('Start Generating')

#origin pipeline
start_time = time.time()
image = pipe(prompt, generator = generator).images[0] 
end_time = time.time()
print("Origin Pipeline: {:.3f} seconds".format(end_time-start_time))
image.save("images/orgin_cat.png")

#------------------------------
register_parallel_pipeline(pipe) # 2. enable parallel. If memory is limited, replace it with  `register_normal_pipeline(pipe)`
register_faster_forward(pipe.unet)  # 3. encoder propagation
#------------------------------


#FasterDiffusion pipeline
generator = torch.Generator().manual_seed(2023)
start_time = time.time()
image = pipe.call(prompt, generator = generator).images[0] 
end_time = time.time()
print("Faster Diffusion: {:.3f} seconds".format(end_time-start_time))

  
image.save("images/faster_cat.png")
