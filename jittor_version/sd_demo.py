# from diffusers import StableDiffusionPipeline
from JDiffusion.pipelines import StableDiffusionPipeline
import jittor as jt
import jtorch
from utils_sd import register_normal_pipeline, register_faster_forward, register_parallel_pipeline, seed_everything  # 1.import package
import time

seed_everything(2025)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id,use_safetensors=True,)
pipe = pipe.to("cuda:0")
print(pipe.unet.dtype)
prompt = "a cat wearing sunglasses"
generator = jtorch.Generator().manual_seed(2025)
# warmup gpu for testing time
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
# register_parallel_pipeline(pipe, mod = 2) # 2. enable parallel. If memory is limited, replace it with  `register_normal_pipeline(pipe)`
register_normal_pipeline(pipe)
register_faster_forward(pipe.unet)  # 3. encoder propagation
#------------------------------

seed_everything(2025)
#FasterDiffusion pipeline
generator = jtorch.Generator().manual_seed(2025)
start_time = time.time()
image = pipe.call(prompt, generator = generator).images[0] 
end_time = time.time()
print("Faster Diffusion: {:.3f} seconds".format(end_time-start_time))

image.save("images/faster_cat.png")
