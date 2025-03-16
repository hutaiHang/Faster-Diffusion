from JDiffusion.pipelines import StableDiffusionPipeline
import jittor as jt
import jtorch
from utils_sd import register_normal_pipeline, register_faster_forward  # 1.import package
import time

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id,use_safetensors=True)
pipe = pipe.to("cuda:0")

prompt = "a cat wearing a sunglasses"
# warmup gpu for testing time
print('Warm up of the gpu')
for i in range(2):
    image = pipe(prompt)
#-------------------


print('Start Generating')

#origin pipeline
start_time = time.time()
image = pipe(prompt).images[0] 
end_time = time.time()
print("Origin Pipeline: {:.3f} seconds".format(end_time-start_time))
image.save("images/orgin_cat.png")

#------------------------------
# register_parallel_pipeline(pipe) # 2. enable parallel. If memory is limited, replace it with  `register_normal_pipeline(pipe)`
register_normal_pipeline(pipe)
register_faster_forward(pipe.unet)  # 3. encoder propagation
#------------------------------

#FasterDiffusion pipeline
start_time = time.time()
image = pipe.call(prompt).images[0] 
end_time = time.time()
print("Faster Diffusion: {:.3f} seconds".format(end_time-start_time))

image.save("images/faster_cat.png")
