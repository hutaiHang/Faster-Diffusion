from diffusers import StableDiffusionPipeline
import transformers
import torch
from utils_sd import register_normal_pipeline, register_faster_forward, register_parallel_pipeline, seed_everything  # 1.import package
import time

text_encoder = transformers.CLIPTextModel.from_pretrained(
    "/runwayml/stable-diffusion-v1-5",
    subfolder = "text_encoder",
    torch_dtype = torch.float16
)

tokenizer = transformers.CLIPTokenizer.from_pretrained(
    "/runwayml/stable-diffusion-v1-5",
    subfolder = "tokenizer",
    torch_dtype = torch.float16
)

#Load customed model
# model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
model_id = "https://huggingface.co/s6yx/ReV_Animated/blob/main/rev_1.2.2/rev_1.2.2-fp16.safetensors"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", safety_checker = None)
pipe = StableDiffusionPipeline.from_single_file(model_id, text_encoder = text_encoder, tokenizer = tokenizer, safety_checker = None,  use_safetensors=True, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = 'a cat wearing sunglasses'

#warmup gpu for testing time
print('Warm up of the gpu')
for i in range(2):
    image = pipe(prompt)
#-------------------
    
#origin pipeline
generator = torch.Generator().manual_seed(7748)
start_time = time.time()
image1 = pipe(prompt, generator = generator, num_inference_steps=50).images[0]
end_time = time.time()
print("Origin Pipeline: {:.3f} seconds".format(end_time-start_time))


#------------------------------
register_faster_forward(pipe.unet, mod = 4) # 2. enable parallel. If memory is limited, replace it with  `register_normal_pipeline(pipe)`
register_parallel_pipeline(pipe, mod = 4)  # 3. encoder propagation
#------------------------------

#FasterDiffusion pipeline
generator = torch.Generator().manual_seed(7748)
start_time = time.time()
image2 = pipe.call(prompt, generator = generator, num_inference_steps=50).images[0] 
end_time = time.time()
print("Faster Diffusion: {:.3f} seconds".format(end_time-start_time))
  
image1.save("images/origin.png")
image2.save("images/faster.png")
