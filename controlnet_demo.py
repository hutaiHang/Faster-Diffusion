import numpy as np
import cv2
import time
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
import torch
from controlnet_aux import HEDdetector, OpenposeDetector
from diffusers.utils import load_image
from utils_sd import register_controlnet_pipeline, register_faster_forward, seed_everything



image = load_image("images/condition.jpeg")
image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image_condition = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
).to('cuda')

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to('cuda')

print('Warm up of the gpu')
for i in range(2):
    image = pipe("Mona Lisa", image_condition).images[0]
#-------------------
    
print("Start Generating")

seed_everything(8888)
start_time = time.time()
image = pipe("Mona Lisa", image_condition).images[0]
end_time = time.time()
image.save('images/canny_out_origin.png')
print("Origin Pipeline: {:.3f} seconds".format(end_time-start_time))

register_controlnet_pipeline(pipe)
register_faster_forward(pipe.unet)

seed_everything(8888)

start_time = time.time()
image = pipe.call("Mona Lisa", image_condition).images[0]
end_time = time.time()
print("Faster Pipeline: {:.3f} seconds".format(end_time-start_time))

image.save('images/canny_out_faster.png')
