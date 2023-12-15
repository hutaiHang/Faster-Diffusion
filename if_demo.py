from diffusers import DiffusionPipeline , IFPipeline, IFSuperResolutionPipeline, StableDiffusionUpscalePipeline
from diffusers.utils import pt_to_pil
from diffusers import DPMSolverMultistepScheduler

import torch
from utils_if import register_if1, register_if2,register_if3, register_faster_forward, seed_everything



seed_everything(2023)
prompt = "a lone sailboat drifting on calm waters"


stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    variant="fp16",
    torch_dtype=torch.float16,
).to('cuda')

stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0",
    text_encoder=None,
    variant="fp16",
    torch_dtype=torch.float16,
).to('cuda')

# stage 3
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": None,
    "watermarker": stage_1.watermarker,
}


stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    **safety_modules,
    torch_dtype=torch.float16
).to('cuda')



register_faster_forward(stage_1.unet, mod = '100ls')
register_if1(stage_1)

register_faster_forward(stage_2.unet, mod = 's2')
register_if2(stage_2)

register_if3(stage_3)
register_faster_forward(stage_3.unet, mod = 'pro')


prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
image1 = stage_1.call(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pt",
    num_inference_steps=25 #100
).images


image2 = stage_2.call(
    image=image1,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pt",
    num_inference_steps=25 #50
).images

image = stage_3.call(prompt=prompt, image=image2, noise_level=20, 
                num_inference_steps= 50 #75
            ).images[0]

image.save(f'if-demo/ours-ddpm/{prompt}.png')
