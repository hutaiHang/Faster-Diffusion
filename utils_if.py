import imp
import numpy as np
import cv2
import torch
import random
from PIL import Image, ImageDraw, ImageFont
import copy
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from tqdm.notebook import tqdm
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    get_down_block,
    get_up_block,
)
from diffusers.utils.torch_utils import randn_tensor

from diffusers.models.unet_2d_condition import UNet2DConditionOutput, logger
from copy import deepcopy
import json

import inspect
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import is_compiled_module

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from tqdm import tqdm
from controlnet_aux import HEDdetector, OpenposeDetector
import time

from diffusers.pipelines.deepfloyd_if import IFPipelineOutput

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_promptls(prompt_path):
    with open(prompt_path) as f:
        prompt_ls = json.load(f)
    prompt_ls = [prompt['caption'].replace('/','_') for prompt in prompt_ls]
    return prompt_ls

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    # print(image_path)
    if type(image_path) is str:
        image = np.array(Image.open(image_path))
        if image.ndim>3:
            image = image[:,:,:3]
        elif image.ndim == 2:
            image = image.reshape(image.shape[0], image.shape[1],1).astype('uint8')
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def get_canny(image_path):
    image = load_512(
        image_path
    )
    image = np.array(image)

    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def get_scribble(image_path, hed):
    image = load_512(
        image_path
    )
    image = hed(image, scribble=True)

    return image

def get_cocoimages(prompt_path):
    data_ls = []
    with open(prompt_path) as f:
        prompt_ls = json.load(f)
    img_path = 'COCO2017-val/val2017'
    for prompt in tqdm(prompt_ls):
        caption = prompt['caption'].replace('/','_')
        image_id = str(prompt['image_id'])
        image_id = (12-len(image_id))*'0' + image_id+'.jpg'
        image_path = os.path.join(img_path, image_id)
        try:
            image = get_canny(image_path)
        except:
            continue
        curr_data = {'image':image, 'prompt':caption}
        data_ls.append(curr_data)
    return data_ls

def get_cocoimages2(prompt_path):
    """scribble condition
    """
    data_ls = []
    with open(prompt_path) as f:
        prompt_ls = json.load(f)
    img_path = 'COCO2017-val/val2017'
    hed = HEDdetector.from_pretrained('ControlNet/detector_weights/annotator', filename='network-bsds500.pth')
    for prompt in tqdm(prompt_ls):
        caption = prompt['caption'].replace('/','_')
        image_id = str(prompt['image_id'])
        image_id = (12-len(image_id))*'0' + image_id+'.jpg'
        image_path = os.path.join(img_path, image_id)
        try:
            image = get_scribble(image_path,hed)
        except:
            continue
        curr_data = {'image':image, 'prompt':caption}
        data_ls.append(curr_data)
    return data_ls

def warpped_feature(sample, step):
    """
    sample: batch_size*dim*h*w, uncond: 0 - batch_size//2, cond: batch_size//2 - batch_size
    step: timestep span
    """
    bs, dim, h, w = sample.shape
    uncond_fea, cond_fea = sample.chunk(2)
    uncond_fea = uncond_fea.repeat(step,1,1,1) # (step * bs//2) * dim * h *w
    cond_fea = cond_fea.repeat(step,1,1,1) # (step * bs//2) * dim * h *w
    return torch.cat([uncond_fea, cond_fea])

def warpped_skip_feature(block_samples, step):
    down_block_res_samples = []
    for sample in block_samples:
        sample_expand = warpped_feature(sample, step)
        down_block_res_samples.append(sample_expand)
    return tuple(down_block_res_samples)

def warpped_text_emb(text_emb, step):
    """
    text_emb: batch_size*77*768, uncond: 0 - batch_size//2, cond: batch_size//2 - batch_size
    step: timestep span
    """
    bs, token_len, dim = text_emb.shape
    uncond_fea, cond_fea = text_emb.chunk(2)
    uncond_fea = uncond_fea.repeat(step,1,1) # (step * bs//2) * 77 *768
    cond_fea = cond_fea.repeat(step,1,1) # (step * bs//2) * 77 * 768
    return torch.cat([uncond_fea, cond_fea]) # (step*bs) * 77 *768

def warpped_timestep(timesteps, bs):
    """
    timestpes: list, such as [981, 961, 941]
    """
    semi_bs = bs//2
    ts = []
    for timestep in timesteps:
        timestep = timestep[None]
        texp = timestep.expand(semi_bs)
        ts.append(texp)
    timesteps = torch.cat(ts)
    return timesteps.repeat(2,1).reshape(-1)

def register_faster_forward(model, mod):
    def faster_forward(self):
        def forward(
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                return_dict: bool = True,
            ) -> Union[UNet2DConditionOutput, Tuple]:
                r"""
                Args:
                    sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
                    timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
                    encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
                    return_dict (`bool`, *optional*, defaults to `True`):
                        Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
                    cross_attention_kwargs (`dict`, *optional*):
                        A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                        `self.processor` in
                        [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

                Returns:
                    [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                    [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
                    returning a tuple, the first element is the sample tensor.
                """
                # By default samples have to be AT least a multiple of the overall upsampling factor.
                # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
                # However, the upsampling interpolation output size can be forced to fit any upsampling size
                # on the fly if necessary.
                default_overall_up_factor = 2**self.num_upsamplers

                # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
                forward_upsample_size = False
                upsample_size = None

                if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                    logger.info("Forward upsample size to force interpolation output size.")
                    forward_upsample_size = True

                # prepare attention_mask
                if attention_mask is not None:
                    attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                    attention_mask = attention_mask.unsqueeze(1)

                # 0. center input if necessary
                if self.config.center_input_sample:
                    sample = 2 * sample - 1.0

                # 1. time
                if isinstance(timestep, list):
                    timesteps = timestep[0]
                    step = len(timestep)
                else:
                    timesteps = timestep
                    step = 1
                if not torch.is_tensor(timesteps) and (not isinstance(timesteps,list)):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = sample.device.type == "mps"
                    if isinstance(timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
                elif (not isinstance(timesteps,list)) and len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(sample.device)
                
                if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timesteps = timesteps.expand(sample.shape[0])
                elif isinstance(timesteps, list):
                    #timesteps list, such as [981,961,941]
                    timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)
                t_emb = self.time_proj(timesteps)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # but time_embedding might actually be running in fp16. so we need to cast here.
                # there might be better ways to encapsulate this.
                t_emb = t_emb.to(dtype=self.dtype)

                emb = self.time_embedding(t_emb, timestep_cond)

                if self.class_embedding is not None:
                    if class_labels is None:
                        raise ValueError("class_labels should be provided when num_class_embeds > 0")

                    if self.config.class_embed_type == "timestep":
                        class_labels = self.time_proj(class_labels)

                        # `Timesteps` does not contain any weights and will always return f32 tensors
                        # there might be better ways to encapsulate this.
                        class_labels = class_labels.to(dtype=sample.dtype)

                    class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

                    if self.config.class_embeddings_concat:
                        emb = torch.cat([emb, class_emb], dim=-1)
                    else:
                        emb = emb + class_emb

                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)
                    emb = emb + aug_emb

                if self.time_embed_act is not None:
                    emb = self.time_embed_act(emb)

                if self.encoder_hid_proj is not None:
                    encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

                order = self.order
                cond = order in [0, 1, 2, 3, 5, 10, 15, 25, 35]
                ipow = int(np.sqrt(9 + 8*order))
                if isinstance(mod, int):
                    cond = order % mod == 0
                elif mod == "pro":
                    cond = ipow * ipow == (9 + 8 * order)
                elif mod == "50ls":
                    cond = order in [0, 1, 2, 3, 5, 10, 15, 25, 35,40] #40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
                elif mod == "50ls2":
                    cond = order in [0, 10, 11, 12, 15, 20, 25, 30,35,45] #40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
                elif mod == "50ls3":
                    cond = order in [0, 20, 25, 30,35,45,46,47,48,49] #40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
                elif mod == "50ls4":
                    cond = order in [0, 9, 13, 14, 15, 28, 29, 32, 36] #40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
                elif mod == "100ls":
                    cond = order > 85 or order < 10 or order % 5 == 0
                elif mod == "75ls":
                    cond = order > 65 or order < 10 or order % 5 == 0
                elif mod == "75ls2":
                    cond = order in [0, 1, 2, 3, 5, 10, 15, 25, 35,45,55,65]
                elif mod == "s2":
                    cond = True
                #===============
                order = self.order #timestep, start by 0
                #===============
                # if ipow*ipow == (9+8*order):  #progressive skip, i.e. [0,2,5,...]
                if cond:
                # if order%2 == 0: # merge 2 step
                    # print(order)
                    # 2. pre-process
                    sample = self.conv_in(sample)

                    # 3. down
                    down_block_res_samples = (sample,)
                    for downsample_block in self.down_blocks:
                        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                            sample, res_samples = downsample_block(
                                hidden_states=sample,
                                temb=emb,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                                cross_attention_kwargs=cross_attention_kwargs,
                            )
                        else:
                            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                        down_block_res_samples += res_samples

                    if down_block_additional_residuals is not None:
                        new_down_block_res_samples = ()

                        for down_block_res_sample, down_block_additional_residual in zip(
                            down_block_res_samples, down_block_additional_residuals
                        ):
                            down_block_res_sample = down_block_res_sample + down_block_additional_residual
                            new_down_block_res_samples += (down_block_res_sample,)

                        down_block_res_samples = new_down_block_res_samples

                    # 4. mid
                    if self.mid_block is not None:
                        sample = self.mid_block(
                            sample,
                            emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                        )

                    if mid_block_additional_residual is not None:
                        sample = sample + mid_block_additional_residual

                    #----------------------save feature-------------------------
                    setattr(self, 'skip_feature', deepcopy(down_block_res_samples))
                    setattr(self, 'toup_feature', sample.detach().clone())
                    #-----------------------save feature------------------------



                    #-------------------expand feature for parallel---------------
                    # print(step)

                    # print('pre emb shape', emb.shape)
                    if isinstance(timestep, list):
                        #timesteps list, such as [981,961,941]
                        timesteps = warpped_timestep(timestep, sample.shape[0]).to(sample.device)
                        t_emb = self.time_proj(timesteps)

                        # `Timesteps` does not contain any weights and will always return f32 tensors
                        # but time_embedding might actually be running in fp16. so we need to cast here.
                        # there might be better ways to encapsulate this.
                        t_emb = t_emb.to(dtype=self.dtype)

                        emb = self.time_embedding(t_emb, timestep_cond)
                        # print('post emb shape', emb.shape)

                    # print('pre sample shape', sample.shape)
                    # print(step, sample.shape)
                    down_block_res_samples = warpped_skip_feature(down_block_res_samples, step)
                    sample = warpped_feature(sample, step)
                    # print('post sample shape', sample.shape)

                    # print('pre text shape', encoder_hidden_states.shape)
                    encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)
                    # print('post text shape', encoder_hidden_states.shape)
                    # print('==========================')
                    #-------------------expand feature for parallel---------------
                    
                else:
                    down_block_res_samples = self.skip_feature
                    sample = self.toup_feature

                    #-------------------expand feature for parallel---------------
                    down_block_res_samples = warpped_skip_feature(down_block_res_samples, step)
                    sample = warpped_feature(sample, step)
                    encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)
                    #-------------------expand feature for parallel---------------

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                        )

                # 6. post-process
                if self.conv_norm_out:
                    sample = self.conv_norm_out(sample)
                    sample = self.conv_act(sample)
                sample = self.conv_out(sample)

                if not return_dict:
                    return (sample,)

                return UNet2DConditionOutput(sample=sample)
        return forward
    if model.__class__.__name__ == 'UNet2DConditionModel':
        model.forward = faster_forward(model)


def refister_time(unet, t):
    setattr(unet, "order", t)

@torch.no_grad()
def multistep_pre(self, noise_pred, t, x):
    step_span = len(t)
    bs = noise_pred.shape[0]
    bs_perstep = bs//step_span

    denoised_latent = x
    for i, timestep in enumerate(t):
        curr_noise = noise_pred[i*bs_perstep:(i+1)*bs_perstep]
        denoised_latent = self.scheduler.step(curr_noise, timestep, denoised_latent)['prev_sample']
    return denoised_latent
    
def register_if1(pipe):
    def new_call(self):
        @torch.no_grad()
        def call(
            prompt: Union[str, List[str]] = None,
            num_inference_steps: int = 100,
            timesteps: List[int] = None,
            guidance_scale: float = 7.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            height: Optional[int] = None,
            width: Optional[int] = None,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            clean_caption: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        ):
            # 1. Check inputs. Raise error if not correct
            self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

            # 2. Define call parameters
            height = height or self.unet.config.sample_size
            width = width or self.unet.config.sample_size

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                clean_caption=clean_caption,
            )

            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 4. Prepare timesteps
            if timesteps is not None:
                self.scheduler.set_timesteps(timesteps=timesteps, device=device)
                timesteps = self.scheduler.timesteps
                num_inference_steps = len(timesteps)
            else:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps

            # 5. Prepare intermediate images
            intermediate_images = self.prepare_intermediate_images(
                batch_size * num_images_per_prompt,
                self.unet.config.in_channels,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # HACK: see comment in `enable_model_cpu_offload`
            if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
                self.text_encoder_offload_hook.offload()

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

            all_timesteps = len(timesteps)
            curr_step = 0
            st = time.time()
            while curr_step<all_timesteps:
                refister_time(self.unet, curr_step)

                time_ls = []
                time_ls.append(timesteps[curr_step])
                curr_step += 1
                cond = curr_step > 85 or curr_step < 10 or (curr_step % 5 == 0)
                
                while (not cond) and (curr_step<all_timesteps):
                    time_ls.append(timesteps[curr_step])
                    curr_step += 1
                    cond = curr_step > 85 or curr_step < 10 or (curr_step % 5 == 0)

                # print('curr_step', curr_step, len(time_ls))
                model_input = (
                    torch.cat([intermediate_images] * 2) if do_classifier_free_guidance else intermediate_images
                )

                # predict the noise residual
                noise_pred = self.unet(
                    model_input,
                    time_ls,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)

                # compute the previous noisy sample x_t -> x_t-1
                # intermediate_images = self.scheduler.step(
                #     noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                # )[0]
                intermediate_images = multistep_pre(
                    self, noise_pred, time_ls, intermediate_images)
            et = time.time()
            print('unet time: ', et-st, 'seconds')
            image = intermediate_images

            if output_type == "pil":
                # 8. Post-processing
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                # 9. Run safety checker
                image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

                # 10. Convert to PIL
                image = self.numpy_to_pil(image)

                # 11. Apply watermark
                if self.watermarker is not None:
                    image = self.watermarker.apply_watermark(image, self.unet.config.sample_size)
            elif output_type == "pt":
                nsfw_detected = None
                watermark_detected = None

                if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                    self.unet_offload_hook.offload()
            else:
                # 8. Post-processing
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                # 9. Run safety checker
                image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, nsfw_detected, watermark_detected)

            return IFPipelineOutput(images=image, nsfw_detected=nsfw_detected, watermark_detected=watermark_detected)
        return call
    pipe.call = new_call(pipe)




def register_if2(pipe):
    def new_call(self):
        @torch.no_grad()
        def call(
            prompt: Union[str, List[str]] = None,
            height: int = None,
            width: int = None,
            image: Union[PIL.Image.Image, np.ndarray, torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            guidance_scale: float = 4.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            noise_level: int = 250,
            clean_caption: bool = True,
        ):
            # 1. Check inputs. Raise error if not correct

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            self.check_inputs(
                prompt,
                image,
                batch_size,
                noise_level,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
            )

            # 2. Define call parameters

            height = height or self.unet.config.sample_size
            width = width or self.unet.config.sample_size

            device = self._execution_device

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                clean_caption=clean_caption,
            )

            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 4. Prepare timesteps
            if timesteps is not None:
                self.scheduler.set_timesteps(timesteps=timesteps, device=device)
                timesteps = self.scheduler.timesteps
                num_inference_steps = len(timesteps)
            else:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps

            # 5. Prepare intermediate images
            num_channels = self.unet.config.in_channels // 2
            intermediate_images = self.prepare_intermediate_images(
                batch_size * num_images_per_prompt,
                num_channels,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Prepare upscaled image and noise level
            image = self.preprocess_image(image, num_images_per_prompt, device)
            upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

            noise_level = torch.tensor([noise_level] * upscaled.shape[0], device=upscaled.device)
            noise = randn_tensor(upscaled.shape, generator=generator, device=upscaled.device, dtype=upscaled.dtype)
            upscaled = self.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)

            if do_classifier_free_guidance:
                noise_level = torch.cat([noise_level] * 2)

            # HACK: see comment in `enable_model_cpu_offload`
            if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
                self.text_encoder_offload_hook.offload()

            # 8. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

            all_timesteps = len(timesteps)
            curr_step = 0
            st = time.time()
            while curr_step<all_timesteps:
                refister_time(self.unet, curr_step)

                time_ls = []
                time_ls.append(timesteps[curr_step])
                curr_step += 1
                cond = curr_step < 20 or curr_step > 40 or (curr_step % 2 == 0)
                
                while (not cond) and (curr_step<all_timesteps):
                    time_ls.append(timesteps[curr_step])
                    curr_step += 1
                    cond = curr_step < 20 or curr_step > 40 or (curr_step % 2 == 0)

                # print('curr_step', curr_step, len(time_ls))
                model_input = torch.cat([intermediate_images, upscaled], dim=1)

                model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input

                # predict the noise residual
                noise_pred = self.unet(
                    model_input,
                    time_ls,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=noise_level,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(intermediate_images.shape[1], dim=1)

                # # compute the previous noisy sample x_t -> x_t-1
                # intermediate_images = self.scheduler.step(
                #     noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                # )[0]

                # compute the previous noisy sample x_t -> x_t-1
                # intermediate_images = self.scheduler.step(
                #     noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                # )[0]
                intermediate_images = multistep_pre(
                    self, noise_pred, time_ls, intermediate_images)
                
            et = time.time()
            print('unet time:', et - st, 'seconds')
            image = intermediate_images

            if output_type == "pil":
                # 9. Post-processing
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                # 10. Run safety checker
                image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

                # 11. Convert to PIL
                image = self.numpy_to_pil(image)

                # 12. Apply watermark
                if self.watermarker is not None:
                    self.watermarker.apply_watermark(image, self.unet.config.sample_size)
            elif output_type == "pt":
                nsfw_detected = None
                watermark_detected = None

                if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                    self.unet_offload_hook.offload()
            else:
                # 9. Post-processing
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                # 10. Run safety checker
                image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, nsfw_detected, watermark_detected)

            return IFPipelineOutput(images=image, nsfw_detected=nsfw_detected, watermark_detected=watermark_detected)

        return call
    pipe.call = new_call(pipe)



def register_if3(pipe):
    def new_call(self):
        @torch.no_grad()
        def call(
        prompt: Union[str, List[str]] = None,
        image = None,
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        ):
            # 1. Check inputs
            self.check_inputs(
                prompt,
                image,
                noise_level,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
            )

            if image is None:
                raise ValueError("`image` input cannot be undefined.")

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 4. Preprocess image
            image = self.image_processor.preprocess(image)
            image = image.to(dtype=prompt_embeds.dtype, device=device)

            # 5. set timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Add noise to image
            noise_level = torch.tensor([noise_level], dtype=torch.long, device=device)
            noise = randn_tensor(image.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
            image = self.low_res_scheduler.add_noise(image, noise, noise_level)

            batch_multiplier = 2 if do_classifier_free_guidance else 1
            image = torch.cat([image] * batch_multiplier * num_images_per_prompt)
            noise_level = torch.cat([noise_level] * image.shape[0])

            # 6. Prepare latent variables
            height, width = image.shape[2:]
            num_channels_latents = self.vae.config.latent_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 7. Check that sizes of image and latents match
            num_channels_image = image.shape[1]
            if num_channels_latents + num_channels_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_image`: {num_channels_image} "
                    f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                    " `pipeline.unet` or your `image` input."
                )

            # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order


            all_timesteps = len(timesteps)
            curr_step = 0
            st = time.time()
            while curr_step<all_timesteps:
                refister_time(self.unet, curr_step)

                time_ls = []
                time_ls.append(timesteps[curr_step])
                curr_step += 1

                ipow = int(np.sqrt(9 + 8*curr_step))
                cond = ipow * ipow == (9 + 8 * curr_step)
                # cond = curr_step in [0, 1, 2, 3, 5, 10, 15, 25, 35,45,55,65]
                while (not cond) and (curr_step<all_timesteps):
                    time_ls.append(timesteps[curr_step])
                    curr_step += 1

                    ipow = int(np.sqrt(9 + 8*curr_step))
                    cond = ipow * ipow == (9 + 8 * curr_step)
                    # cond = curr_step in [0, 1, 2, 3, 5, 10, 15, 25, 35,45,55,65]

                # print(curr_step, len(time_ls))
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = torch.cat([latent_model_input, image], dim=1)

                input = (latent_model_input,time_ls[0],
                    prompt_embeds,noise_level, None, None,
                    cross_attention_kwargs,None, None, None,None,None,
                    False)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    time_ls,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=noise_level,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = multistep_pre(self, noise_pred, time_ls, latents)
                
            et = time.time()
            print('unet time:', et - st, 'seconds')

            if not output_type == "latent":
                # make sure the VAE is in float32 mode, as it overflows in float16
                needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

                if needs_upcasting:
                    self.upcast_vae()

                # Ensure latents are always the same type as the VAE
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

                # cast back to fp16 if needed
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)

                image, has_nsfw_concept, _ = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            # 11. Apply watermark
            if output_type == "pil" and self.watermarker is not None:
                image = self.watermarker.apply_watermark(image)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        return call
    pipe.call = new_call(pipe)