"""Stable Diffusion — local image generation."""

import os
import time
import torch
from typing import Optional


class VisionEngine:
    OUTPUT_DIR = "./outputs/images"

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", cache_dir="./models/stable-diffusion", device="auto"):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.pipe = None
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        self._load()

    def _load(self):
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, cache_dir=self.cache_dir, safety_checker=None, requires_safety_checker=False)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
            print("Stable Diffusion loaded")
        except ImportError:
            print("Install: pip install diffusers accelerate")
        except Exception as e:
            print(f"Vision failed: {e}")

    def generate(self, prompt, negative_prompt="blurry, bad quality", width=512, height=512, num_inference_steps=20, guidance_scale=7.5, seed=None, output_path=None):
        if self.pipe is None:
            return "./outputs/images/placeholder.png"
        width = (width // 8) * 8
        height = (height // 8) * 8
        generator = torch.Generator(self.device).manual_seed(seed) if seed else None
        t0 = time.time()
        ctx = torch.autocast(self.device) if self.device == "cuda" else torch.no_grad()
        with ctx:
            result = self.pipe(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
        if not output_path:
            output_path = os.path.join(self.OUTPUT_DIR, f"img_{int(time.time())}.png")
        result.images[0].save(output_path)
        print(f"Image saved: {output_path} ({time.time()-t0:.1f}s)")
        return output_path
