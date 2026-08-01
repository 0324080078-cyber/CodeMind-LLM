"""
Local Image Generation using Stable Diffusion
100% local — no API, no cost, runs on your GPU
Uses diffusers library with Stable Diffusion v1.5 (free weights)
"""

import os
import torch
import time
from typing import Optional


class ImageGenerator:
    """
    Local Stable Diffusion image generator.
    Runs entirely on your machine — no external API.
    First run downloads ~4GB model weights (one time only).
    """

    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    OUTPUT_DIR = "./outputs/images"

    def __init__(self, device: str = "auto", model_id: Optional[str] = None):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_id or self.MODEL_ID
        self.pipe = None
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self._load_pipeline()

    def _load_pipeline(self):
        """Load Stable Diffusion pipeline."""
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            import torch

            print(f"Loading Stable Diffusion on {self.device}...")
            print("(First run downloads ~4GB — subsequent runs are instant)")

            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )

            # Use faster DPM scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            self.pipe = self.pipe.to(self.device)

            # Memory optimizations
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("xformers enabled for faster generation")
                except Exception:
                    pass

            print("Stable Diffusion loaded successfully")

        except ImportError:
            print("diffusers not installed. Run: pip install diffusers accelerate xformers")
            self.pipe = None
        except Exception as e:
            print(f"Failed to load image pipeline: {e}")
            self.pipe = None

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted, ugly, low resolution",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate an image from a text prompt.
        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid in the image
            width: Image width (must be multiple of 8)
            height: Image height (must be multiple of 8)
            num_inference_steps: More steps = better quality but slower
            guidance_scale: How closely to follow the prompt (7-12 recommended)
            seed: Random seed for reproducibility
            output_path: Where to save the image
        Returns:
            Path to the generated image
        """
        if self.pipe is None:
            return "Image generation unavailable. Install diffusers: pip install diffusers"

        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"Generating image: '{prompt[:50]}...'")
        start = time.time()

        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        image = result.images[0]
        elapsed = time.time() - start

        # Save image
        if output_path is None:
            timestamp = int(time.time())
            output_path = os.path.join(self.OUTPUT_DIR, f"generated_{timestamp}.png")

        image.save(output_path)
        print(f"Image saved: {output_path} ({elapsed:.1f}s)")
        return output_path

    def generate_batch(
        self,
        prompts: list,
        **kwargs,
    ) -> list:
        """Generate multiple images from a list of prompts."""
        return [self.generate(p, **kwargs) for p in prompts]

    def img2img(
        self,
        init_image_path: str,
        prompt: str,
        strength: float = 0.75,
        **kwargs,
    ) -> str:
        """
        Transform an existing image based on a prompt.
        strength: 0=no change, 1=completely new image
        """
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            from PIL import Image

            pipe_i2i = StableDiffusionImg2ImgPipeline(**self.pipe.components)
            pipe_i2i = pipe_i2i.to(self.device)

            init_image = Image.open(init_image_path).convert("RGB")
            init_image = init_image.resize((512, 512))

            result = pipe_i2i(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=kwargs.get("guidance_scale", 7.5),
                num_inference_steps=kwargs.get("num_inference_steps", 20),
            )

            timestamp = int(time.time())
            output_path = os.path.join(self.OUTPUT_DIR, f"img2img_{timestamp}.png")
            result.images[0].save(output_path)
            return output_path

        except Exception as e:
            return f"img2img failed: {e}"
