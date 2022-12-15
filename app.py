import torch
import os.path
from slugify import slugify
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from googletrans import Translator
from InquirerPy import inquirer

translator = Translator()

language_code = inquirer.text(message="What is your language code?").execute()
description = inquirer.text(message="Describe the image in your language:").execute()

translation = translator.translate(description, src=language_code)

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.enable_attention_slicing()
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

images = pipe(translation.text).images
images[0].save(os.path.join('images', slugify(description) + ".png"))
images[0].show()