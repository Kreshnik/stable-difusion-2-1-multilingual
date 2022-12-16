import torch

import os.path
from slugify import slugify
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from googletrans import Translator
import gradio as gr


def generate(language_code, description):
    if language_code != "en":
        translator = Translator()
        translation = translator.translate(description, src=language_code)
        text = translation.text
    else:
        text = description

    model_id = "stabilityai/stable-diffusion-2-1"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.enable_attention_slicing()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    images = pipe(text).images
    images[0].save(os.path.join('images', slugify(description) + ".png"))
    return images[0]


app = gr.Interface(
    fn=generate,
    title="StableDiffusion 2.1 Multilingual",
    inputs=
    [
        gr.Textbox(value="sq", label="Language code", lines=1, placeholder="What is your language code"),
        gr.Textbox(label="What do you want to see", lines=2, placeholder="Describe the image in your language")
    ],
    allow_flagging="never",
    outputs="image",
)

if __name__ == "__main__":
    app.launch()
