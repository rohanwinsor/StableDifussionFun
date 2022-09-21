from io import BytesIO
import gradio as gr
from torch import autocast
import requests
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline

device = "cuda"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
).to(device)


def predict(dict, prompt):
    init_img = dict["image"].convert("RGB").resize((512, 512))
    mask_img = dict["mask"].convert("RGB").resize((512, 512))
    with autocast("cuda"):
        images = pipe(
            prompt=prompt, init_image=init_img, mask_image=mask_img, strength=0.75
        )["sample"]

    return images[0]


gr.Interface(
    predict,
    title="Stable Diffusion In-Painting Tool on Colab with Gradio",
    inputs=[
        gr.Image(source="upload", tool="sketch", type="pil"),
        gr.Textbox(label="prompt"),
    ],
    outputs=[gr.Image()],
).launch(share=True)
