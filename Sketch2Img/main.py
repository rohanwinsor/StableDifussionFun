from torch import autocast
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import gradio as gr

device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
).to(device)


def predict(image, prompt):
    prompt += "photorealistic"
    init_img = image.convert("RGB").resize((512, 512))
    with autocast("cuda"):
        images = pipe(prompt=prompt, init_image=init_img, strength=0.8)["sample"]

    return images[0]


gr.Interface(
    predict,
    title="Sketch2Image",
    inputs=[gr.Image(source="canvas", type="pil"), gr.Textbox(label="prompt")],
    outputs=[gr.Image()],
).launch(share=True)
