import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Pick the fastest device available
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load the model (you can remove safety_checker=None for public deploys)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

def generate(prompt: str, steps: int, guidance: float, seed: float):
    """
    Generate one or more images from a text prompt.
    """
    # If seed > 0, use it; else let Diffusers pick a random seed.
    generator = (
        torch.Generator(device=device).manual_seed(int(seed))
        if seed and seed > 0
        else None
    )
    output = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    )
    # returns a list of PIL images
    return output.images

# Build the Gradio UI
demo = gr.Blocks()

with demo:
    gr.Markdown("# Stable Diffusion Text→Image Generation Demo")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="e.g. ‘A serene forest at dawn’")
            steps  = gr.Slider(1, 100, value=50, step=1, label="Inference Steps")
            guidance = gr.Slider(1, 15, value=7.5, step=0.1, label="Guidance Scale")
            seed   = gr.Number(value=0, label="Random Seed (0 = random)")
            btn    = gr.Button("Generate")
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images", columns=2, height="auto")

    # wire up the button
    btn.click(
        fn=generate,
        inputs=[prompt, steps, guidance, seed],
        outputs=gallery,
    )

if __name__ == "__main__":
    demo.launch()
