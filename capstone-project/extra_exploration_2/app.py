import gradio as gr
import torch
import functools
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_OPTS = {
    "SD v1.5 (base)":        "runwayml/stable-diffusion-v1-5",
    "SD-Turbo (ultra-fast)": "stabilityai/sd-turbo"
}

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()     else
    "cpu"
)
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

@functools.lru_cache(maxsize=len(MODEL_OPTS))
def get_pipeline(model_id: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        safety_checker=None
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def generate(prompt, steps, guidance, seed, model_name):
    model_id = MODEL_OPTS[model_name]
    if "Turbo" in model_name:
        steps = min(int(steps), 4)
    pipe = get_pipeline(model_id)
    generator = None if seed == 0 else torch.manual_seed(int(seed))
    imgs = pipe(
        prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator
    ).images
    return imgs

with gr.Blocks() as demo:
    gr.Markdown("## Model-Switcher Stable Diffusion Demo")
    prompt     = gr.Textbox("Retro robot in neon city", label="Prompt")
    checkpoint = gr.Dropdown(list(MODEL_OPTS.keys()), value="SD v1.5 (base)", label="Checkpoint")
    steps      = gr.Slider(1, 50, value=30, label="Inference Steps")
    guidance   = gr.Slider(1, 15, value=7.5, label="Guidance Scale")
    seed       = gr.Number(0, label="Seed (0=random)")
    btn        = gr.Button("Generate")
    gallery    = gr.Gallery(label="Gallery", columns=2, height="auto")

    btn.click(
        fn=generate,
        inputs=[prompt, steps, guidance, seed, checkpoint],
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch()
