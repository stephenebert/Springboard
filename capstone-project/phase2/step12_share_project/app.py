import os
import json
import requests
import gradio as gr

# 1) Where is the FastAPI service?
#    a. On Spaces, set this in Settings -> Variables as  API_URL=https://capstone-retrieval-api.onrender.com/search
#    b. Fallback to the literal string below if the variable is missing.
API_URL = os.getenv("API_URL", "https://capstone-retrieval-api.onrender.com/search").strip()

# 2) Helper that actually calls /search
def search(caption: str, k: int) -> dict | str:
    if not caption:
        return {"error": "Caption is empty"}
    payload = {"caption": caption, "k": int(k)}
    try:
        resp = requests.post(API_URL, json=payload, timeout=15)
        resp.raise_for_status()           # -> raise on 4xx/5xx
        return resp.json()                # Gradio JSON component can render dicts
    except requests.exceptions.HTTPError as e:
        # backend returned 422 / 500 / …
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    except requests.exceptions.RequestException as e:
        # network problems, time-outs, …
        return {"error": str(e)}

# 3) Tiny Gradio UI
with gr.Blocks(title="Image ↔ Text Retrieval") as demo:
    gr.Markdown(
        "### Image ↔ Text Retrieval\n"
        "Enter a caption, pick *k*, click **Submit** – the UI calls your FastAPI + FAISS service and "
        "shows the top-K matches."
    )
    
    caption_in = gr.Textbox(lines=2, label="caption", placeholder="e.g. king on a throne")
    k_in       = gr.Slider(1, 10, value=3, step=1, label="Top-K")

    output     = gr.JSON(label="results")

    submit_btn = gr.Button("Submit")
    submit_btn.click(search, inputs=[caption_in, k_in], outputs=output)

demo.launch()            
