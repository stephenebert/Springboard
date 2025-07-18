# Image-to-Text Retrieval Demo  (BLIP → CLIP → FAISS)

Turn any image into a short caption with **BLIP**, embed that caption with **CLIP**, and retrieve the most similar human-written captions from MS-COCO using an in-memory **FAISS** index – all wrapped in a simple **Gradio** UI.

![Screenshot of the Gradio demo UI](extra_exploration/data/UI1.png)


---

## What It Does

1. **Upload an image**  
2. **BLIP** generates a caption  
3. **CLIP** encodes that caption to a 512-D embedding  
4. **FAISS** finds the *k* most similar captions from a pre-embedded COCO corpus  
5. Ranked results (distance ↓ = similarity ↑) are displayed

---

## Repository Layout

``` bash
├── gradio_demo.py # ← main app (run this)
├── requirements.txt # ← pip deps (loose pins)
├── environment.yml # ← exact conda env (tight pins)
├── scripts/
│ ├── coco_caption_clip.index # 591 753 × 512 float32 vectors
│ └── coco_caption_texts.npy # array of captions aligned with index order
└── docs/
└── demo_screenshot.png
```
> *The FAISS index + captions array are a few-hundred MB; use Git LFS if pushing to GitHub.*

---

## Quick Start (Conda — recommended)

```bash
git clone https://github.com/<your-handle>/image2text-faiss-demo.git
cd image2text-faiss-demo

# create the exact working env  (Python 3.10 · NumPy 2.x · SciPy 1.15 · FAISS 1.11 …)
conda env create -f environment.yml
conda activate capstone-gradio-py310

python gradio_demo.py
```
Open the URL printed in the terminal (output in terminal is like ```http://127.0.0.1:7860```) and drop an image.

Need a public link? Edit the last line of ```gradio_demo.py → demo.launch(share=True)```.

## Quick Start (pip / virtualenv)
```
bash
python -m venv .venv
source .venv/bin/activate          # Windows → .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python gradio_demo.py
```
## ```requirements.txt``` (minimum tested versions)
``` bash
numpy>=2.2
scipy>=1.13
faiss-cpu>=1.11
torch>=2.2          # CPU or MPS wheels both work
transformers>=4.41
sentence-transformers>=2.7
gradio>=4.27
pillow
tqdm
pydub
pyaudioop; python_version>="3.13"   # shim for audioop removal in Py3.13+
```
## Apple-Silicon Speed-Up (optional)
PyTorch ≥ 2.2 bundles Metal support. Tell Sentence-Transformers to use it:

``` bash
clip_model = SentenceTransformer("clip-ViT-B-32", device="mps")
```
≈ 2-3 times faster caption embedding on an M-series GPU.

## Re-building the FAISS Index (optional / advanced)

1. Create captions.txt - one caption per line
2. Run:
``` bash
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, tqdm

CAPTIONS = [l.strip() for l in open("captions.txt")]
model    = SentenceTransformer("clip-ViT-B-32", device="cpu")   # "mps"/"cuda" ok

vecs = model.encode(
    CAPTIONS, normalize_embeddings=True,
    convert_to_numpy=True, show_progress_bar=True
).astype("float32")

index = faiss.IndexFlatL2(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, "scripts/coco_caption_clip.index")
np.save("scripts/coco_caption_texts.npy", np.array(CAPTIONS, dtype=object))
```
## Troubleshooting
| Error / Symptom                                | Cause                           | Fix                                                                                |
| ---------------------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------- |
| `ValueError: input not a numpy array` (FAISS)  | NumPy / FAISS ABI mismatch      | Use **NumPy 2.x** with **faiss-cpu 1.11** (or rebuild FAISS against current NumPy) |
| `ModuleNotFoundError: scipy`                   | SciPy missing                   | `conda install -c conda-forge scipy>=1.13`                                         |
| `ModuleNotFoundError: audioop` on Python 3.13+ | `audioop` removed from stdlib   | `pip install pyaudioop` **or** run on Python ≤ 3.12                                |
| Long SciPy compile via pip                     | Building from source            | Use conda-forge wheel: `conda install -c conda-forge scipy`                        |
| FAISS dimension mismatch                       | Wrong embedding model vs. index | Rebuild index with the same model (`clip-ViT-B-32`)                                |

## Minimal Smoke Tests
``` bash
python - <<'PY'
import numpy as np, faiss
faiss.IndexFlatL2(512).add(np.random.rand(1,512).astype("float32"))
print("FAISS sanity-check passed")
PY
```

``` bash
python - <<'PY'
import numpy, scipy
print("NumPy:", numpy.__version__, "| SciPy:", scipy.__version__)
PY
```
All should run without traceback.
## Credits
1. BLIP — Salesforce Research

2. CLIP — OpenAI

3. FAISS — Meta AI

4. Gradio — Hugging Face

5. MS-COCO captions — COCO Consortium
