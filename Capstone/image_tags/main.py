from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List
import io, json
from PIL import Image

from .tagger import tag_pil_image

app = FastAPI(title="Image Tagger API", version="0.3.0")


class TagOut(BaseModel):
    filename: str
    caption: str
    tags: List[str]


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/upload", response_model=TagOut)
async def upload(
    file: UploadFile = File(..., description="PNG or JPEG image"),
    top_k: int = Query(5, ge=1, le=20, description="Maximum number of tags"),
    nouns: bool = Query(True, description="Include noun tags"),
    adjs: bool = Query(True, description="Include adjective tags"),
    verbs: bool = Query(True, description="Include verb tags"),
):
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(415, "Only PNG or JPEG supported")

    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(400, "Could not decode image")

    stem = Path(file.filename).stem or "upload"
    tags = tag_pil_image(
        img,
        stem,
        top_k=top_k,
        keep_nouns=nouns,
        keep_adjs=adjs,
        keep_verbs=verbs,
    )

    # pull the caption back out of the side-car JSON
    caption = ""
    meta = Path.home() / "Desktop" / "image_tags" / f"{stem}.json"
    if meta.exists():
        try:
            caption = json.loads(meta.read_text())["caption"]
        except:
            pass

    return JSONResponse(
        {"filename": file.filename, "caption": caption, "tags": tags}
    )
