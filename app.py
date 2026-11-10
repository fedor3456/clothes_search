"""Веб-сайт: загружаешь фото — получаешь похожие вещи из каталога.

Запуск:  uvicorn app:app --reload
Открыть: http://127.0.0.1:8000
"""
import io
import json
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import config as C
from search import Searcher

app = FastAPI(title="ClothSearch")
searcher = Searcher()

# Необязательный файл с описанием товаров: {"item_id": {"title": ..., "url": ..., "price": ...}}
META_PATH = C.ROOT / "data" / "catalog.json"
META = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
ALLOWED = set(searcher.paths)


@app.post("/api/search")
async def api_search(file: UploadFile = File(...), top: int = 12):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Не удалось прочитать картинку")
    results = searcher.search(img, top=min(top, 50))
    for r in results:
        r["image"] = "/api/image?path=" + quote(r["path"])
        r.update(META.get(r["item_id"], {}))
    return {"results": results}


@app.get("/api/image")
def api_image(path: str):
    if path not in ALLOWED:
        raise HTTPException(404)
    return FileResponse(path)


app.mount("/", StaticFiles(directory=C.ROOT / "static", html=True), name="static")
