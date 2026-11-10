"""Строит индекс каталога: векторы всех картинок из data/gallery.

Запуск:  python build_index.py            (по умолчанию data/gallery)
         python build_index.py --dir путь/к/каталогу
"""
import argparse
from pathlib import Path

import torch

import config as C
from evaluate import extract
from model import ClothNet


def load_model(device):
    ckpt = torch.load(C.MODEL_PATH, map_location=device)
    model = ClothNet(ckpt["emb_dim"], ckpt["width"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(C.GALLERY_DIR))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    embs, items, paths = extract(model, Path(args.dir), device)
    rel = [str(Path(p).resolve()) for p in paths]
    torch.save({"embs": embs, "items": items, "paths": rel}, C.INDEX_PATH)
    print(f"Индекс: {len(items)} картинок, {len(set(items))} вещей -> {C.INDEX_PATH}")


if __name__ == "__main__":
    main()
