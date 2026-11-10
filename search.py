"""Поиск похожей одежды по одному фото из консоли.

Запуск:  python search.py фото.jpg --top 5
"""
import argparse

import torch
from PIL import Image

import config as C
from build_index import load_model
from dataset import eval_transform


class Searcher:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.device)
        idx = torch.load(C.INDEX_PATH)
        self.embs, self.items, self.paths = idx["embs"], idx["items"], idx["paths"]
        self.tf = eval_transform()

    @torch.no_grad()
    def search(self, img: Image.Image, top=10):
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        q = self.model(x).float().cpu()
        sims = (self.embs @ q.T).squeeze(1)
        order = sims.argsort(descending=True).tolist()
        # одна вещь = один результат (берём её лучшее фото)
        results, seen = [], set()
        for i in order:
            if self.items[i] in seen:
                continue
            seen.add(self.items[i])
            results.append({"item_id": self.items[i], "path": self.paths[i],
                            "score": round(float(sims[i]), 4)})
            if len(results) >= top:
                break
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    s = Searcher()
    for r in s.search(Image.open(args.image), args.top):
        print(f"{r['score']:.3f}  {r['item_id']}  {r['path']}")


if __name__ == "__main__":
    main()
