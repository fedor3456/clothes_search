"""Оценка качества: Recall@K на парах query/gallery.

Recall@K = доля запросов, у которых правильная вещь есть среди K ближайших.

Запуск:  python evaluate.py
"""
import argparse

import torch
from torch.utils.data import DataLoader

import config as C
from dataset import ClothesDataset, eval_transform
from model import ClothNet


@torch.no_grad()
def extract(model, root, device, batch_size=128):
    ds = ClothesDataset(root, eval_transform())
    loader = DataLoader(ds, batch_size=batch_size, num_workers=C.NUM_WORKERS)
    model.eval()
    embs = []
    for x, _ in loader:
        embs.append(model(x.to(device)).float().cpu())
    items = [item for _, item in ds.samples]
    return torch.cat(embs), items, [p for p, _ in ds.samples]


@torch.no_grad()
def recall_at_k(model, device, ks=(1, 5, 10, 20)):
    q, q_items, _ = extract(model, C.QUERY_DIR, device)
    g, g_items, _ = extract(model, C.GALLERY_DIR, device)
    sims = q @ g.T
    topk = sims.topk(min(max(ks), g.shape[0]), dim=1).indices
    g_items_arr = [g_items[i] for i in range(len(g_items))]
    hits = {k: 0 for k in ks}
    for qi, row in enumerate(topk.tolist()):
        found = [g_items_arr[j] == q_items[qi] for j in row]
        for k in ks:
            if any(found[:k]):
                hits[k] += 1
    return {k: hits[k] / len(q_items) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.MODEL_PATH))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = ClothNet(ckpt["emb_dim"], ckpt["width"]).to(device)
    model.load_state_dict(ckpt["model"])
    r = recall_at_k(model, device)
    for k, v in r.items():
        print(f"Recall@{k}: {v * 100:.2f}%")


if __name__ == "__main__":
    main()
