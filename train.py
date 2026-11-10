"""Обучение ClothNet с нуля.

Запуск:            python train.py
Продолжить:        python train.py --resume checkpoints/clothnet_last.pt
Быстрый тест:      python train.py --epochs 1
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader

import config as C
from dataset import ClothesDataset, PKSampler, train_transform
from evaluate import recall_at_k
from losses import BatchHardTripletLoss
from model import ClothNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    C.CHECKPOINT_DIR.mkdir(exist_ok=True)

    ds = ClothesDataset(C.TRAIN_DIR, train_transform(), min_images=2)
    sampler = PKSampler(ds.labels, C.P_ITEMS, C.K_IMAGES)
    loader = DataLoader(ds, batch_size=C.P_ITEMS * C.K_IMAGES, sampler=sampler,
                        num_workers=C.NUM_WORKERS, pin_memory=(device == "cuda"),
                        drop_last=True, persistent_workers=C.NUM_WORKERS > 0)
    print(f"Обучающих фото: {len(ds)}, вещей: {len(ds.item_to_label)}, батчей за эпоху: {len(loader)}")

    model = ClothNet(C.EMB_DIM, C.WIDTH).to(device)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    loss_fn = BatchHardTripletLoss(C.MARGIN)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    # warmup 3 эпохи, потом косинусное затухание
    steps_per_epoch = len(loader)
    total = args.epochs * steps_per_epoch
    warm = min(3 * steps_per_epoch, total // 5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.LR, total_steps=total,
                                                pct_start=max(warm / total, 0.01))
    use_amp = C.USE_AMP and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", 0.0)
        print(f"Продолжаю с эпохи {start_epoch}")

    has_eval = C.QUERY_DIR.exists() and C.GALLERY_DIR.exists()
    if not has_eval:
        print("Нет data/query и data/gallery — Recall@K считаться не будет.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, s_loss, s_active = time.time(), 0.0, 0.0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                emb = model(x)
            loss, active = loss_fn(emb.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < total - 1:
                sched.step()
            s_loss += loss.item()
            s_active += active.item()
            if step % 50 == 0:
                print(f"  ep {epoch} step {step}/{steps_per_epoch} loss {loss.item():.4f} "
                      f"active {active.item():.2f} lr {opt.param_groups[0]['lr']:.2e}")

        n = len(loader)
        print(f"Эпоха {epoch}: loss {s_loss / n:.4f}, active {s_active / n:.2f}, {time.time() - t0:.0f} c")

        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                 "best": best, "emb_dim": C.EMB_DIM, "width": C.WIDTH}
        torch.save(state, C.CHECKPOINT_DIR / "clothnet_last.pt")

        if has_eval and ((epoch + 1) % C.EVAL_EVERY == 0 or epoch == args.epochs - 1):
            r = recall_at_k(model, device)
            print("  " + "  ".join(f"R@{k}={v * 100:.1f}%" for k, v in r.items()))
            if r[1] >= best:
                best = r[1]
                state["best"] = best
                torch.save(state, C.MODEL_PATH)
                print(f"  Новая лучшая модель: R@1={best * 100:.1f}%")
        elif not has_eval:
            torch.save(state, C.MODEL_PATH)


if __name__ == "__main__":
    main()
