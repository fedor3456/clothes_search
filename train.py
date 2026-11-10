"""Обучение ClothNet с нуля: ArcFace + batch-hard triplet.

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
from losses import ArcFaceLoss, BatchHardTripletLoss
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
    n_items = len(ds.item_to_label)
    print(f"Обучающих фото: {len(ds)}, вещей: {n_items}, батчей за эпоху: {len(loader)}")

    model = ClothNet(C.EMB_DIM, C.WIDTH).to(device)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    tri_fn = BatchHardTripletLoss(C.MARGIN)
    arc_fn = ArcFaceLoss(n_items, C.EMB_DIM, C.ARC_S, C.ARC_M).to(device) if C.USE_ARCFACE else None

    groups = [{"params": model.parameters(), "weight_decay": C.WEIGHT_DECAY}]
    if arc_fn is not None:
        groups.append({"params": arc_fn.parameters(), "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=C.LR)

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
        if arc_fn is not None and "arc" in ck:
            arc_fn.load_state_dict(ck["arc"])
        try:
            opt.load_state_dict(ck["opt"])
        except ValueError:
            print("Состояние оптимизатора не подошло, начинаю с новым.")
        start_epoch, best = ck["epoch"] + 1, ck.get("best", 0.0)
        print(f"Продолжаю с эпохи {start_epoch}")

    has_eval = C.QUERY_DIR.exists() and C.GALLERY_DIR.exists()
    if not has_eval:
        print("Нет data/query и data/gallery — Recall@K считаться не будет.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        s_tri = s_arc = s_acc = s_active = 0.0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                emb = model(x)
            emb = emb.float()
            tri, active = tri_fn(emb, y)
            loss = C.TRIPLET_WEIGHT * tri
            arc_l = arc_acc = torch.zeros((), device=device)
            if arc_fn is not None:
                prog = min(1.0, (epoch + step / steps_per_epoch) / max(C.ARC_WARMUP, 1e-6))
                arc_l, arc_acc = arc_fn(emb, y, m=C.ARC_M * prog)
                loss = loss + C.ARC_WEIGHT * arc_l
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < total - 1:
                sched.step()
            s_tri += tri.item()
            s_arc += arc_l.item()
            s_acc += arc_acc.item()
            s_active += active.item()
            if step % 50 == 0:
                print(f"  ep {epoch} step {step}/{steps_per_epoch} triplet {tri.item():.4f} "
                      f"arc {arc_l.item():.3f} arc_acc {arc_acc.item():.2f} lr {opt.param_groups[0]['lr']:.2e}")

        n = len(loader)
        print(f"Эпоха {epoch}: triplet {s_tri / n:.4f}, arc {s_arc / n:.3f}, "
              f"arc_acc {s_acc / n:.2f}, active {s_active / n:.2f}, {time.time() - t0:.0f} c")

        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                 "best": best, "emb_dim": C.EMB_DIM, "width": C.WIDTH}
        if arc_fn is not None:
            state["arc"] = arc_fn.state_dict()
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
