"""Раскладывает DeepFashion In-shop Clothes Retrieval по папкам проекта.

1. Скачай датасет (In-shop Clothes Retrieval Benchmark) со страницы DeepFashion:
   https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html
   Нужны: img.zip (или img_highres.zip) и Eval/list_eval_partition.txt
2. Распакуй, чтобы получилось:  <inshop>/img/...  и  <inshop>/list_eval_partition.txt
3. Запусти:  python prepare_inshop.py --src путь/к/inshop

Результат: data/train/<item_id>/, data/query/<item_id>/, data/gallery/<item_id>/
"""
import argparse
import shutil
from pathlib import Path

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="папка, где лежат img/ и list_eval_partition.txt")
    ap.add_argument("--link", action="store_true", help="делать ссылки вместо копий (экономит место)")
    args = ap.parse_args()
    src = Path(args.src)

    part = next(iter(src.rglob("list_eval_partition.txt")), None)
    if part is None:
        raise SystemExit("Не найден list_eval_partition.txt")
    lines = part.read_text().splitlines()[2:]  # 1-я строка: число, 2-я: заголовок

    dirs = {"train": C.TRAIN_DIR, "query": C.QUERY_DIR, "gallery": C.GALLERY_DIR}
    n = {k: 0 for k in dirs}
    for line in lines:
        parts = line.split()
        if len(parts) != 3:
            continue
        rel, item, status = parts
        img = src / rel
        if not img.exists():
            continue
        # имя файла делаем уникальным: категория + исходное имя
        cat = "_".join(Path(rel).parts[1:3])
        dst = dirs[status] / item / f"{cat}_{img.name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            if args.link:
                try:
                    dst.hardlink_to(img)
                except OSError:
                    shutil.copy2(img, dst)
            else:
                shutil.copy2(img, dst)
        n[status] += 1
    print("Готово:", ", ".join(f"{k}={v}" for k, v in n.items()))


if __name__ == "__main__":
    main()
