"""Генерирует игрушечный датасет, чтобы проверить, что весь конвейер работает,
ещё до скачивания настоящих данных. Каждая «вещь» — цветная фигура с узором.

Запуск:  python make_demo_data.py
Потом:   python train.py --epochs 3
"""
import random
import shutil

from PIL import Image, ImageDraw, ImageFilter

import config as C

SHAPES = ["tshirt", "dress", "pants", "jacket"]


def draw_item(seed, variant):
    rnd = random.Random(seed)
    color = tuple(rnd.randint(30, 230) for _ in range(3))
    stripe = tuple(rnd.randint(0, 255) for _ in range(3))
    shape = rnd.choice(SHAPES)
    n_stripes = rnd.randint(0, 5)

    v = random.Random(seed * 1000 + variant)
    bg = tuple(v.randint(200, 255) for _ in range(3))
    img = Image.new("RGB", (256, 256), bg)
    d = ImageDraw.Draw(img)
    ox, oy = v.randint(-20, 20), v.randint(-20, 20)
    if shape == "tshirt":
        poly = [(70, 60), (110, 50), (146, 50), (186, 60), (210, 100), (180, 110),
                (180, 210), (76, 210), (76, 110), (46, 100)]
    elif shape == "dress":
        poly = [(100, 40), (156, 40), (160, 100), (200, 220), (56, 220), (96, 100)]
    elif shape == "pants":
        poly = [(80, 40), (176, 40), (190, 220), (140, 220), (128, 110), (116, 220), (66, 220)]
    else:
        poly = [(60, 50), (196, 50), (220, 210), (140, 210), (128, 120), (116, 210), (36, 210)]
    poly = [(x + ox, y + oy) for x, y in poly]
    d.polygon(poly, fill=color)
    for s in range(n_stripes):
        y = 70 + s * 30 + oy
        d.line([(40, y), (216, y)], fill=stripe, width=6)
    d.polygon(poly, outline=(0, 0, 0))
    img = img.rotate(v.uniform(-15, 15), fillcolor=bg)
    if v.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(v.uniform(0, 1.5)))
    return img


def main(n_train=120, n_test=40, per_item=5):
    for d in (C.TRAIN_DIR, C.QUERY_DIR, C.GALLERY_DIR):
        if d.exists():
            shutil.rmtree(d)
    for i in range(n_train):
        out = C.TRAIN_DIR / f"item_{i:04d}"
        out.mkdir(parents=True)
        for v in range(per_item):
            draw_item(i, v).save(out / f"{v}.jpg")
    for i in range(n_train, n_train + n_test):
        q = C.QUERY_DIR / f"item_{i:04d}"
        g = C.GALLERY_DIR / f"item_{i:04d}"
        q.mkdir(parents=True)
        g.mkdir(parents=True)
        draw_item(i, 0).save(q / "0.jpg")
        for v in range(1, per_item):
            draw_item(i, v).save(g / f"{v}.jpg")
    print(f"Демо-данные готовы: train={n_train} вещей, query/gallery={n_test} вещей")


if __name__ == "__main__":
    main()
