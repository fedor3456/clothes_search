"""Загрузка данных и PK-сэмплер для metric learning."""
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T

import config as C

MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def train_transform(size=C.IMG_SIZE):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2, 0.02),
        T.RandomGrayscale(0.05),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.3),
    ])


def eval_transform(size=C.IMG_SIZE):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def scan_folder(root):
    """Возвращает список (путь, item_id) из структуры root/<item_id>/*.jpg"""
    root = Path(root)
    samples = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in C.IMAGE_EXTS:
                samples.append((f, d.name))
    return samples


class ClothesDataset(Dataset):
    def __init__(self, root, transform, min_images=1):
        samples = scan_folder(root)
        counts = {}
        for _, item in samples:
            counts[item] = counts.get(item, 0) + 1
        self.samples = [(p, i) for p, i in samples if counts[i] >= min_images]
        if not self.samples:
            raise RuntimeError(f"В {root} не найдено картинок. Структура: {root}/<item_id>/*.jpg")
        items = sorted({i for _, i in self.samples})
        self.item_to_label = {it: n for n, it in enumerate(items)}
        self.labels = [self.item_to_label[i] for _, i in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.labels[idx]


class PKSampler(Sampler):
    """Каждый батч: P разных вещей по K фото. Нужен для batch-hard triplet loss."""

    def __init__(self, labels, p, k):
        self.p, self.k = p, k
        self.by_label = {}
        for idx, lab in enumerate(labels):
            self.by_label.setdefault(lab, []).append(idx)
        self.label_list = [l for l, ids in self.by_label.items() if len(ids) >= 2]
        if len(self.label_list) < p:
            raise RuntimeError(f"Нужно хотя бы {p} вещей с 2+ фото, найдено {len(self.label_list)}")
        self.n_batches = len(self.label_list) // p

    def __len__(self):
        return self.n_batches * self.p * self.k

    def __iter__(self):
        labels = self.label_list[:]
        random.shuffle(labels)
        for b in range(self.n_batches):
            for lab in labels[b * self.p:(b + 1) * self.p]:
                ids = self.by_label[lab]
                if len(ids) >= self.k:
                    chosen = random.sample(ids, self.k)
                else:
                    chosen = random.choices(ids, k=self.k)
                yield from chosen
