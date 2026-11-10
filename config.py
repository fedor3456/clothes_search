"""Общие настройки проекта. Меняй значения здесь, а не в коде."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Папки с данными. Внутри каждой: <item_id>/<картинки>.jpg
TRAIN_DIR = ROOT / "data" / "train"
QUERY_DIR = ROOT / "data" / "query"      # фото-запросы для оценки
GALLERY_DIR = ROOT / "data" / "gallery"  # «каталог», в котором ищем

CHECKPOINT_DIR = ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "clothnet_best.pt"
INDEX_PATH = CHECKPOINT_DIR / "index.pt"

# Модель
IMG_SIZE = 224
EMB_DIM = 128
WIDTH = 48          # ширина первого слоя; 32 -> ~11M параметров, 48 -> ~25M, 16 -> ~3M

# Обучение
EPOCHS = 100
P_ITEMS = 16        # вещей в батче
K_IMAGES = 4        # фото каждой вещи в батче -> batch = P*K = 64
LR = 1e-3
WEIGHT_DECAY = 5e-4
MARGIN = 0.3
NUM_WORKERS = 4     # на Windows при проблемах поставь 0
EVAL_EVERY = 2      # считать Recall@K каждые N эпох
USE_AMP = True      # смешанная точность на GPU (быстрее и меньше памяти)

# ArcFace
USE_ARCFACE = True
ARC_S = 30.0            # масштаб логитов
ARC_M = 0.4             # угловой зазор (рад); если не сходится — 0.3, если легко — 0.5
ARC_WARMUP = 8          # эпох, за которые зазор растёт от 0 до ARC_M
ARC_WEIGHT = 1.0
TRIPLET_WEIGHT = 1.0

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
