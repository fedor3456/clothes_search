# ClothSearch — поиск одежды по фото своей нейросетью

Собственная CNN (ResNet-подобная, ~11M параметров), обученная **с нуля** на PyTorch.
Модель превращает фото в вектор из 128 чисел. Фото одной вещи дают близкие векторы,
поэтому поиск сводится к поиску ближайших векторов в каталоге.

## Структура

| Файл | Что делает |
|---|---|
| `config.py` | Все настройки: пути, размер сети, батч, lr, эпохи |
| `model.py` | Архитектура ClothNet: ResBlock, GeM-пулинг, голова-эмбеддинг |
| `losses.py` | Batch-hard triplet loss |
| `dataset.py` | Датасет, аугментации, PK-сэмплер (P вещей × K фото в батче) |
| `train.py` | Обучение с warmup, AMP, чекпоинтами и подсчётом Recall@K |
| `evaluate.py` | Recall@1/5/10/20 на query/gallery |
| `build_index.py` | Считает векторы всего каталога и сохраняет индекс |
| `search.py` | Поиск из консоли: `python search.py фото.jpg` |
| `app.py` | Сайт на FastAPI |
| `static/` | Фронтенд: `index.html`, `style.css`, `app.js` |
| `prepare_inshop.py` | Раскладывает датасет DeepFashion In-shop по папкам |
| `make_demo_data.py` | Игрушечный датасет для проверки, что всё запускается |

## Установка (Windows, PowerShell)

```powershell
cd clothsearch
python -m venv .venv
.venv\Scripts\Activate.ps1
# PyTorch с CUDA — точную команду возьми на pytorch.org под свою версию CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # должно быть True
```

## Шаг 0. Проверка на демо-данных (2 минуты)

```powershell
python make_demo_data.py
python train.py --epochs 4
python evaluate.py
```

Если видишь строки `R@1=...%`, значит, конвейер работает целиком.

## Шаг 1. Настоящие данные

1. Скачай **In-shop Clothes Retrieval Benchmark** со страницы
   [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html):
   файлы `img.zip` и `Eval/list_eval_partition.txt`.
2. Распакуй в любую папку, например `D:\datasets\inshop`.
3. Разложи по папкам проекта:

```powershell
python prepare_inshop.py --src D:\datasets\inshop --link
```

Флаг `--link` создаёт жёсткие ссылки вместо копий и экономит место (нужен тот же диск).

Можно использовать и свои данные. Главное, соблюсти структуру:
```
data/train/<id_вещи>/*.jpg     # минимум 2 фото на вещь
data/query/<id_вещи>/*.jpg     # фото-запросы для проверки
data/gallery/<id_вещи>/*.jpg   # каталог, в котором ищем
```

## Шаг 2. Обучение

```powershell
python train.py
```

- Лучшая модель по Recall@1 сохраняется в `checkpoints/clothnet_best.pt`.
- После каждой эпохи пишется `checkpoints/clothnet_last.pt`. Продолжить обучение можно так: `python train.py --resume checkpoints/clothnet_last.pt`.
- `active` в логах — доля «трудных» троек, на которых модель ещё ошибается. Это значение должно падать.

Если не хватает памяти видеокарты, уменьши в `config.py` `P_ITEMS` (например до 8)
или `WIDTH` до 16. Если DataLoader зависает на Windows, поставь `NUM_WORKERS = 0`.

## Шаг 3. Индекс и поиск

```powershell
python build_index.py
python search.py мое_фото.jpg --top 5
```

## Шаг 4. Сайт

```powershell
uvicorn app:app --reload
```

Открой http://127.0.0.1:8000, загрузи фото и посмотри на похожие вещи.

Чтобы у результатов были названия, цены и ссылки на магазин, создай `data/catalog.json`
(пример в `catalog.example.json`):

```json
{
  "id_00000002": {"title": "Платье миди", "price": "3 990 ₽", "url": "https://..."}
}
```

## Чего ждать

- Сеть с нуля на In-shop (~25k обучающих фото) обычно выходит на Recall@1 около 50–70%
  за 60–100 эпох. Предобученные модели дают ~90%, но это уже не «с нуля».
- Точное совпадение возможно, только если эта вещь есть в твоём каталоге.

## Идеи для улучшения

1. Больше эпох и больше данных (DeepFashion2, свои фото).
2. Детектор одежды перед энкодером: вырезать вещь из кадра.
3. Увеличить `IMG_SIZE` до 256 и `WIDTH` до 48.
4. Добавить к triplet loss классификационную голову (ArcFace) — обычно +5–10% к R@1.
5. FAISS вместо матричного умножения, когда каталог станет больше ~100k фото.
