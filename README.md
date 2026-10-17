# clothes_search

Учебный ML-проект: поиск одежды по фотографии вещи и бирки.

## Текущий этап

Подготовка окружения и проверка GPU. Обучение, OCR и поиск товаров пока не реализованы.

## Установка на Windows

Команды для cmd из корня проекта:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Установите torch и torchvision командой из официального конфигуратора:
https://pytorch.org/get-started/locally/
Выберите Windows, Pip, Python и CUDA-сборку для NVIDIA GPU.

Затем установите остальные зависимости:

```bat
python -m pip install -r requirements.txt
python scripts/check_gpu.py
```

Проверка должна показать доступность CUDA и название видеокарты. Скрипт завершится с ненулевым кодом, если GPU недоступна.

## Структура

- scripts/ — запускаемые скрипты.
- src/clothes_search/ — основной код проекта.
- notebooks/ — учебные эксперименты.
- tests/ — будущие тесты.

Данные, веса моделей, секреты и виртуальное окружение не добавляем в Git. Зависимости пока не зафиксированы по версиям; файл requirements.txt не является lock-файлом.
