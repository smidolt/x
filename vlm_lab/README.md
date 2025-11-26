## VLM Lab (Step 12)

Экспериментальная площадка для задач **UMBRAVA-INV-012**: сравнение нескольких vision-language моделей на наших инвойсах без переписывания основного пайплайна.  
Цель – быстро прогнать кандидатов, собрать сырые ответы и понять, какая модель подходит для последующей интеграции.

### Структура

```
vlm_lab/
├── README.md                # текущее описание и порядок работы
├── requirements.txt         # отдельные зависимости для VLM PoC
├── models.yaml              # список кандидатов + их промпты
├── run_poc.py               # основной скрипт прогонов
├── results/                 # сюда складываются логи/JSON с ответами
└── cache/                   # опционально использовать для локальных весов
```

### Подготовка окружения

```bash
cd vlm_lab
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Примечание:** скрипт автоматически выставляет `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, поэтому при наличии кеша модели берутся локально. При необходимости можно переопределить переменные окружения перед запуском.  
> Также включён `TRANSFORMERS_NO_TORCHVISION_IMPORT=1`, поэтому torchvision не обязателен.  
> Если окружение не имеет выхода в интернет, скачайте веса заранее (см. ниже) и используйте offline-кеш `~/.cache/huggingface`.

### Запуск эксперимента

```bash
python run_poc.py \
  --documents ../data/input \
  --models-file models.yaml \
  --output-dir results \
  --max-pages 1 \
  --device auto
```

- `--documents` — папка с JPG/PNG/PDF (по умолчанию используется уже существующий `data/input`).
- `--models-file` — конфиг с перечнем VLM кандидатов (можно создать несколько конфигов для разных сетов).
- `--output-dir` — куда складывать результаты. Для каждого прогонов создаётся подпапка `results/<timestamp>/<model>/<document>.json`.
- `--max-pages` — сколько страниц превращать в изображения (для PDF).
- `--device` — `auto | cpu | cuda | mps`.
- `--dry-run` — если нужно проверить пайплайн без фактического вызова моделей (полезно при отладке).
- Если модели отсутствуют в локальном кеше и нет сети, используйте `huggingface-cli download <model_id>` заранее и положите снапшот в `~/.cache/huggingface`. Пример:

```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ~/.cache/huggingface/qwen2-vl-2b
huggingface-cli download microsoft/Phi-3-vision-128k-instruct --local-dir ~/.cache/huggingface/phi3-vision
```

После этого в `models.yaml` можно либо оставить исходный `model_id` (скрипт возьмёт данные из кеша), либо явно указать локальный путь.

### Формат вывода

Каждый JSON содержит:
- исходный документ и страницу,
- имя модели и промпт,
- фактический текст/JSON, который вернула модель,
- время выполнения, параметры генерации,
- базовую статистику (число токенов, вероятность/score, если модель возвращает).

Пример (усечённый):

```json
{
  "document": "Pager.jpg",
  "page_index": 0,
  "model": "qwen-vl-mini",
  "prompt": "...",
  "raw_response": "Seller: PagerDuty Inc.; ...",
  "elapsed_seconds": 4.21,
  "generation_kwargs": {
    "max_new_tokens": 512,
    "temperature": 0.2
  }
}
```

### Как сравнивать результаты

1. Запустить `run_poc.py` на выбранном наборе документов.
2. Открыть `results/<timestamp>/report.json` — агрегированный файл со сравнением (скрипт создаёт автоматически).
3. Ручная проверка: сравнить `raw_response` с эталонным `output/json/*.json`.
4. Отметить в модели `models.yaml` колонку `status` → `chosen | rejected | needs_more_data`.

### Дальнейшие шаги

- После выбора модели переносим выводы из `results/` в wiki/Confluence.
- На основе лучших промптов формируем требования для шагов 13–15.
- Сохраняем скачанные веса в `vlm_lab/cache` и документируем пути.

### Известные нюансы загрузки моделей

- Phi-3-Vision требует `torchvision`; оно теперь прописано в `vlm_lab/requirements.txt`.
- Зависимость `flash-attn` необязательна: загрузка сначала пробует FlashAttention2, а при отсутствии автоматически откатывается на стандартный SDPA.
- Для Llava v1.5/1.6/Next используем `AutoModelForVision2Seq` + `AutoProcessor`, поэтому конфиги `LlavaConfig/LlavaNextConfig` подхватываются без ручных классов.

> **Важно:** PoC репа живёт внутри текущего проекта. Не трогаем основной `requirements.txt` и не ломаем существующий CLI. Когда переход через PoC закончится, лучшие практики переносим в основные `src/`.
