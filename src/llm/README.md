# LLM validation bench (Colab-ready)

Цель: быстро прогнать несколько локальных LLM (HF веса) как пост‑валидатор на готовых VLM артефактах. Все лежит в `src/llm`, данные — в `src/llm/data`.

## Что есть
- `data/google.jpg`, `data/google-3.txt` — вход и raw VLM ответ.
- `data/summary_vlm_orchestrator-6.json` — пример вывода оркестратора.
- `data/google_golden.json` — эталон meta/items для google.jpg.
- `data/sample_payload.json` — payload для LLM (meta/items + required_fields/amount_tolerance).
- `models.py` — реестр моделей (llama31-8b/70b, qwen2-7b/72b, mistral-7b, mixtral-8x7b, deepseek-7b, phi3-medium).
- `runner.py` — единый запуск модели.
- `colab_eval.ipynb` — ноутбук для поочередного прогона всех моделей в Colab.

## Быстрый старт (локально)
```bash
PYTHONPATH=. python -m src.llm.runner \
  --model qwen2-7b \
  --payload src/llm/data/sample_payload.json \
  --output src/llm/data/out_qwen2-7b.json
```
Требования: `transformers`, `accelerate`, `bitsandbytes` (для 4/8bit). Для Llama 3.1 нужен HF token.

## Colab
1) Смонтируй репозиторий в `/content/OCR` (или поправь `repo_path` в ноутбуке).
2) Открой `src/llm/colab_eval.ipynb`.
3) Запускай ячейки сверху вниз — установки, загрузка payload, прогон моделей, сохранение результатов в `src/llm/data/out/`.

## Формат payload
- `meta`, `items` — нормализованный JSON после VLM.
- `required_fields`, `amount_tolerance` — правила валидации.
- `raw_vlm`, `schema_errors` — опционально для контекста.
