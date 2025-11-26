"""Run quick PoC for several VLM models on invoice images."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

import torch

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORT", "1")

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)


def _resolve_model_source(model_id: str) -> str:
    path = Path(model_id).expanduser()
    return str(path) if path.exists() else model_id


@dataclass
class ModelConfig:
    name: str
    model_id: str
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.1


class VLMRunner:
    """Helper that loads a model/processor pair once and reuses it."""

    def __init__(self, cfg: ModelConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
        model_source = _resolve_model_source(cfg.model_id)
        self.processor, self.model = self._load_model_and_processor(model_source, dtype)
        self.model.to(self.device, dtype=dtype)
        self.model.eval()

    def _load_model_and_processor(self, model_source: str, dtype: torch.dtype):
        config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        model_type = getattr(config, "model_type", "").lower()
        lower_source = model_source.lower()

        # Phi-3 Vision
        if "phi3" in model_type or "phi-3" in lower_source:
            processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    use_flash_attention_2=False,
                )
            return processor, model

        # Llava Next models
        if "llava" in model_type or "llava" in lower_source:
            try:
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

                processor = LlavaNextProcessor.from_pretrained(model_source, trust_remote_code=True)
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_source,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
                return processor, model
            except Exception:
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

                processor = LlavaNextProcessor.from_pretrained(model_source, trust_remote_code=True)
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_source,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
                return processor, model

        # Fallback generic causal LM
        processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return processor, model

    def generate(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        prompt = self.cfg.prompt.strip()
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
        }

        start = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - start

        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        response = decoded[0] if decoded else ""
        info = {"elapsed_seconds": elapsed, **gen_kwargs}
        return response, info


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if device_arg == "mps" or (device_arg == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def load_models_config(path: Path) -> List[ModelConfig]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])
    configs: List[ModelConfig] = []
    for entry in candidates:
        try:
            cfg = ModelConfig(
                name=entry["name"],
                model_id=entry["model_id"],
                prompt=entry["prompt"],
                max_new_tokens=int(entry.get("max_new_tokens", 512)),
                temperature=float(entry.get("temperature", 0.1)),
            )
            configs.append(cfg)
        except KeyError as exc:  # pragma: no cover - config hygiene
            raise ValueError(f"Invalid model entry: missing {exc}") from exc
    return configs


def iter_documents(documents_path: Path) -> Iterable[Path]:
    if documents_path.is_file():
        yield documents_path
        return
    for path in sorted(documents_path.rglob("*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}:
            yield path


def extract_images(path: Path, max_pages: int) -> List[Tuple[int, Image.Image]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = convert_from_path(path, dpi=300, fmt="png")
        return [(idx, page) for idx, page in enumerate(pages[:max_pages])]
    image = Image.open(path).convert("RGB")
    return [(0, image)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_result(output_path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM PoC on documents.")
    parser.add_argument("--documents", type=Path, default=Path("../data/input"), help="Folder or file with invoices")
    parser.add_argument("--models-file", type=Path, default=Path("models.yaml"), help="YAML file with candidate models")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Destination for run artifacts")
    parser.add_argument("--max-pages", type=int, default=1, help="For multi-page PDFs convert only this many pages")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Skip model inference (for testing)")
    args = parser.parse_args()

    configs = load_models_config(args.models_file)
    if not configs:
        raise ValueError("models.yaml does not contain any candidates")
    documents = list(iter_documents(args.documents))
    if not documents:
        raise FileNotFoundError(f"No documents found under {args.documents}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / timestamp
    ensure_dir(run_dir)

    summary: List[Dict[str, object]] = []
    device = resolve_device(args.device)

    for cfg in configs:
        model_dir = run_dir / cfg.name
        ensure_dir(model_dir)
        print(f"=== Model {cfg.name} ({cfg.model_id}) on device {device} ===")
        if args.dry_run:
            runner = None
        else:
            try:
                runner = VLMRunner(cfg, device)
            except Exception as exc:  # pragma: no cover - runtime guard
                error_payload = {
                    "model": cfg.name,
                    "model_id": cfg.model_id,
                    "status": "load_failed",
                    "error": str(exc),
                }
                summary.append(error_payload)
                print(f"[ERROR] Failed to load {cfg.name}: {exc}")
                continue

        for doc_path in tqdm(documents, desc=f"{cfg.name}"):
            try:
                images = extract_images(doc_path, args.max_pages)
            except Exception as exc:  # pragma: no cover - conversion guard
                print(f"[WARN] Failed to load {doc_path}: {exc}")
                continue

            for page_index, image in images:
                payload = {
                    "document": str(doc_path),
                    "page_index": page_index,
                    "model": cfg.name,
                    "model_id": cfg.model_id,
                    "prompt": cfg.prompt.strip(),
                    "generation_kwargs": {
                        "max_new_tokens": cfg.max_new_tokens,
                        "temperature": cfg.temperature,
                    },
                }
                if args.dry_run:
                    payload["raw_response"] = ""
                    payload["elapsed_seconds"] = 0.0
                else:
                    response, info = runner.generate(image)
                    payload["raw_response"] = response
                    payload.update(info)

                output_path = model_dir / f"{doc_path.stem}_p{page_index}.json"
                save_result(output_path, payload)
                summary.append(payload)

    report_path = run_dir / "report.json"
    save_result(report_path, {"documents": len(documents), "runs": summary})
    print(f"Run completed. Results stored in {run_dir}")


if __name__ == "__main__":
    main()
