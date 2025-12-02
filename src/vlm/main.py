"""Standalone VLM runners: block detection (heuristic) and Qwen2-VL reasoner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.vlm.blocks import BlockDetector
from src.vlm.reasoner import VLMReasoner
from src.config import VLMConfig
from src.ocr.engine import OCRWord


def _load_ocr_words(path: Path) -> list[OCRWord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words = []
    for w in data.get("words", []):
        words.append(
            OCRWord(
                text=w.get("text", ""),
                bbox=w.get("bbox", [0, 0, 0, 0]),
                confidence=w.get("confidence"),
                page_num=w.get("page_num", 1),
                block_num=w.get("block_num", 0),
                line_num=w.get("line_num", 0),
                word_num=w.get("word_num", 0),
            )
        )
    return words


def run_blocks(ocr_json: Path, output_path: Path, cfg: VLMConfig) -> dict:
    words = _load_ocr_words(ocr_json)
    detector = BlockDetector(cfg)
    result = detector.run(words, output_path)
    return result.to_dict() if result else {}


def run_reasoner(image_path: Path, cfg: VLMConfig) -> dict:
    reasoner = VLMReasoner(cfg)
    res = reasoner.run(image_path)
    return {
        "raw_response": res.raw_response,
        "parsed": res.parsed,
        "error": res.error,
        "elapsed_seconds": res.elapsed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM utilities (blocks + reasoner).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_blocks = sub.add_parser("blocks", help="Run block detector on OCR JSON.")
    p_blocks.add_argument("--ocr-json", type=Path, required=True, help="Path to OCR JSON with words.")
    p_blocks.add_argument("--output", type=Path, required=True, help="Output path for blocks JSON.")
    p_blocks.add_argument("--backend", type=str, default="heuristic", help="VLM backend for blocks (heuristic).")

    p_reasoner = sub.add_parser("reasoner", help="Run Qwen2-VL reasoner on image.")
    p_reasoner.add_argument("--image", type=Path, required=True, help="Path to page image.")
    p_reasoner.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model name.")
    p_reasoner.add_argument("--device", type=str, default="auto", help="Device: auto|cuda|mps|cpu.")
    p_reasoner.add_argument("--max-tokens", type=int, default=256, help="Max new tokens.")
    p_reasoner.add_argument("--temperature", type=float, default=0.1, help="Generation temperature.")
    p_reasoner.add_argument("--prompt", type=str, default=None, help="Custom prompt.")

    args = parser.parse_args()

    if args.command == "blocks":
        cfg = VLMConfig(enabled=True, backend=args.backend)
        result = run_blocks(args.ocr_json, args.output, cfg)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote blocks to {args.output}")
    else:
        cfg = VLMConfig(
            enabled=True,
            backend="qwen2_vl",
            model_name=args.model_name,
            device=args.device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            prompt=args.prompt or VLMConfig.prompt,
        )
        result = run_reasoner(args.image, cfg)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
