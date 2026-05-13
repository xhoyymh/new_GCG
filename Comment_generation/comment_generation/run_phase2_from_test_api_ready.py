from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from test_input_loader import load_and_normalize_platform_records


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMENT_CODE_ROOT = Path(__file__).resolve().parent

PLATFORM_CONFIG = {
    "douyin": {
        "module_path": COMMENT_CODE_ROOT / "douyin" / "comment_generate_api_ready.py",
        "module_name": "douyin_phase2_api_ready_module",
        "enable_meme_search": False,
    },
    "youtube": {
        "module_path": COMMENT_CODE_ROOT / "youtube" / "youtube_comment_generate_api_ready.py",
        "module_name": "youtube_phase2_api_ready_module",
        "enable_meme_search": False,
    },
}


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _select_records(records: list[dict], video_id: str | None, limit: int | None) -> list[dict]:
    selected = records
    if video_id:
        selected = [
            item for item in selected
            if str(item.get("id", "")).strip() == video_id or str(item.get("video_id", "")).strip() == video_id
        ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_platform(platform: str, video_id: str | None, limit: int | None, save_normalized: bool,
                 backend: str, model_alias: str | None) -> dict[str, str | int]:
    cfg = PLATFORM_CONFIG[platform]
    module = _load_module(cfg["module_name"], cfg["module_path"])
    records = load_and_normalize_platform_records(platform, save_normalized=save_normalized)
    records = _select_records(records, video_id=video_id, limit=limit)
    if not records:
        raise SystemExit(f"No records selected for platform={platform!r}, video_id={video_id!r}")

    print(f"\n=== {platform.upper()} (API READY) ===")
    print(f"Input records: {len(records)}")
    print(f"Original comment file: {module.COMMENT_CONFIG['original_comment_file']}")
    print(f"Sample file: {module.COMMENT_CONFIG['learning_sample_file']}")
    print(f"Output file: {module.COMMENT_CONFIG['output_comment_file']}")
    print(f"Generation backend: {backend}")
    if model_alias:
        print(f"Model alias: {model_alias}")

    kwargs = {
        "learning_sample_file": module.COMMENT_CONFIG["learning_sample_file"],
        "cache_file": module.COMMENT_CONFIG["cache_file"],
        "output_file": module.COMMENT_CONFIG["output_comment_file"],
        "backend": backend,
        "model_alias": model_alias,
    }
    if platform == "youtube":
        kwargs["enable_meme_search"] = bool(cfg["enable_meme_search"])

    results = module.run_phase2_with_records(records, **kwargs)
    return {
        "platform": platform,
        "records": len(results),
        "output_file": module.COMMENT_CONFIG["output_comment_file"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the api-ready Phase 2 pipeline with platform original comments as prompt input."
    )
    parser.add_argument("--platform", choices=["douyin", "youtube", "all"], required=True)
    parser.add_argument("--video-id", default="", help="Optional id/video_id to run a single record.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit after filtering.")
    parser.add_argument(
        "--skip-normalized-save",
        action="store_true",
        help="Do not persist normalized test inputs under comment_generation/json/<platform>/.",
    )
    parser.add_argument(
        "--generation-backend",
        choices=["ollama", "api"],
        default="ollama",
        help="Choose local Ollama generation or the future API adapter.",
    )
    parser.add_argument(
        "--model-alias",
        default="",
        help="Optional future API model alias. Ignored by the local Ollama backend.",
    )
    args = parser.parse_args()

    if args.generation_backend == "api" and not args.model_alias.strip():
        parser.error(
            "--generation-backend api requires --model-alias. "
            "Use one of: glm, dsr1, kimi, llama, gptoss, qwen, or pass a full OpenRouter model slug."
        )

    platforms = ["douyin", "youtube"] if args.platform == "all" else [args.platform]
    summaries = []
    for platform in platforms:
        summaries.append(
            run_platform(
                platform,
                video_id=args.video_id.strip() or None,
                limit=args.limit,
                save_normalized=not args.skip_normalized_save,
                backend=args.generation_backend,
                model_alias=args.model_alias.strip() or None,
            )
        )

    print("\n=== Summary ===")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
