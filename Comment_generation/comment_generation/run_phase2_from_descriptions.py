from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMENT_CODE_ROOT = REPO_ROOT / "data_pre" / "code_active" / "comment_generation"

PLATFORM_CONFIG = {
    "douyin": {
        "module_path": COMMENT_CODE_ROOT / "douyin" / "comment_generate.py",
        "module_name": "douyin_phase2_module",
        "input_file": REPO_ROOT / "data_pre" / "json" / "douyin" / "data_pre" / "douyin_video_description_codex.json",
        "enable_meme_search": False,
    },
    "youtube": {
        "module_path": COMMENT_CODE_ROOT / "youtube" / "youtube_comment_generate_hotmeme.py",
        "module_name": "youtube_phase2_module",
        "input_file": REPO_ROOT / "data_pre" / "json" / "youtube" / "data_pre" / "youtube_video_description_codex.json",
        "enable_meme_search": True,
    },
}


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _coerce_records(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, dict)]
    raise ValueError("Expected JSON object or array")


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


def run_platform(platform: str, video_id: str | None, limit: int | None) -> dict[str, str | int]:
    cfg = PLATFORM_CONFIG[platform]
    input_file = cfg["input_file"]
    if not input_file.exists():
        raise SystemExit(f"Missing input file: {input_file}")

    module = _load_module(cfg["module_name"], cfg["module_path"])
    module.COMMENT_CONFIG["input_video_file"] = str(input_file)

    records = _coerce_records(_load_json(input_file))
    records = _select_records(records, video_id=video_id, limit=limit)
    if not records:
        raise SystemExit(f"No records selected for platform={platform!r}, video_id={video_id!r}")

    print(f"\n=== {platform.upper()} ===")
    print(f"Input records: {len(records)}")
    print(f"Input file: {module.COMMENT_CONFIG['input_video_file']}")
    print(f"Sample file: {module.COMMENT_CONFIG['learning_sample_file']}")
    print(f"Output file: {module.COMMENT_CONFIG['output_comment_file']}")

    kwargs = {
        "learning_sample_file": module.COMMENT_CONFIG["learning_sample_file"],
        "cache_file": module.COMMENT_CONFIG["cache_file"],
        "output_file": module.COMMENT_CONFIG["output_comment_file"],
    }
    if platform == "youtube":
        kwargs["enable_meme_search"] = bool(cfg["enable_meme_search"])

    results = module.run_phase2_with_records(records, **kwargs)
    return {
        "platform": platform,
        "records": len(results),
        "input_file": module.COMMENT_CONFIG["input_video_file"],
        "output_file": module.COMMENT_CONFIG["output_comment_file"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run optional Phase 2 comment generation from current Step 5 Codex descriptions."
    )
    parser.add_argument("--platform", choices=["douyin", "youtube", "all"], required=True)
    parser.add_argument("--video-id", default="", help="Optional id/video_id to run a single record.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit after filtering.")
    args = parser.parse_args()

    platforms = ["douyin", "youtube"] if args.platform == "all" else [args.platform]
    summaries = []
    for platform in platforms:
        summaries.append(
            run_platform(
                platform,
                video_id=args.video_id.strip() or None,
                limit=args.limit,
            )
        )

    print("\n=== Summary ===")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
