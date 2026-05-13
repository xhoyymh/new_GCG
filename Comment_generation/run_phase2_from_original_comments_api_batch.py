from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from model_api_adapter import generate_comment_with_backend
from original_comment_loader import build_original_comment_compare_records, get_original_comment_file


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "comment_generation" / "json" / "result"

MODEL_REGISTRY = {
    "r1": {
        "backend": "api",
        "model_slug": "deepseek/deepseek-r1",
    },
    "llama": {
        "backend": "api",
        "model_slug": "meta-llama/llama-4-maverick",
    },
    "gpt54": {
        "backend": "api",
        "model_slug": "openai/gpt-5.4",
    },
}

PLATFORM_CONFIG = {
    "douyin": {
        "module_path": REPO_ROOT / "comment_generation" / "code" / "douyin" / "comment_generate_api_ready.py",
        "module_name": "douyin_original_comments_api_batch_module",
        "enable_meme_search": False,
        "output_file": RESULT_DIR / "douyin_original_comments_paid_model_outputs.json",
    },
    "youtube": {
        "module_path": REPO_ROOT / "comment_generation" / "code" / "youtube" / "youtube_comment_generate_api_ready.py",
        "module_name": "youtube_original_comments_api_batch_module",
        "enable_meme_search": False,
        "output_file": RESULT_DIR / "youtube_original_comments_paid_model_outputs.json",
    },
}


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_id(record: dict) -> str:
    return str(record.get("id") or record.get("video_id") or "").strip()


def _result_fields() -> set[str]:
    fields: set[str] = set()
    for alias in MODEL_REGISTRY:
        fields.add(f"{alias}_generated_comment")
        fields.add(f"{alias}_generated_c_label")
        fields.add(f"{alias}_error")
    return fields


def _merge_base_and_existing(base_records: list[dict], existing_records: list[dict] | None) -> list[dict]:
    existing_index = {
        _record_id(item): item for item in (existing_records or [])
        if isinstance(item, dict) and _record_id(item)
    }
    result_fields = _result_fields()
    merged_records: list[dict] = []

    for base in base_records:
        record = dict(base)
        existing = existing_index.get(_record_id(record))
        if existing:
            for key, value in existing.items():
                if key in result_fields:
                    record[key] = value
        merged_records.append(record)

    return merged_records


def _select_records(records: list[dict], video_id: str | None, limit: int | None) -> list[dict]:
    selected = records
    if video_id:
        selected = [item for item in selected if _record_id(item) == video_id]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _prepare_jobs(module, platform: str, records: list[dict]) -> dict[str, dict]:
    cfg = PLATFORM_CONFIG[platform]
    samples = module._base.get_preprocessed_samples(
        learning_sample_file=module.COMMENT_CONFIG["learning_sample_file"],
        cache_file=module.COMMENT_CONFIG["cache_file"],
    )
    label_index = module._base.build_label_index(samples)

    kwargs = {
        "include_original_comments": False,
        "backend": "api",
        "model_alias": None,
    }
    if platform == "youtube":
        kwargs["enable_meme_search"] = bool(cfg["enable_meme_search"])

    jobs = []
    try:
        print(f"[{platform}] Stage 1/2: preparing API generation jobs for {len(records)} records")
        jobs = module.prepare_comment_jobs(records, samples, label_index, **kwargs)
    finally:
        module._base.stop_ollama_model(
            module.COMMENT_CONFIG.get("ollama_embed_model", ""),
            "api batch embedding stage",
            wait_timeout=20.0,
        )

    return {_record_id(job["video"]): job for job in jobs if _record_id(job["video"])}


def run_platform(platform: str, *, video_id: str | None,
                 limit: int | None, save_normalized: bool, resume: bool) -> dict[str, str | int]:
    cfg = PLATFORM_CONFIG[platform]
    module = _load_module(cfg["module_name"], cfg["module_path"])
    base_records = build_original_comment_compare_records(platform, save_normalized=save_normalized)
    existing_records = _load_json_if_exists(cfg["output_file"]) if resume else None
    working_records = _merge_base_and_existing(base_records, existing_records)
    working_index = {_record_id(item): item for item in working_records if _record_id(item)}

    selected_records = _select_records(working_records, video_id=video_id, limit=limit)
    if not selected_records:
        raise SystemExit(f"No records selected for platform={platform!r}, video_id={video_id!r}")

    pending_id_set = {
        _record_id(record)
        for record in selected_records
        if any(not str(record.get(f"{alias}_generated_comment", "")).strip() for alias in MODEL_REGISTRY)
    }

    print(f"\n=== {platform.upper()} ORIGINAL COMMENTS API BATCH ===")
    print(f"Input records: {len(working_records)}")
    print(f"Selected for this run: {len(selected_records)}")
    print(f"Original comment file: {get_original_comment_file(platform)}")
    print(f"Sample file: {module.COMMENT_CONFIG['learning_sample_file']}")
    print(f"Output file: {cfg['output_file']}")
    print("Models: r1, llama, gpt54")
    print("Leakage control: same-video original comments are NOT injected into prompts")

    if not pending_id_set:
        print(f"[{platform}] Nothing to do; all requested model outputs already exist.")
        _save_json(cfg["output_file"], working_records)
        return {
            "platform": platform,
            "records": len(working_records),
            "processed_records": 0,
            "output_file": str(cfg["output_file"]),
        }

    jobs_by_id = _prepare_jobs(
        module,
        platform,
        [dict(working_index[record_id]) for record_id in pending_id_set if record_id in working_index],
    )

    completed_pairs = 0
    for alias, spec in MODEL_REGISTRY.items():
        field_comment = f"{alias}_generated_comment"
        field_label = f"{alias}_generated_c_label"
        field_error = f"{alias}_error"
        model_slug = spec["model_slug"]

        record_ids = [
            _record_id(record)
            for record in selected_records
            if not str(record.get(field_comment, "")).strip()
            and _record_id(record) in jobs_by_id
        ]
        if not record_ids:
            print(f"[{platform}] {alias}: nothing pending")
            continue

        print(f"[{platform}] Model {alias} -> {model_slug}, pending={len(record_ids)}")

        for record_id in record_ids:
            output_record = working_index[record_id]
            job = jobs_by_id[record_id]
            output_record[field_label] = job["generated_c_label"]
            try:
                comment = generate_comment_with_backend(
                    job["prompt"],
                    backend="api",
                    platform=platform,
                    model_alias=model_slug,
                    video_record=output_record,
                )
                output_record[field_comment] = str(comment or "").strip()
                output_record[field_error] = ""
            except Exception as exc:
                output_record[field_error] = str(exc)
            finally:
                completed_pairs += 1
                _save_json(cfg["output_file"], working_records)

    _save_json(cfg["output_file"], working_records)
    return {
        "platform": platform,
        "records": len(working_records),
        "processed_records": len(selected_records),
        "completed_model_pairs": completed_pairs,
        "output_file": str(cfg["output_file"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the no-leakage original_comments batch generation pipeline with paid OpenRouter API models."
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
        "--resume",
        action="store_true",
        help="Resume from existing outputs and only fill missing model fields.",
    )
    args = parser.parse_args()

    platforms = ["douyin", "youtube"] if args.platform == "all" else [args.platform]
    summaries = []
    for platform in platforms:
        summaries.append(
            run_platform(
                platform,
                video_id=args.video_id.strip() or None,
                limit=args.limit,
                save_normalized=not args.skip_normalized_save,
                resume=args.resume,
            )
        )

    print("\n=== Summary ===")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
