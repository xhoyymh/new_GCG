from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from model_api_adapter import OPENROUTER_MODEL_ALIASES, generate_comment_with_backend
from original_comment_loader import build_original_comment_compare_records, get_original_comment_file


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMENT_CODE_ROOT = Path(__file__).resolve().parent
RESULT_DIR = REPO_ROOT / "comment_generation" / "json" / "result"

MODEL_REGISTRY = {
    "qwen3.5_9b": {
        "backend": "ollama",
        "local_model": "qwen3.5:9b",
    },
    "qwen_free": {
        "backend": "api",
        "model_alias": "qwen",
    },
    "qwen3.5_27b": {
        "backend": "api",
        "model_alias": "qwen/qwen3.5-27b",
    },
    "glm_air": {
        "backend": "api",
        "model_alias": "glm",
    },
}

PLATFORM_CONFIG = {
    "douyin": {
        "module_path": COMMENT_CODE_ROOT / "douyin" / "comment_generate_api_ready.py",
        "module_name": "douyin_original_comments_compare_module",
        "enable_meme_search": False,
        "output_file": RESULT_DIR / "douyin_original_comments_model_compare.json",
        "meta_file": RESULT_DIR / "douyin_original_comments_model_compare_meta.json",
    },
    "youtube": {
        "module_path": COMMENT_CODE_ROOT / "youtube" / "youtube_comment_generate_api_ready.py",
        "module_name": "youtube_original_comments_compare_module",
        "enable_meme_search": False,
        "output_file": RESULT_DIR / "youtube_original_comments_model_compare.json",
        "meta_file": RESULT_DIR / "youtube_original_comments_model_compare_meta.json",
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


def _merge_base_and_existing(base_records: list[dict], existing_records: list[dict] | None) -> list[dict]:
    existing_index = {
        _record_id(item): item for item in (existing_records or [])
        if isinstance(item, dict) and _record_id(item)
    }
    merged_records: list[dict] = []

    for base in base_records:
        record = dict(base)
        existing = existing_index.get(_record_id(record))
        if existing:
            for key, value in existing.items():
                if (
                    key.endswith("_generated_comment")
                    or key.endswith("_generated_c_label")
                    or key.endswith("_error")
                ):
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


def _resolve_model_name(model_alias: str | None, local_model: str | None) -> str:
    if local_model:
        return local_model
    if not model_alias:
        return ""
    if "/" in model_alias:
        return model_alias
    return OPENROUTER_MODEL_ALIASES.get(model_alias, model_alias)


def _build_meta(platform: str, record_count: int, selected_models: list[str]) -> dict:
    cfg = PLATFORM_CONFIG[platform]
    return {
        "platform": platform,
        "input_file": str(get_original_comment_file(platform)),
        "output_file": str(cfg["output_file"]),
        "no_leakage": True,
        "include_original_comments_in_prompt": False,
        "record_count": record_count,
        "selected_models": selected_models,
        "models": {
            alias: {
                "backend": MODEL_REGISTRY[alias]["backend"],
                "model_alias": MODEL_REGISTRY[alias].get("model_alias"),
                "resolved_model": _resolve_model_name(
                    MODEL_REGISTRY[alias].get("model_alias"),
                    MODEL_REGISTRY[alias].get("local_model"),
                ),
            }
            for alias in selected_models
        },
    }


def _set_local_ollama_model(module, model_name: str) -> None:
    module.COMMENT_CONFIG["ollama_model"] = model_name
    module._base.COMMENT_CONFIG["ollama_model"] = model_name


def _prepare_jobs(module, platform: str, records: list[dict]) -> dict[str, dict]:
    cfg = PLATFORM_CONFIG[platform]
    samples = module._base.get_preprocessed_samples(
        learning_sample_file=module.COMMENT_CONFIG["learning_sample_file"],
        cache_file=module.COMMENT_CONFIG["cache_file"],
    )
    label_index = module._base.build_label_index(samples)

    kwargs = {
        "include_original_comments": False,
        "backend": "ollama",
        "model_alias": None,
    }
    if platform == "youtube":
        kwargs["enable_meme_search"] = bool(cfg["enable_meme_search"])

    jobs = []
    try:
        print(f"[{platform}] Stage 1/2: preparing compare jobs for {len(records)} records")
        jobs = module.prepare_comment_jobs(records, samples, label_index, **kwargs)
    finally:
        module._base.stop_ollama_model(
            module.COMMENT_CONFIG.get("ollama_embed_model", ""),
            "compare embedding stage",
            wait_timeout=20.0,
        )

    return {_record_id(job["video"]): job for job in jobs if _record_id(job["video"])}


def run_platform(platform: str, *, selected_models: list[str], video_id: str | None,
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

    meta = _build_meta(platform, len(working_records), selected_models)
    _save_json(cfg["meta_file"], meta)

    pending_id_set = {
        _record_id(record)
        for record in selected_records
        if any(not str(record.get(f"{alias}_generated_comment", "")).strip() for alias in selected_models)
    }

    print(f"\n=== {platform.upper()} ORIGINAL COMMENTS COMPARE ===")
    print(f"Input records: {len(working_records)}")
    print(f"Selected for this run: {len(selected_records)}")
    print(f"Original comment file: {get_original_comment_file(platform)}")
    print(f"Sample file: {module.COMMENT_CONFIG['learning_sample_file']}")
    print(f"Output file: {cfg['output_file']}")
    print(f"Meta file: {cfg['meta_file']}")
    print(f"Models: {', '.join(selected_models)}")
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
    for alias in selected_models:
        spec = MODEL_REGISTRY[alias]
        backend = spec["backend"]
        model_alias = spec.get("model_alias")
        local_model = spec.get("local_model")
        resolved_model = _resolve_model_name(model_alias, local_model)
        field_comment = f"{alias}_generated_comment"
        field_label = f"{alias}_generated_c_label"
        field_error = f"{alias}_error"

        record_ids = [
            _record_id(record)
            for record in selected_records
            if not str(record.get(field_comment, "")).strip()
            and _record_id(record) in jobs_by_id
        ]
        if not record_ids:
            print(f"[{platform}] {alias}: nothing pending")
            continue

        print(f"[{platform}] Model {alias} -> {resolved_model} ({backend}), pending={len(record_ids)}")

        if backend == "ollama" and local_model:
            _set_local_ollama_model(module, local_model)

        try:
            for record_id in record_ids:
                output_record = working_index[record_id]
                job = jobs_by_id[record_id]
                output_record[field_label] = job["generated_c_label"]
                try:
                    comment = generate_comment_with_backend(
                        job["prompt"],
                        backend=backend,
                        platform=platform,
                        model_alias=model_alias,
                        video_record=output_record,
                    )
                    output_record[field_comment] = str(comment or "").strip()
                    output_record[field_error] = ""
                except Exception as exc:
                    output_record[field_error] = str(exc)
                finally:
                    completed_pairs += 1
                    _save_json(cfg["output_file"], working_records)
        finally:
            if backend == "ollama" and local_model:
                module._base.stop_ollama_model(
                    local_model,
                    f"{alias} compare generation stage",
                    wait_timeout=20.0,
                )

    _save_json(cfg["output_file"], working_records)
    return {
        "platform": platform,
        "records": len(working_records),
        "processed_records": len(selected_records),
        "completed_model_pairs": completed_pairs,
        "output_file": str(cfg["output_file"]),
        "meta_file": str(cfg["meta_file"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the no-leakage multi-model compare pipeline on original_comments sample sets."
    )
    parser.add_argument("--platform", choices=["douyin", "youtube", "all"], required=True)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Optional subset of compare models. Defaults to all registered models.",
    )
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
        help="Resume from existing compare outputs and only fill missing model fields.",
    )
    args = parser.parse_args()

    selected_models = args.models or list(MODEL_REGISTRY.keys())
    platforms = ["douyin", "youtube"] if args.platform == "all" else [args.platform]
    summaries = []
    for platform in platforms:
        summaries.append(
            run_platform(
                platform,
                selected_models=selected_models,
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
