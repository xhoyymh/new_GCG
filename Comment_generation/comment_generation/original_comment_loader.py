from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from test_input_loader import load_and_normalize_platform_records


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMENT_GENERATION_DIR = REPO_ROOT / "comment_generation"

ORIGINAL_COMMENT_FILES = {
    "douyin": COMMENT_GENERATION_DIR / "original_comments_for_douyin.json",
    "youtube": COMMENT_GENERATION_DIR / "original_comments_for_youtube.json",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _coerce_records(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, dict)]
    raise ValueError("Expected JSON object or array")


def _normalize_id(record: dict) -> str:
    raw_id = record.get("id") or record.get("video_id")
    return str(raw_id or "").strip()


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def get_original_comment_file(platform: str) -> Path:
    if platform not in ORIGINAL_COMMENT_FILES:
        raise ValueError(f"Unsupported platform: {platform}")
    return ORIGINAL_COMMENT_FILES[platform]


def load_original_comment_records(platform: str) -> list[dict]:
    path = get_original_comment_file(platform)
    records = _coerce_records(_load_json(path))
    normalized_records: list[dict] = []

    for item in records:
        record = dict(item)
        record_id = _normalize_id(record)
        if not record_id:
            continue
        record["id"] = record_id
        record.setdefault("video_id", record_id)
        normalized_records.append(record)

    return normalized_records


def load_original_comment_index(platform: str) -> dict[str, dict]:
    index: dict[str, dict] = {}

    for item in load_original_comment_records(platform):
        key = _normalize_id(item)
        if not key:
            continue
        index[key] = dict(item)

    return index


def build_original_comment_compare_records(platform: str, *, save_normalized: bool = True) -> list[dict]:
    original_records = load_original_comment_records(platform)
    normalized_records = load_and_normalize_platform_records(platform, save_normalized=save_normalized)
    normalized_index = {_normalize_id(item): dict(item) for item in normalized_records if _normalize_id(item)}

    merged_records: list[dict] = []
    for item in original_records:
        record = dict(item)
        record_id = _normalize_id(record)
        if not record_id:
            continue

        backup = normalized_index.get(record_id, {})
        record["id"] = record_id
        record.setdefault("video_id", record_id)

        for key in (
            "video_url",
            "label",
            "video_introduction",
            "all_transcription",
            "video_description",
        ):
            if not _normalize_text(record.get(key)):
                fallback_value = backup.get(key, "")
                if _normalize_text(fallback_value):
                    record[key] = fallback_value

        merged_records.append(record)

    return merged_records


def extract_original_comments(record: dict | None) -> list[str]:
    if not record:
        return []

    comments = []
    for i in range(1, 6):
        text = str(record.get(f"comment_{i}", "")).strip()
        if text:
            comments.append(text)
    return comments
