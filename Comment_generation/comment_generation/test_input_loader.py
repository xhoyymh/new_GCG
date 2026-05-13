from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = REPO_ROOT / "test"
COMMENT_JSON_DIR = REPO_ROOT / "comment_generation" / "json"

COMMON_CHINESE_CHARS = set(
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动"
    "同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高"
    "自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政"
    "四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道"
    "命此变条只没结解问意建月公无军很情者最立代想已通并提直题党程展五果料象员革命位"
    "入常文总次品式活设及管特件长求老儿尔位她们剧笑搞短评说段子脱口秀表演相声视频描"
    "述字幕简介动物幽默其他生活女孩男生朋友评论真实自然梗应用普通内容提取押韵谐音反"
    "话日常搞笑短剧类大学生上车一定记得先看后排"
)
SUSPICIOUS_MARKERS = ("锟", "鈥", "銆", "鍦", "鎴", "鐨", "\ufffd")

PLATFORM_FILES = {
    "douyin": {
        "description": TEST_DIR / "douyin_test_descrption.json",
        "url": TEST_DIR / "douyin_url_test.json",
        "normalized": COMMENT_JSON_DIR / "douyin" / "douyin_test_description_normalized.json",
    },
    "youtube": {
        "description": TEST_DIR / "youtube_test_description.json",
        "url": TEST_DIR / "youtube_url_test.json",
        "normalized": COMMENT_JSON_DIR / "youtube" / "youtube_test_description_normalized.json",
    },
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _repair_json_text(text: str) -> str:
    repaired = text.lstrip("\ufeff").replace("\x00", "")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def _load_json(path: Path) -> Any:
    raw = _read_text(path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_repair_json_text(raw))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _quality_score(text: str) -> float:
    if not text:
        return 0.0
    score = 0.0
    score += sum(2.0 for ch in text if ch in COMMON_CHINESE_CHARS)
    score += sum(0.25 for ch in text if ch.isascii() and (ch.isalnum() or ch in " \n\r\t.,!?;:'\"-_/\\#()[]{}"))
    score += sum(1.0 for token in (" the ", " and ", " with ", "video", "comment", "funny") if token in text.lower())
    score -= sum(3.0 for marker in SUSPICIOUS_MARKERS for _ in range(text.count(marker)))
    score -= text.count("?") * 0.5
    return score


def _maybe_fix_mojibake(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text

    candidates = [text]
    for source_encoding in ("gbk", "cp1252", "latin-1"):
        try:
            candidate = text.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if candidate and candidate != text:
            candidates.append(candidate)

    best = max(candidates, key=_quality_score)
    return best if _quality_score(best) >= _quality_score(text) + 2.0 else text


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return _maybe_fix_mojibake(text)


def _coerce_records(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, dict)]
    raise ValueError("Expected JSON object or array")


def _build_url_index(platform: str) -> dict[str, dict]:
    records = _coerce_records(_load_json(PLATFORM_FILES[platform]["url"]))
    index: dict[str, dict] = {}
    for item in records:
        raw_id = item.get("id") or item.get("video_id")
        if raw_id in (None, ""):
            continue
        key = str(raw_id).strip()
        if not key:
            continue
        index[key] = {
            "id": key,
            "video_url": _normalize_text(item.get("video_url", "")),
            "label": _normalize_text(item.get("label", "")),
            "video_introduction": _normalize_text(item.get("video_introduction", "")),
        }
    return index


def _normalize_record(record: dict, url_index: dict[str, dict]) -> dict:
    normalized = dict(record)
    record_id = normalized.get("id") or normalized.get("video_id")
    if record_id in (None, ""):
        raise ValueError("Missing id/video_id in test record")

    record_id = str(record_id).strip()
    if not record_id:
        raise ValueError("Blank id/video_id in test record")

    normalized["id"] = record_id
    if "video_id" not in normalized or str(normalized.get("video_id", "")).strip() == "":
        normalized["video_id"] = record_id

    backup = url_index.get(record_id, {})
    for key in ("video_url", "label", "video_introduction", "all_transcription", "video_description"):
        normalized[key] = _normalize_text(normalized.get(key, ""))

    if not normalized["video_url"]:
        normalized["video_url"] = backup.get("video_url", "")

    if not normalized["label"]:
        normalized["label"] = backup.get("label", "")

    if not normalized["video_introduction"]:
        normalized["video_introduction"] = backup.get("video_introduction", "")

    return normalized


def load_and_normalize_platform_records(platform: str, save_normalized: bool = True) -> list[dict]:
    if platform not in PLATFORM_FILES:
        raise ValueError(f"Unsupported platform: {platform}")

    records = _coerce_records(_load_json(PLATFORM_FILES[platform]["description"]))
    url_index = _build_url_index(platform)
    normalized = [_normalize_record(item, url_index) for item in records]

    if save_normalized:
        _save_json(PLATFORM_FILES[platform]["normalized"], normalized)
    return normalized


def get_platform_paths(platform: str) -> dict[str, Path]:
    if platform not in PLATFORM_FILES:
        raise ValueError(f"Unsupported platform: {platform}")
    return PLATFORM_FILES[platform].copy()
