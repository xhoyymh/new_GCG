from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont, ImageOps

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "codex-step5-materials"
SERVER_VERSION = "0.2.0"
VALID_STATUSES = {"pending", "in_progress", "done", "failed"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STALE_IN_PROGRESS_MINUTES = int(os.environ.get("CODEX_STEP5_STALE_MINUTES", "30"))

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PRE_ROOT = REPO_ROOT / "data_pre"

MANIFEST_PATH = Path(
    os.environ.get(
        "CODEX_STEP5_MANIFEST_PATH",
        REPO_ROOT / "data_pre" / "json" / "materials" / "ten_video_step5_candidates_20260328.json",
    )
)
RUN_STATE_PATH = Path(
    os.environ.get(
        "CODEX_STEP5_RUN_STATE_PATH",
        REPO_ROOT / "data_pre" / "json" / "materials" / "codex_step5_run_20260328.json",
    )
)
PREVIEW_DIR = Path(
    os.environ.get(
        "CODEX_STEP5_PREVIEW_DIR",
        REPO_ROOT / "data_pre" / "json" / "materials" / "previews",
    )
)

DOUYIN_DESCRIPTION_PATH = Path(
    os.environ.get(
        "CODEX_STEP5_DOUYIN_DESCRIPTION_PATH",
        REPO_ROOT / "data_pre" / "json" / "douyin" / "data_pre" / "douyin_video_description.json",
    )
)
YOUTUBE_DESCRIPTION_PATH = Path(
    os.environ.get(
        "CODEX_STEP5_YOUTUBE_DESCRIPTION_PATH",
        REPO_ROOT / "data_pre" / "json" / "youtube" / "data_pre" / "youtube_video_description.json",
    )
)

DOUYIN_CHOUZHEN_PATH = Path(
    os.environ.get(
        "CODEX_STEP5_DOUYIN_CHOUZHEN_PATH",
        REPO_ROOT / "data_pre" / "json" / "douyin" / "data_pre" / "douyin_chouzhen.json",
    )
)
YOUTUBE_CHOUZHEN_PATH = Path(
    os.environ.get(
        "CODEX_STEP5_YOUTUBE_CHOUZHEN_PATH",
        REPO_ROOT / "data_pre" / "json" / "youtube" / "data_pre" / "youtube_chouzhen.json",
    )
)

DOUYIN_TOP5_PATH = REPO_ROOT / "data_pre" / "json" / "douyin" / "data_pre" / "douyin_top5_comments.json"
YOUTUBE_TOP5_PATH = REPO_ROOT / "data_pre" / "json" / "youtube" / "data_pre" / "youtube_top5_comment.json"


def _configure_runtime_paths(
    manifest_path: str | None = None,
    run_state_path: str | None = None,
    preview_dir: str | None = None,
    douyin_description_path: str | None = None,
    youtube_description_path: str | None = None,
) -> None:
    global MANIFEST_PATH, RUN_STATE_PATH, PREVIEW_DIR
    global DOUYIN_DESCRIPTION_PATH, YOUTUBE_DESCRIPTION_PATH

    if manifest_path:
        MANIFEST_PATH = Path(manifest_path)
    if run_state_path:
        RUN_STATE_PATH = Path(run_state_path)
    if preview_dir:
        PREVIEW_DIR = Path(preview_dir)
    if douyin_description_path:
        DOUYIN_DESCRIPTION_PATH = Path(douyin_description_path)
    if youtube_description_path:
        YOUTUBE_DESCRIPTION_PATH = Path(youtube_description_path)

    _load_manifest.cache_clear()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_platform(platform: Any) -> str:
    value = str(platform or "").strip().lower()
    if value not in {"douyin", "youtube"}:
        raise ValueError(f"Unsupported platform: {platform!r}")
    return value


def _normalize_video_id(video_id: Any) -> str:
    value = str(video_id or "").strip()
    if not value:
        raise ValueError("Video id cannot be empty.")
    return value


def _repo_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _repo_relative_str(value: Any) -> str:
    path = _repo_path(value)
    if not path:
        return ""
    try:
        rel = path.relative_to(REPO_ROOT)
        return str(rel).replace("/", "\\")
    except ValueError:
        return str(path)


def _path_from_manifest(record: dict, key: str, fallback: str = "") -> Path | None:
    raw = record.get(key) or fallback
    return _repo_path(raw)


def _json_default_container(default: Any) -> Any:
    if isinstance(default, dict):
        return {}
    if isinstance(default, list):
        return []
    return default


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return _json_default_container(default)
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _dump_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _sort_records_by_id(records: list[dict]) -> list[dict]:
    def _sort_key(item: dict) -> tuple[int, str]:
        raw = str(item.get("id", "")).strip()
        if raw.isdigit():
            return (0, f"{int(raw):012d}")
        return (1, raw)

    return sorted(records, key=_sort_key)


def _dedupe_keep_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _natural_path_key(path: Path) -> tuple[int, str, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return (0, f"{int(match.group(1)):012d}", path.name.lower())
    return (1, path.stem.lower(), path.name.lower())


def _read_text_if_exists(path: Path | None) -> str:
    if not path or not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _print_paths() -> dict:
    return {
        "repo_root": str(REPO_ROOT.resolve()),
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "run_state_path": str(RUN_STATE_PATH.resolve()),
        "preview_dir": str(PREVIEW_DIR.resolve()),
        "douyin_description_path": str(DOUYIN_DESCRIPTION_PATH.resolve()),
        "youtube_description_path": str(YOUTUBE_DESCRIPTION_PATH.resolve()),
        "douyin_chouzhen_path": str(DOUYIN_CHOUZHEN_PATH.resolve()),
        "youtube_chouzhen_path": str(YOUTUBE_CHOUZHEN_PATH.resolve()),
    }


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    manifest = _load_json(MANIFEST_PATH, {})
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be an object: {MANIFEST_PATH}")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Manifest records must be a list: {MANIFEST_PATH}")

    normalized: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest record #{index} must be an object.")

        platform = _normalize_platform(item.get("platform"))
        video_id = _normalize_video_id(item.get("id"))
        key = (platform, video_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate record in manifest: {platform}/{video_id}")
        seen_keys.add(key)

        image_root = _path_from_manifest(item, "image_root")
        frame_root = _path_from_manifest(
            item,
            "frame_root",
            str(Path(str(item.get("image_root") or "")) / "frames"),
        )
        transcription_path = _path_from_manifest(
            item,
            "transcription_path",
            str(image_root / "transcription.txt") if image_root else "",
        )
        video_path = _path_from_manifest(item, "video_path")

        normalized.append(
            {
                "platform": platform,
                "id": video_id,
                "label": str(item.get("label") or "").strip(),
                "video_url": str(item.get("video_url") or "").strip(),
                "video_path": _repo_relative_str(video_path),
                "video_path_abs": str(video_path) if video_path else "",
                "image_root": _repo_relative_str(image_root),
                "image_root_abs": str(image_root) if image_root else "",
                "frame_root": _repo_relative_str(frame_root),
                "frame_root_abs": str(frame_root) if frame_root else "",
                "transcription_path": _repo_relative_str(transcription_path),
                "transcription_path_abs": str(transcription_path) if transcription_path else "",
                "frame_count": int(item.get("frame_count") or 0),
                "transcription_chars": int(item.get("transcription_chars") or 0),
                "top5_comments": [
                    str(value).strip()
                    for value in item.get("top5_comments", [])
                    if str(value).strip()
                ],
                "video_introduction": str(item.get("video_introduction") or "").strip(),
                "source_result_file": str(item.get("source_result_file") or "").strip(),
                "order": index,
            }
        )

    output = dict(manifest)
    output["records"] = normalized
    output["record_count"] = len(normalized)
    return output


@lru_cache(maxsize=1)
def _manifest_map() -> dict[tuple[str, str], dict]:
    return {
        (record["platform"], record["id"]): record
        for record in _load_manifest()["records"]
    }


def _empty_state_from_manifest() -> dict:
    manifest = _load_manifest()
    now = _now_iso()
    return {
        "schema_version": 1,
        "generated_at": now,
        "updated_at": now,
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "record_count": manifest["record_count"],
        "records": [
            {
                "platform": record["platform"],
                "id": record["id"],
                "status": "pending",
                "attempts": 0,
                "started_at": "",
                "last_error": "",
                "last_duration_seconds": None,
                "updated_at": now,
            }
            for record in manifest["records"]
        ],
    }


def _load_run_state() -> dict:
    state = _load_json(RUN_STATE_PATH, {})
    if not state:
        return _empty_state_from_manifest()
    if not isinstance(state, dict):
        raise ValueError(f"Run state must be an object: {RUN_STATE_PATH}")
    if not isinstance(state.get("records"), list):
        raise ValueError(f"Run state records must be a list: {RUN_STATE_PATH}")
    return state


def _sync_run_state(state: dict) -> dict:
    manifest = _load_manifest()
    current_records = state.get("records", [])
    current_map: dict[tuple[str, str], dict] = {}
    for item in current_records:
        if not isinstance(item, dict):
            continue
        try:
            platform = _normalize_platform(item.get("platform"))
            video_id = _normalize_video_id(item.get("id"))
        except ValueError:
            continue
        status = str(item.get("status") or "").strip()
        if status not in VALID_STATUSES:
            status = "pending"
        current_map[(platform, video_id)] = {
            "platform": platform,
            "id": video_id,
            "status": status,
            "attempts": int(item.get("attempts") or 0),
            "started_at": str(item.get("started_at") or ""),
            "last_error": str(item.get("last_error") or "").strip(),
            "last_duration_seconds": item.get("last_duration_seconds"),
            "updated_at": str(item.get("updated_at") or state.get("updated_at") or _now_iso()),
        }

    now = _now_iso()
    existing_output_keys = _existing_description_keys()
    synced_records: list[dict] = []
    for record in manifest["records"]:
        key = (record["platform"], record["id"])
        entry = current_map.get(
            key,
            {
                "platform": record["platform"],
                "id": record["id"],
                "status": "pending",
                "attempts": 0,
                "started_at": "",
                "last_error": "",
                "last_duration_seconds": None,
                "updated_at": now,
            },
        )
        if key in existing_output_keys:
            if (
                entry["status"] != "done"
                or str(entry.get("last_error") or "").strip()
            ):
                entry["status"] = "done"
                entry["last_error"] = ""
                entry["updated_at"] = now
        elif entry["status"] == "done":
            entry["status"] = "pending"
            entry["started_at"] = ""
            entry["last_error"] = (
                "Reset to pending during run-state sync because no persisted video_description was found."
            )
            entry["last_duration_seconds"] = None
            entry["updated_at"] = now
        synced_records.append(entry)

    return {
        "schema_version": 1,
        "generated_at": str(state.get("generated_at") or now),
        "updated_at": now,
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "record_count": len(synced_records),
        "records": synced_records,
    }


def _ensure_run_state() -> dict:
    state = _sync_run_state(_load_run_state())
    _dump_json_atomic(RUN_STATE_PATH, state)
    return state


def _parse_dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _seconds_since(started_at: str) -> float | None:
    if not started_at:
        return None
    started_dt = _parse_dt(started_at)
    if not started_dt:
        return None
    if started_dt.tzinfo is None:
        started_dt = started_dt.astimezone()
    delta = datetime.now().astimezone() - started_dt
    return round(max(delta.total_seconds(), 0.0), 3)


def _reclaim_stale_in_progress(state: dict) -> tuple[dict, list[str]]:
    threshold = timedelta(minutes=STALE_IN_PROGRESS_MINUTES)
    now = datetime.now().astimezone()
    reclaimed: list[str] = []

    for record in state.get("records", []):
        if record.get("status") != "in_progress":
            continue
        updated_at = _parse_dt(str(record.get("updated_at") or ""))
        if not updated_at:
            continue
        if updated_at.tzinfo is None:
            updated_at = updated_at.astimezone()
        if now - updated_at < threshold:
            continue
        record["status"] = "pending"
        record["started_at"] = ""
        record["last_error"] = (
            f"Auto-reclaimed stale in_progress record after {STALE_IN_PROGRESS_MINUTES} minutes."
        )
        record["last_duration_seconds"] = None
        record["updated_at"] = _now_iso()
        reclaimed.append(f'{record["platform"]}/{record["id"]}')

    if reclaimed:
        state["updated_at"] = _now_iso()
    return state, reclaimed


def _save_run_state(state: dict) -> dict:
    state["updated_at"] = _now_iso()
    _dump_json_atomic(RUN_STATE_PATH, state)
    return state


def _state_record_map(state: dict) -> dict[tuple[str, str], dict]:
    return {
        (_normalize_platform(item.get("platform")), _normalize_video_id(item.get("id"))): item
        for item in state.get("records", [])
    }


@lru_cache(maxsize=2)
def _load_chouzhen_map(platform: str) -> dict[str, dict]:
    platform = _normalize_platform(platform)
    path = DOUYIN_CHOUZHEN_PATH if platform == "douyin" else YOUTUBE_CHOUZHEN_PATH
    data = _load_json(path, [])
    if not isinstance(data, list):
        raise ValueError(f"Chouzhen JSON must be a list: {path}")
    result: dict[str, dict] = {}
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        result[str(item["id"]).strip()] = item
    return result


@lru_cache(maxsize=2)
def _load_top5_map(platform: str) -> dict[str, dict]:
    platform = _normalize_platform(platform)
    path = DOUYIN_TOP5_PATH if platform == "douyin" else YOUTUBE_TOP5_PATH
    data = _load_json(path, [])
    if not isinstance(data, list):
        raise ValueError(f"Top5 JSON must be a list: {path}")
    result: dict[str, dict] = {}
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        result[str(item["id"]).strip()] = item
    return result


def _description_json_path(platform: str) -> Path:
    platform = _normalize_platform(platform)
    return DOUYIN_DESCRIPTION_PATH if platform == "douyin" else YOUTUBE_DESCRIPTION_PATH


def _load_description_records(platform: str) -> list[dict]:
    data = _load_json(_description_json_path(platform), [])
    if not isinstance(data, list):
        raise ValueError(f"Description JSON must be a list: {_description_json_path(platform)}")
    return data


def _existing_description_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for platform in ("douyin", "youtube"):
        for item in _load_description_records(platform):
            if not isinstance(item, dict) or "id" not in item:
                continue
            description = _normalize_description_text(item.get("video_description", ""))
            if not description:
                continue
            try:
                video_id = _normalize_video_id(item.get("id"))
            except ValueError:
                continue
            keys.add((platform, video_id))
    return keys


def _target_sample_size(frame_count: int) -> int:
    if frame_count <= 12:
        return frame_count
    if frame_count <= 60:
        return 12
    if frame_count <= 160:
        return 16
    return 24


def _sample_indices(total: int, target: int) -> list[int]:
    if total <= 0:
        return []
    if target >= total:
        return list(range(total))
    if target <= 1:
        return [total // 2]
    if target == 2:
        return [0, total - 1]

    indices = [0]
    interior = total - 2
    middle_slots = target - 2
    for slot in range(middle_slots):
        bucket_start = 1 + math.floor(slot * interior / middle_slots)
        bucket_end = 1 + math.floor((slot + 1) * interior / middle_slots)
        if bucket_end <= bucket_start:
            idx = bucket_start
        else:
            idx = bucket_start + (bucket_end - bucket_start - 1) // 2
        indices.append(max(1, min(total - 2, idx)))
    indices.append(total - 1)
    return _dedupe_keep_order(indices)


def _candidate_frame_roots(record: dict, chouzhen: dict) -> list[Path]:
    roots: list[Path] = []
    manifest_frame_root = _repo_path(record.get("frame_root"))
    manifest_image_root = _repo_path(record.get("image_root"))
    chouzhen_image_root = _repo_path(chouzhen.get("image_root"))
    if manifest_frame_root:
        roots.append(manifest_frame_root)
    if manifest_image_root:
        roots.append(manifest_image_root / "frames")
        roots.append(manifest_image_root)
    if chouzhen_image_root:
        roots.append(chouzhen_image_root / "frames")
        roots.append(chouzhen_image_root)
    return [path for path in roots if str(path)]


def _resolve_frame_paths(record: dict, chouzhen: dict) -> list[Path]:
    frame_paths: list[Path] = []

    raw_images = chouzhen.get("image")
    if isinstance(raw_images, list):
        for item in raw_images:
            path = _repo_path(item)
            if path and path.is_file():
                frame_paths.append(path)

    if frame_paths:
        return sorted(_dedupe_keep_order(frame_paths), key=_natural_path_key)

    for root in _candidate_frame_roots(record, chouzhen):
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
            frame_paths.append(root)
            continue
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                frame_paths.append(child)
        if frame_paths:
            break

    return sorted(_dedupe_keep_order(frame_paths), key=_natural_path_key)


def _resolve_transcript(record: dict, chouzhen: dict) -> tuple[str, str]:
    text = str(chouzhen.get("all_transcription") or "").strip()
    if text:
        return text, "chouzhen_json"

    candidates = [
        _repo_path(record.get("transcription_path")),
        _repo_path(record.get("image_root")) / "transcription.txt" if record.get("image_root") else None,
        _repo_path(chouzhen.get("image_root")) / "transcription.txt" if chouzhen.get("image_root") else None,
    ]
    for path in candidates:
        content = _read_text_if_exists(path)
        if content:
            return content, str(path)
    return "", ""


def _resolve_top_comments(record: dict, top5_meta: dict) -> list[str]:
    comments = [
        str(value).strip()
        for value in record.get("top5_comments", [])
        if str(value).strip()
    ]
    if comments:
        return comments[:5]

    derived: list[str] = []
    for index in range(1, 6):
        value = str(top5_meta.get(f"comment_{index}") or "").strip()
        if value:
            derived.append(value)
    return derived


def _preferred_output_id(platform: str, video_id: str, existing: dict, top5_meta: dict, chouzhen: dict) -> Any:
    for source in (existing, top5_meta, chouzhen):
        if isinstance(source, dict) and "id" in source:
            return source["id"]
    if platform == "youtube" and video_id.isdigit():
        return int(video_id)
    return video_id


def _normalize_description_text(description: Any) -> str:
    text = str(description or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


def _build_contact_sheet(sampled_frames: list[Path], platform: str, video_id: str, label: str) -> Path:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = PREVIEW_DIR / f"{platform}_{video_id}_contact_sheet.jpg"

    if not sampled_frames:
        raise ValueError(f"No sampled frames available for {platform}/{video_id}")

    frame_count = len(sampled_frames)
    if frame_count <= 12:
        cols = min(4, frame_count)
    elif frame_count <= 16:
        cols = 4
    else:
        cols = 5
    rows = math.ceil(frame_count / cols)

    cell_width = 300
    cell_height = 220
    caption_height = 28
    padding = 14
    header_height = 84
    width = padding + cols * (cell_width + padding)
    height = header_height + padding + rows * (cell_height + caption_height + padding)

    canvas = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    title = f"{platform.upper()} #{video_id} | {label or 'Unknown Label'}"
    subtitle = f"Sampled storyboard frames: {frame_count}"
    draw.text((padding, 14), title, fill="#f8fafc", font=title_font)
    draw.text((padding, 40), subtitle, fill="#cbd5e1", font=body_font)

    for index, frame_path in enumerate(sampled_frames):
        row = index // cols
        col = index % cols
        x = padding + col * (cell_width + padding)
        y = header_height + padding + row * (cell_height + caption_height + padding)

        tile = Image.new("RGB", (cell_width, cell_height), "#1e293b")
        try:
            with Image.open(frame_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((cell_width, cell_height))
                offset_x = (cell_width - image.width) // 2
                offset_y = (cell_height - image.height) // 2
                tile.paste(image, (offset_x, offset_y))
        except Exception:
            placeholder = Image.new("RGB", (cell_width, cell_height), "#334155")
            ImageDraw.Draw(placeholder).text((12, 12), "Unreadable frame", fill="#f8fafc", font=body_font)
            tile = placeholder

        canvas.paste(tile, (x, y))
        draw.text(
            (x, y + cell_height + 6),
            f"{index + 1:02d}. {frame_path.name}",
            fill="#e2e8f0",
            font=body_font,
        )

    canvas.save(preview_path, format="JPEG", quality=86)
    return preview_path.resolve()


def _base_video_meta(record: dict, chouzhen: dict, top5_meta: dict) -> dict:
    video_path_abs = _repo_path(record.get("video_path"))
    image_root_abs = _repo_path(record.get("image_root"))
    frame_root_abs = _repo_path(record.get("frame_root"))
    transcription_abs = _repo_path(record.get("transcription_path"))

    return {
        "platform": record["platform"],
        "id": record["id"],
        "label": record.get("label", ""),
        "video_url": record.get("video_url") or chouzhen.get("video_url") or top5_meta.get("video_url") or "",
        "video_path": record.get("video_path") or _repo_relative_str(chouzhen.get("video_path")),
        "video_path_abs": str(video_path_abs) if video_path_abs else "",
        "image_root": record.get("image_root") or _repo_relative_str(chouzhen.get("image_root")),
        "image_root_abs": str(image_root_abs) if image_root_abs else "",
        "frame_root": record.get("frame_root"),
        "frame_root_abs": str(frame_root_abs) if frame_root_abs else "",
        "transcription_path": record.get("transcription_path"),
        "transcription_path_abs": str(transcription_abs) if transcription_abs else "",
        "source_result_file": record.get("source_result_file", ""),
    }


def _resolve_record(platform: Any, video_id: Any) -> tuple[dict, dict, dict]:
    normalized_platform = _normalize_platform(platform)
    normalized_id = _normalize_video_id(video_id)
    key = (normalized_platform, normalized_id)
    manifest_record = _manifest_map().get(key)
    if not manifest_record:
        raise KeyError(f"Video not present in manifest: {normalized_platform}/{normalized_id}")

    chouzhen = _load_chouzhen_map(normalized_platform).get(normalized_id, {})
    top5_meta = _load_top5_map(normalized_platform).get(normalized_id, {})
    return manifest_record, chouzhen, top5_meta


def _mark_in_progress_if_needed(platform: str, video_id: str) -> dict:
    state = _ensure_run_state()
    state, _ = _reclaim_stale_in_progress(state)
    state_map = _state_record_map(state)
    record = state_map.get((platform, video_id))
    if not record:
        raise KeyError(f"Video not found in state: {platform}/{video_id}")
    if record["status"] in {"pending", "failed"}:
        now = _now_iso()
        record["status"] = "in_progress"
        record["attempts"] = int(record.get("attempts") or 0) + 1
        record["started_at"] = now
        record["last_duration_seconds"] = None
        record["updated_at"] = now
        _save_run_state(state)
    elif record["status"] == "in_progress":
        _save_run_state(state)
    return record


def step5_run_status() -> dict:
    state = _ensure_run_state()
    state, reclaimed = _reclaim_stale_in_progress(state)
    if reclaimed:
        _save_run_state(state)

    counts = {name: 0 for name in VALID_STATUSES}
    records_by_status: dict[str, list[dict]] = {name: [] for name in VALID_STATUSES}
    for item in state.get("records", []):
        status = item.get("status", "pending")
        if status not in VALID_STATUSES:
            status = "pending"
        counts[status] += 1
        records_by_status[status].append(
            {
                "platform": item["platform"],
                "id": item["id"],
                "attempts": int(item.get("attempts") or 0),
                "started_at": item.get("started_at", ""),
                "last_duration_seconds": item.get("last_duration_seconds"),
                "updated_at": item.get("updated_at", ""),
            }
        )

    return {
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "state_path": str(RUN_STATE_PATH.resolve()),
        "record_count": int(state.get("record_count") or len(state.get("records", []))),
        "pending": counts["pending"],
        "in_progress": counts["in_progress"],
        "done": counts["done"],
        "failed": counts["failed"],
        "reclaimed": reclaimed,
        "records_by_status": records_by_status,
    }


def step5_next_pending_video() -> dict | None:
    state = _ensure_run_state()
    state, reclaimed = _reclaim_stale_in_progress(state)
    state_map = _state_record_map(state)

    for manifest_record in _load_manifest()["records"]:
        key = (manifest_record["platform"], manifest_record["id"])
        state_record = state_map.get(key)
        if not state_record:
            continue
        if state_record["status"] != "pending":
            continue
        now = _now_iso()
        state_record["status"] = "in_progress"
        state_record["attempts"] = int(state_record.get("attempts") or 0) + 1
        state_record["started_at"] = now
        state_record["last_duration_seconds"] = None
        state_record["updated_at"] = now
        _save_run_state(state)
        return {
            "platform": state_record["platform"],
            "id": state_record["id"],
            "status": state_record["status"],
            "attempts": state_record["attempts"],
            "started_at": state_record["started_at"],
            "updated_at": state_record["updated_at"],
            "reclaimed": reclaimed,
            "state_path": str(RUN_STATE_PATH.resolve()),
        }

    if reclaimed:
        _save_run_state(state)
    return None


def step5_prepare_video_context(platform: Any, id: Any) -> dict:
    manifest_record, chouzhen, top5_meta = _resolve_record(platform, id)
    platform = manifest_record["platform"]
    video_id = manifest_record["id"]
    _mark_in_progress_if_needed(platform, video_id)

    frame_paths = _resolve_frame_paths(manifest_record, chouzhen)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found for {platform}/{video_id}")

    target = _target_sample_size(len(frame_paths))
    sampled_frames = [frame_paths[index] for index in _sample_indices(len(frame_paths), target)]
    preview_path = _build_contact_sheet(sampled_frames, platform, video_id, manifest_record.get("label", ""))

    transcript, transcript_source = _resolve_transcript(manifest_record, chouzhen)
    if not transcript:
        raise FileNotFoundError(f"No transcript found for {platform}/{video_id}")

    top_comments = _resolve_top_comments(manifest_record, top5_meta)
    video_introduction = (
        manifest_record.get("video_introduction")
        or str(chouzhen.get("video_introduction") or "").strip()
        or str(top5_meta.get("video_introduction") or "").strip()
    )

    video_meta = _base_video_meta(manifest_record, chouzhen, top5_meta)
    video_meta["video_introduction"] = video_introduction
    if chouzhen.get("video_api_description"):
        video_meta["video_api_description"] = chouzhen.get("video_api_description")
    elif top5_meta.get("video_api_description"):
        video_meta["video_api_description"] = top5_meta.get("video_api_description")

    return {
        "video_meta": video_meta,
        "transcript": transcript,
        "transcript_source": transcript_source,
        "top_comments": top_comments,
        "sampled_frame_count": len(sampled_frames),
        "sampled_frame_names": [path.name for path in sampled_frames],
        "sampled_frame_paths": [_repo_relative_str(path) for path in sampled_frames],
        "preview_image_path": str(preview_path),
        "output_contract": (
            "Return exactly one polished video description paragraph in plain text. "
            "Do not use Markdown, bullets, JSON, headings, or analysis. "
            "Describe the video itself, reconcile noisy transcript text with the storyboard, "
            "and do not mention frames, screenshots, ASR, OCR, prompts, or model uncertainty."
        ),
    }


def _build_description_record(
    platform: str,
    video_id: str,
    description: str,
    existing_record: dict | None = None,
) -> dict:
    manifest_record, chouzhen, top5_meta = _resolve_record(platform, video_id)
    transcript, _ = _resolve_transcript(manifest_record, chouzhen)
    video_introduction = (
        manifest_record.get("video_introduction")
        or str(chouzhen.get("video_introduction") or "").strip()
        or str(top5_meta.get("video_introduction") or "").strip()
    )

    record = dict(existing_record or {})
    record["id"] = _preferred_output_id(platform, video_id, record, top5_meta, chouzhen)
    record["platform"] = platform
    record["video_url"] = (
        record.get("video_url")
        or manifest_record.get("video_url")
        or str(chouzhen.get("video_url") or "").strip()
        or str(top5_meta.get("video_url") or "").strip()
    )
    record["video_path"] = (
        manifest_record.get("video_path")
        or _repo_relative_str(chouzhen.get("video_path"))
        or str(record.get("video_path") or "")
    )
    record["video_introduction"] = video_introduction
    record["label"] = (
        manifest_record.get("label")
        or str(chouzhen.get("label") or "").strip()
        or str(top5_meta.get("label") or "").strip()
        or str(record.get("label") or "").strip()
    )
    record["image_root"] = (
        manifest_record.get("image_root")
        or _repo_relative_str(chouzhen.get("image_root"))
        or str(record.get("image_root") or "")
    )
    record["all_transcription"] = transcript
    record["video_description"] = description

    video_api_description = (
        str(record.get("video_api_description") or "").strip()
        or str(chouzhen.get("video_api_description") or "").strip()
        or str(top5_meta.get("video_api_description") or "").strip()
    )
    if video_api_description:
        record["video_api_description"] = video_api_description

    return record


def step5_save_description(platform: Any, id: Any, description: Any) -> dict:
    platform = _normalize_platform(platform)
    video_id = _normalize_video_id(id)
    normalized_description = _normalize_description_text(description)
    if not normalized_description:
        raise ValueError("Description must not be empty.")

    manifest_record, _, _ = _resolve_record(platform, video_id)
    output_path = _description_json_path(platform)
    records = _load_description_records(platform)
    record_map = {
        str(item.get("id", "")).strip(): item
        for item in records
        if isinstance(item, dict) and "id" in item
    }
    existing_record = record_map.get(video_id)
    existing_description = _normalize_description_text(
        existing_record.get("video_description", "") if existing_record else ""
    )

    if existing_record and existing_description:
        if existing_description != normalized_description:
            raise ValueError(
                f"Description already exists for {platform}/{video_id} and differs from the new content."
            )
        merged = _build_description_record(platform, video_id, existing_description, existing_record)
        record_map[video_id] = merged
        _dump_json_atomic(output_path, _sort_records_by_id(list(record_map.values())))
        state = _ensure_run_state()
        state_map = _state_record_map(state)
        state_record = state_map.get((platform, video_id))
        if state_record:
            duration = _seconds_since(state_record.get("started_at", ""))
            if duration is not None:
                state_record["last_duration_seconds"] = duration
            state_record["status"] = "done"
            state_record["started_at"] = ""
            state_record["last_error"] = ""
            state_record["updated_at"] = _now_iso()
            _save_run_state(state)
        return {
            "ok": True,
            "idempotent": True,
            "platform": platform,
            "id": video_id,
            "last_duration_seconds": state_record.get("last_duration_seconds") if state_record else None,
            "output_path": str(output_path.resolve()),
            "manifest_video_path": manifest_record.get("video_path", ""),
        }

    record_map[video_id] = _build_description_record(
        platform,
        video_id,
        normalized_description,
        existing_record=existing_record,
    )
    _dump_json_atomic(output_path, _sort_records_by_id(list(record_map.values())))

    state = _ensure_run_state()
    state_map = _state_record_map(state)
    state_record = state_map.get((platform, video_id))
    if state_record:
        duration = _seconds_since(state_record.get("started_at", ""))
        if duration is not None:
            state_record["last_duration_seconds"] = duration
        state_record["status"] = "done"
        state_record["started_at"] = ""
        state_record["last_error"] = ""
        state_record["updated_at"] = _now_iso()
        _save_run_state(state)

    return {
        "ok": True,
        "idempotent": False,
        "platform": platform,
        "id": video_id,
        "last_duration_seconds": state_record.get("last_duration_seconds") if state_record else None,
        "output_path": str(output_path.resolve()),
        "manifest_video_path": manifest_record.get("video_path", ""),
    }


def step5_mark_failed(platform: Any, id: Any, reason: Any) -> dict:
    platform = _normalize_platform(platform)
    video_id = _normalize_video_id(id)
    failure_reason = str(reason or "").strip()
    if not failure_reason:
        raise ValueError("Failure reason must not be empty.")

    _resolve_record(platform, video_id)
    state = _ensure_run_state()
    state_map = _state_record_map(state)
    state_record = state_map.get((platform, video_id))
    if not state_record:
        raise KeyError(f"Video not found in state: {platform}/{video_id}")

    if state_record["status"] != "done":
        duration = _seconds_since(state_record.get("started_at", ""))
        if duration is not None:
            state_record["last_duration_seconds"] = duration
        state_record["status"] = "failed"
        state_record["started_at"] = ""
        state_record["last_error"] = failure_reason
        state_record["updated_at"] = _now_iso()
        _save_run_state(state)
        changed = True
    else:
        changed = False

    return {
        "ok": True,
        "changed": changed,
        "platform": platform,
        "id": video_id,
        "status": state_record["status"],
        "last_error": state_record.get("last_error", ""),
        "last_duration_seconds": state_record.get("last_duration_seconds"),
        "state_path": str(RUN_STATE_PATH.resolve()),
    }


def step5_requeue_video(platform: Any, id: Any, reason: Any = "") -> dict:
    platform = _normalize_platform(platform)
    video_id = _normalize_video_id(id)
    note = str(reason or "").strip()

    _resolve_record(platform, video_id)
    state = _ensure_run_state()
    state_map = _state_record_map(state)
    state_record = state_map.get((platform, video_id))
    if not state_record:
        raise KeyError(f"Video not found in state: {platform}/{video_id}")

    if state_record["status"] == "done":
        changed = False
    else:
        state_record["status"] = "pending"
        state_record["started_at"] = ""
        state_record["last_duration_seconds"] = None
        if note:
            state_record["last_error"] = note
        state_record["updated_at"] = _now_iso()
        _save_run_state(state)
        changed = True

    return {
        "ok": True,
        "changed": changed,
        "platform": platform,
        "id": video_id,
        "status": state_record["status"],
        "last_error": state_record.get("last_error", ""),
        "attempts": int(state_record.get("attempts") or 0),
        "state_path": str(RUN_STATE_PATH.resolve()),
    }


TOOL_SPECS = [
    {
        "name": "step5_next_pending_video",
        "description": "Return the next pending video from the configured manifest and mark it in_progress.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "step5_prepare_video_context",
        "description": "Prepare transcript, comments, sampled frames, and one contact-sheet preview for a single video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["douyin", "youtube"]},
                "id": {"type": ["string", "integer"]},
            },
            "required": ["platform", "id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "step5_save_description",
        "description": "Write a non-empty final description into the standard description JSON and mark the video done.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["douyin", "youtube"]},
                "id": {"type": ["string", "integer"]},
                "description": {"type": "string"},
            },
            "required": ["platform", "id", "description"],
            "additionalProperties": False,
        },
    },
    {
        "name": "step5_mark_failed",
        "description": "Mark a video as failed without writing into the standard description JSON.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["douyin", "youtube"]},
                "id": {"type": ["string", "integer"]},
                "reason": {"type": "string"},
            },
            "required": ["platform", "id", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "step5_requeue_video",
        "description": "Move a non-done video back to pending so it can be retried later without duplicating saved descriptions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["douyin", "youtube"]},
                "id": {"type": ["string", "integer"]},
                "reason": {"type": "string"},
            },
            "required": ["platform", "id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "step5_run_status",
        "description": "Return pending, in_progress, done, and failed counts for the codex-native step5 run state.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]

TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "step5_next_pending_video": step5_next_pending_video,
    "step5_prepare_video_context": step5_prepare_video_context,
    "step5_save_description": step5_save_description,
    "step5_mark_failed": step5_mark_failed,
    "step5_requeue_video": step5_requeue_video,
    "step5_run_status": step5_run_status,
}


def _mcp_tool_content(name: str, result: Any) -> list[dict]:
    content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
    if name == "step5_prepare_video_context":
        preview_path = result.get("preview_image_path")
        if preview_path:
            try:
                data = base64.b64encode(Path(preview_path).read_bytes()).decode("ascii")
                content.append(
                    {
                        "type": "image",
                        "mimeType": "image/jpeg",
                        "data": data,
                    }
                )
            except Exception:
                pass
    return content


def _handle_tool_call(name: str, arguments: Any) -> dict:
    if name not in TOOL_HANDLERS:
        raise KeyError(f"Unknown tool: {name}")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")
    result = TOOL_HANDLERS[name](**arguments)
    return {
        "content": _mcp_tool_content(name, result),
        "structuredContent": result,
        "isError": False,
    }


def _read_mcp_message(stream) -> dict | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            if headers:
                break
            continue
        decoded = line.decode("utf-8").strip()
        if not decoded:
            if headers:
                break
            continue
        if ":" not in decoded:
            raise ValueError(f"Invalid MCP header line: {decoded!r}")
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        raise ValueError("Missing or invalid Content-Length header.")
    payload = stream.read(content_length)
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))


def _write_mcp_message(stream, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    stream.write(header)
    stream.write(body)
    stream.flush()


def _jsonrpc_error(message_id: Any, code: int, message: str, data: Any = None) -> dict:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": message_id, "error": error}


def _jsonrpc_result(message_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _handle_request(message: dict) -> dict | None:
    method = message.get("method")
    params = message.get("params") or {}
    message_id = message.get("id")

    if method == "initialize":
        _ensure_run_state()
        return _jsonrpc_result(
            message_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "capabilities": {"tools": {}},
            },
        )

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return _jsonrpc_result(message_id, {})

    if method == "tools/list":
        return _jsonrpc_result(message_id, {"tools": TOOL_SPECS})

    if method == "tools/call":
        try:
            name = params["name"]
        except Exception:
            return _jsonrpc_error(message_id, -32602, "Missing tool name.")
        try:
            result = _handle_tool_call(name, params.get("arguments", {}))
        except Exception as exc:
            return _jsonrpc_result(
                message_id,
                {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            )
        return _jsonrpc_result(message_id, result)

    return _jsonrpc_error(message_id, -32601, f"Method not found: {method}")


def run_mcp_server() -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        try:
            message = _read_mcp_message(stdin)
        except Exception as exc:
            error = _jsonrpc_error(None, -32700, "Parse error", str(exc))
            _write_mcp_message(stdout, error)
            continue

        if message is None:
            break
        response = _handle_request(message)
        if response is None:
            continue
        if "id" not in message:
            continue
        _write_mcp_message(stdout, response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MCP server for Codex-native step5 video description workflow.")
    parser.add_argument("--tool", help="Call one tool directly without starting the MCP server.")
    parser.add_argument("--args", default="{}", help="JSON object used with --tool.")
    parser.add_argument("--bootstrap-state", action="store_true", help="Create or sync the run-state JSON and exit.")
    parser.add_argument("--print-paths", action="store_true", help="Print resolved file paths and exit.")
    parser.add_argument("--manifest-path", help="Override the configured manifest path for this process.")
    parser.add_argument("--run-state-path", help="Override the configured run-state path for this process.")
    parser.add_argument("--preview-dir", help="Override the configured preview output directory for this process.")
    parser.add_argument("--douyin-description-path", help="Override the Douyin description JSON path.")
    parser.add_argument("--youtube-description-path", help="Override the YouTube description JSON path.")
    args = parser.parse_args()

    _configure_runtime_paths(
        manifest_path=args.manifest_path,
        run_state_path=args.run_state_path,
        preview_dir=args.preview_dir,
        douyin_description_path=args.douyin_description_path,
        youtube_description_path=args.youtube_description_path,
    )

    if args.print_paths:
        print(json.dumps(_print_paths(), ensure_ascii=False, indent=2))
        return

    if args.bootstrap_state:
        state = _ensure_run_state()
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    if args.tool:
        if args.tool not in TOOL_HANDLERS:
            raise SystemExit(f"Unknown tool: {args.tool}")
        call_args = json.loads(args.args or "{}")
        result = TOOL_HANDLERS[args.tool](**call_args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    run_mcp_server()


if __name__ == "__main__":
    main()
