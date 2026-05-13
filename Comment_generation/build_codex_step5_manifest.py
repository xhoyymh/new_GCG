from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PRE_ROOT = REPO_ROOT / "data_pre"

DOUYIN_TOP5_PATH = DATA_PRE_ROOT / "json" / "douyin" / "data_pre" / "douyin_top5_comments.json"
YOUTUBE_TOP5_PATH = DATA_PRE_ROOT / "json" / "youtube" / "data_pre" / "youtube_top5_comment.json"
DOUYIN_CHOUZHEN_PATH = DATA_PRE_ROOT / "json" / "douyin" / "data_pre" / "douyin_chouzhen.json"
YOUTUBE_CHOUZHEN_PATH = DATA_PRE_ROOT / "json" / "youtube" / "data_pre" / "youtube_chouzhen.json"
DOUYIN_DESCRIPTION_PATH = DATA_PRE_ROOT / "json" / "douyin" / "data_pre" / "douyin_video_description.json"
YOUTUBE_DESCRIPTION_PATH = DATA_PRE_ROOT / "json" / "youtube" / "data_pre" / "youtube_video_description.json"

DEFAULT_OUTPUT_PATH = DATA_PRE_ROOT / "json" / "materials" / "successful_step5_candidates_20260329.json"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _repo_relative(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Missing video id")
    return text


def _sort_key(value: str) -> tuple[int, str]:
    if value.isdigit():
        return (0, f"{int(value):012d}")
    return (1, value)


def _non_empty_text(value: Any) -> str:
    return str(value or "").strip()


def _collect_top5_comments(item: dict) -> list[str]:
    values: list[str] = []
    raw_list = item.get("top5_comments")
    if isinstance(raw_list, list):
        values.extend(_non_empty_text(entry) for entry in raw_list if _non_empty_text(entry))
    for index in range(1, 6):
        comment = _non_empty_text(item.get(f"comment_{index}"))
        if comment:
            values.append(comment)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _resolved_path(raw: Any) -> Path | None:
    text = _non_empty_text(raw)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _frame_root(image_root: Path | None) -> Path | None:
    if not image_root:
        return None
    frames_dir = image_root / "frames"
    if frames_dir.exists() and frames_dir.is_dir():
        return frames_dir
    return image_root if image_root.exists() and image_root.is_dir() else None


def _frame_paths_from_root(root: Path | None) -> list[Path]:
    if not root or not root.exists() or not root.is_dir():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: path.name.lower(),
    )


def _description_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for item in _load_json(path):
        if not isinstance(item, dict) or "id" not in item:
            continue
        if _non_empty_text(item.get("video_description")):
            ids.add(_normalize_id(item.get("id")))
    return ids


def _build_platform_records(
    platform: str,
    top5_path: Path,
    chouzhen_path: Path,
    require_comments: bool,
) -> tuple[list[dict], dict[str, int]]:
    top5_data = _load_json(top5_path)
    chouzhen_data = _load_json(chouzhen_path)

    top5_map = {
        _normalize_id(item.get("id")): item
        for item in top5_data
        if isinstance(item, dict) and "id" in item
    }
    chouzhen_map = {
        _normalize_id(item.get("id")): item
        for item in chouzhen_data
        if isinstance(item, dict) and "id" in item
    }

    stats = {
        "top5_records": len(top5_map),
        "chouzhen_records": len(chouzhen_map),
        "eligible_records": 0,
        "skipped_missing_top5": 0,
        "skipped_missing_comments": 0,
        "skipped_missing_transcript": 0,
        "skipped_missing_image_root": 0,
        "skipped_missing_frame_files": 0,
        "skipped_missing_video_file": 0,
        "skipped_missing_transcription_file": 0,
    }

    records: list[dict] = []
    for video_id in sorted(chouzhen_map.keys(), key=_sort_key):
        chouzhen = chouzhen_map[video_id]
        top5 = top5_map.get(video_id, {})
        if require_comments and not top5:
            stats["skipped_missing_top5"] += 1
            continue

        top5_comments = _collect_top5_comments(top5)
        if require_comments and not top5_comments:
            stats["skipped_missing_comments"] += 1
            continue

        transcript = _non_empty_text(chouzhen.get("all_transcription"))
        if not transcript:
            stats["skipped_missing_transcript"] += 1
            continue

        image_root = _resolved_path(chouzhen.get("image_root"))
        if not image_root or not image_root.exists():
            stats["skipped_missing_image_root"] += 1
            continue

        frame_root = _frame_root(image_root)
        frame_paths = _frame_paths_from_root(frame_root)
        if not frame_paths:
            stats["skipped_missing_frame_files"] += 1
            continue

        video_path = _resolved_path(chouzhen.get("video_path") or top5.get("video_path"))
        if not video_path or not video_path.exists():
            stats["skipped_missing_video_file"] += 1
            continue

        transcription_path = image_root / "transcription.txt"
        if not transcription_path.exists():
            stats["skipped_missing_transcription_file"] += 1
            continue

        video_introduction = (
            _non_empty_text(top5.get("video_introduction"))
            or _non_empty_text(chouzhen.get("video_introduction"))
        )
        video_url = _non_empty_text(chouzhen.get("video_url")) or _non_empty_text(top5.get("video_url"))
        label = _non_empty_text(chouzhen.get("label")) or _non_empty_text(top5.get("label"))
        source_result_file = (
            _non_empty_text(chouzhen.get("source_result_file"))
            or _non_empty_text(top5.get("source_result_file"))
        )

        records.append(
            {
                "platform": platform,
                "id": video_id,
                "label": label,
                "video_path": _repo_relative(video_path),
                "image_root": _repo_relative(image_root),
                "frame_root": _repo_relative(frame_root),
                "frame_count": len(frame_paths),
                "transcription_path": _repo_relative(transcription_path),
                "transcription_chars": len(transcript),
                "top5_comments": top5_comments,
                "video_introduction": video_introduction,
                "video_url": video_url,
                "source_result_file": source_result_file,
            }
        )
        stats["eligible_records"] += 1

    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Codex-native step5 backlog manifest from videos with transcript and frame materials.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output manifest path.")
    parser.add_argument(
        "--require-comments",
        action="store_true",
        help="Only include videos that also have non-empty top5 comments.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    douyin_records, douyin_stats = _build_platform_records(
        "douyin",
        DOUYIN_TOP5_PATH,
        DOUYIN_CHOUZHEN_PATH,
        require_comments=args.require_comments,
    )
    youtube_records, youtube_stats = _build_platform_records(
        "youtube",
        YOUTUBE_TOP5_PATH,
        YOUTUBE_CHOUZHEN_PATH,
        require_comments=args.require_comments,
    )

    all_records = douyin_records + youtube_records
    described_count = len(_description_ids(DOUYIN_DESCRIPTION_PATH)) + len(_description_ids(YOUTUBE_DESCRIPTION_PATH))

    payload = {
        "generated_at": _now_iso(),
        "purpose": (
            "Codex-native step5 backlog for videos that already have transcript and local frame "
            "materials. Comments are included when available and are only required when the CLI "
            "is run with --require-comments. Existing description JSONs are used later to mark "
            "done items."
        ),
        "selection": {
            "douyin": len(douyin_records),
            "youtube": len(youtube_records),
        },
        "criteria": {
            "requires_comments": bool(args.require_comments),
            "requires_non_empty_all_transcription": True,
            "requires_existing_image_root_and_frame_files": True,
            "requires_existing_video_file": True,
            "requires_existing_transcription_file": True,
        },
        "existing_description_count": described_count,
        "source_files": {
            "douyin_top5": _repo_relative(DOUYIN_TOP5_PATH),
            "youtube_top5": _repo_relative(YOUTUBE_TOP5_PATH),
            "douyin_chouzhen": _repo_relative(DOUYIN_CHOUZHEN_PATH),
            "youtube_chouzhen": _repo_relative(YOUTUBE_CHOUZHEN_PATH),
            "douyin_description": _repo_relative(DOUYIN_DESCRIPTION_PATH),
            "youtube_description": _repo_relative(YOUTUBE_DESCRIPTION_PATH),
        },
        "platform_stats": {
            "douyin": douyin_stats,
            "youtube": youtube_stats,
        },
        "record_count": len(all_records),
        "records": all_records,
    }

    _dump_json(output_path, payload)
    print(json.dumps({
        "ok": True,
        "output_path": str(output_path),
        "record_count": len(all_records),
        "selection": payload["selection"],
        "existing_description_count": described_count,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
