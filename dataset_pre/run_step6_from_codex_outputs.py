from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PRE_ROOT = REPO_ROOT / "data_pre"
CODE_ROOT = Path(__file__).resolve().parent

PLATFORM_CONFIGS = {
    "douyin": {
        "rules_source": CODE_ROOT / "douyin" / "douyin_dataset_all_in_one_ollama.py",
        "train_path": DATA_PRE_ROOT / "json" / "sample" / "douyin_comments_sample.json",
        "comments_path": DATA_PRE_ROOT / "json" / "douyin" / "data_pre" / "douyin_top5_comments.json",
        "description_path": DATA_PRE_ROOT / "json" / "douyin" / "data_pre" / "douyin_video_description_codex.json",
        "output_path": DATA_PRE_ROOT / "json" / "douyin" / "sample" / "douyin_sample_codex.json",
    },
    "youtube": {
        "rules_source": CODE_ROOT / "youtube" / "youtube_dataset_all_in_one_ollama.py",
        "train_path": DATA_PRE_ROOT / "json" / "sample" / "youtube_comments_sample.json",
        "comments_path": DATA_PRE_ROOT / "json" / "youtube" / "data_pre" / "youtube_top5_comment.json",
        "description_path": DATA_PRE_ROOT / "json" / "youtube" / "data_pre" / "youtube_video_description_codex.json",
        "output_path": DATA_PRE_ROOT / "json" / "youtube" / "sample" / "youtube_sample_codex.json",
    },
}


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> Any:
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


def _normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Missing id")
    return text


def _sort_records_by_id(records: list[dict]) -> list[dict]:
    def _sort_key(item: dict) -> tuple[int, str]:
        raw = str(item.get("id", "")).strip()
        if raw.isdigit():
            return (0, f"{int(raw):012d}")
        return (1, raw)

    return sorted(records, key=_sort_key)


def _non_empty_text(value: Any) -> str:
    return str(value or "").strip()


def _assign_record_value(record: dict, key: str, value: Any, force: bool = False) -> None:
    normalized = value
    if isinstance(value, str):
        normalized = value.strip()
    if isinstance(normalized, str) and not normalized:
        return
    if force:
        record[key] = normalized
        return
    if key not in record or record.get(key) in ("", None):
        record[key] = normalized


def _load_existing_output(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return {
        _normalize_id(item.get("id")): item
        for item in data
        if isinstance(item, dict) and item.get("id") not in (None, "")
    }


def _load_records(path: Path, expected: str) -> list[dict]:
    data = _load_json(path)
    if isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported {expected} JSON container: {path}")
    return [item for item in records if isinstance(item, dict)]


def _check_duplicates(records: list[dict], name: str) -> tuple[set[str], list[str]]:
    ids: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    dup_seen: set[str] = set()
    for item in records:
        raw = item.get("id")
        if raw in (None, ""):
            continue
        video_id = _normalize_id(raw)
        ids.append(video_id)
        if video_id in seen and video_id not in dup_seen:
            duplicates.append(video_id)
            dup_seen.add(video_id)
        seen.add(video_id)
    if duplicates:
        print(f"[warn] {name} duplicate ids: {duplicates[:10]}{' ...' if len(duplicates) > 10 else ''}")
    else:
        print(f"[ok] {name} has no duplicate ids")
    return set(ids), duplicates


@lru_cache(maxsize=None)
def _extract_literal_assignment(source_path: str, variable_name: str) -> Any:
    path = Path(source_path)
    source = path.read_text(encoding="utf-8-sig")
    module = ast.parse(source, filename=str(path))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == variable_name:
                return ast.literal_eval(node.value)
    raise KeyError(f"{variable_name} not found in {path}")


@lru_cache(maxsize=None)
def _platform_rules(platform: str) -> dict[str, Any]:
    config = PLATFORM_CONFIGS[platform]
    source_path = str(config["rules_source"])
    return {
        "emotion_rules": _extract_literal_assignment(source_path, "_S6_EMOTION_RULES"),
        "kw_rules": _extract_literal_assignment(source_path, "_S6_KW_RULES"),
        "emotion_to_label": _extract_literal_assignment(source_path, "_S6_EMOTION_TO_LABEL"),
    }


def _s6_clean(text: str) -> str:
    text = re.sub(r"\[[\w\u4e00-\u9fff]+\]", " ", text)
    text = re.sub(r"[#@][\w\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _s6_tokenize(text: str) -> list[str]:
    text = _s6_clean(_non_empty_text(text))
    tokens = [text[i : i + 2] for i in range(max(len(text) - 1, 0))]
    tokens += [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    return tokens


def _s6_build_idf(corpus: list[list[str]]) -> dict[str, float]:
    df: defaultdict[str, int] = defaultdict(int)
    total_docs = len(corpus)
    for doc in corpus:
        for token in set(doc):
            df[token] += 1
    return {token: math.log((total_docs + 1) / (count + 1)) + 1.0 for token, count in df.items()}


def _s6_tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    if not tokens:
        return {}
    tf = Counter(tokens)
    total = len(tokens)
    return {token: (count / total) * idf.get(token, 1.0) for token, count in tf.items()}


def _s6_cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(token, 0.0) * value for token, value in b.items())
    ma = math.sqrt(sum(value * value for value in a.values()))
    mb = math.sqrt(sum(value * value for value in b.values()))
    return dot / (ma * mb) if ma and mb else 0.0


class _S6TrainingIndex:
    def __init__(self, records: list[dict], idf: dict[str, float]) -> None:
        self.entries: list[tuple[dict[str, float], str, str]] = []
        self.prior: defaultdict[str, Counter] = defaultdict(Counter)
        for item in records:
            video_label = _non_empty_text(item.get("label"))
            for i in range(1, 6):
                comment = _non_empty_text(item.get(f"comment_{i}"))
                clabel = _non_empty_text(item.get(f"C{i}_label"))
                if comment and clabel:
                    vec = _s6_tfidf(_s6_tokenize(comment), idf)
                    self.entries.append((vec, clabel, video_label))
                    if video_label:
                        self.prior[video_label][clabel] += 1

    def predict(self, vec: dict[str, float], video_label: str = "", k: int = 5) -> str:
        if not self.entries:
            return "Plain Humor"
        scored = sorted(
            ((_s6_cosine(vec, entry[0]), entry[1]) for entry in self.entries),
            reverse=True,
        )
        top = scored[:k]
        if top and top[0][0] < 0.05 and video_label in self.prior:
            return self.prior[video_label].most_common(1)[0][0]
        return Counter(label for _, label in top).most_common(1)[0][0]


def _s6_detect_emotion(text: str, rules: dict[str, Any]) -> str:
    text = _non_empty_text(text)
    if not text:
        return "empty"
    for pattern, label in rules["emotion_rules"]:
        if re.search(pattern, text):
            return label
    return "neutral"


def _s6_kw_label(text: str, rules: dict[str, Any]) -> str | None:
    for _, pattern, label in sorted(rules["kw_rules"], key=lambda item: item[0]):
        if re.search(pattern, text):
            return label
    return None


def _s6_predict_clabel(
    comment: str,
    video_desc: str,
    video_label: str,
    idf: dict[str, float],
    index: _S6TrainingIndex,
    rules: dict[str, Any],
    desc_vec: dict[str, float],
) -> str:
    comment = _non_empty_text(comment)
    if not comment:
        return ""
    label = _s6_kw_label(comment, rules)
    if label:
        return label
    comment_vec = _s6_tfidf(_s6_tokenize(comment), idf)
    similarity = _s6_cosine(comment_vec, desc_vec)
    if similarity >= 0.10:
        return "Content Extraction"
    emotion = _s6_detect_emotion(comment, rules)
    if emotion in rules["emotion_to_label"]:
        return rules["emotion_to_label"][emotion]
    return index.predict(comment_vec, video_label)


def _comment_slots(top5_record: dict | None) -> list[str]:
    top5_record = top5_record or {}
    values = [_non_empty_text(top5_record.get(f"comment_{i}")) for i in range(1, 6)]
    if any(values):
        return values
    raw_top5 = top5_record.get("top5_comments")
    if isinstance(raw_top5, list):
        for index, value in enumerate(raw_top5[:5]):
            values[index] = _non_empty_text(value)
    return values


def _has_nonempty_comments(comment_slots: list[str]) -> bool:
    return any(_non_empty_text(item) for item in comment_slots)


def _build_output_record(
    desc_record: dict,
    top5_record: dict | None,
    comments: list[str],
    labels: list[str],
) -> dict:
    top5_record = top5_record or {}
    record = {
        "id": desc_record.get("id"),
        "video_url": _non_empty_text(desc_record.get("video_url")) or _non_empty_text(top5_record.get("video_url")),
        "video_path": _non_empty_text(desc_record.get("video_path")) or _non_empty_text(top5_record.get("video_path")),
        "video_introduction": _non_empty_text(desc_record.get("video_introduction"))
        or _non_empty_text(top5_record.get("video_introduction")),
        "label": _non_empty_text(desc_record.get("label")) or _non_empty_text(top5_record.get("label")),
        "video_description": _non_empty_text(desc_record.get("video_description")),
    }
    video_api_description = _non_empty_text(desc_record.get("video_api_description")) or _non_empty_text(
        top5_record.get("video_api_description")
    )
    image_root = _non_empty_text(desc_record.get("image_root"))
    if image_root:
        record["image_root"] = image_root
    if video_api_description:
        record["video_api_description"] = video_api_description

    for index in range(1, 6):
        record[f"comment_{index}"] = comments[index - 1]
        record[f"C{index}_label"] = labels[index - 1]
    return record


def _merge_existing_record(existing: dict, desc_record: dict, top5_record: dict | None, comments: list[str]) -> dict:
    top5_record = top5_record or {}
    record = dict(existing)
    record["id"] = existing.get("id", desc_record.get("id"))
    _assign_record_value(
        record,
        "video_url",
        _non_empty_text(desc_record.get("video_url")) or _non_empty_text(top5_record.get("video_url")),
        force=True,
    )
    _assign_record_value(record, "video_path", _non_empty_text(desc_record.get("video_path")), force=True)
    _assign_record_value(record, "image_root", _non_empty_text(desc_record.get("image_root")), force=True)
    _assign_record_value(
        record,
        "video_introduction",
        _non_empty_text(desc_record.get("video_introduction")) or _non_empty_text(top5_record.get("video_introduction")),
    )
    _assign_record_value(
        record,
        "video_api_description",
        _non_empty_text(desc_record.get("video_api_description")) or _non_empty_text(top5_record.get("video_api_description")),
    )
    _assign_record_value(record, "label", _non_empty_text(desc_record.get("label")) or _non_empty_text(top5_record.get("label")))
    _assign_record_value(record, "video_description", _non_empty_text(desc_record.get("video_description")))
    for index in range(1, 6):
        _assign_record_value(record, f"comment_{index}", comments[index - 1])
        record.setdefault(f"C{index}_label", _non_empty_text(existing.get(f"C{index}_label")))
    return record


def _eligible_desc_records(
    desc_records: list[dict],
    top5_map: dict[str, dict],
    include_empty_comments: bool,
) -> tuple[list[dict], int]:
    eligible: list[dict] = []
    skipped_no_comments = 0
    for record in desc_records:
        video_id = _normalize_id(record.get("id"))
        comments = _comment_slots(top5_map.get(video_id))
        if not include_empty_comments and not _has_nonempty_comments(comments):
            skipped_no_comments += 1
            continue
        eligible.append(record)
    return eligible, skipped_no_comments


def _load_platform_inputs(platform: str) -> dict[str, Any]:
    config = PLATFORM_CONFIGS[platform]
    train_records = _load_records(config["train_path"], "training")
    desc_records = _load_records(config["description_path"], "description")
    top5_records = _load_records(config["comments_path"], "comments")
    existing = _load_existing_output(config["output_path"])

    desc_map = {_normalize_id(item.get("id")): item for item in desc_records if item.get("id") not in (None, "")}
    top5_map = {_normalize_id(item.get("id")): item for item in top5_records if item.get("id") not in (None, "")}
    return {
        "config": config,
        "train_records": train_records,
        "desc_records": _sort_records_by_id(desc_records),
        "top5_records": top5_records,
        "desc_map": desc_map,
        "top5_map": top5_map,
        "existing": existing,
    }


def _run_platform(platform: str, include_empty_comments: bool, rebuild: bool) -> dict[str, Any]:
    inputs = _load_platform_inputs(platform)
    config = inputs["config"]
    train_records = inputs["train_records"]
    desc_records = inputs["desc_records"]
    top5_records = inputs["top5_records"]
    top5_map = inputs["top5_map"]
    existing = {} if rebuild else inputs["existing"]

    print(f"\n=== Step6 codex run: {platform} ===")
    _check_duplicates(train_records, f"{platform} train")
    desc_ids, _ = _check_duplicates(desc_records, f"{platform} codex descriptions")
    top5_ids, _ = _check_duplicates(top5_records, f"{platform} top5 comments")

    only_in_desc = len(desc_ids - top5_ids)
    only_in_top5 = len(top5_ids - desc_ids)
    print(
        json.dumps(
            {
                "platform": platform,
                "description_records": len(desc_records),
                "top5_records": len(top5_records),
                "existing_output_records": len(existing),
                "desc_without_top5_record": only_in_desc,
                "top5_without_desc_record": only_in_top5,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    eligible_records, skipped_no_comments = _eligible_desc_records(desc_records, top5_map, include_empty_comments)
    print(
        json.dumps(
            {
                "platform": platform,
                "eligible_records": len(eligible_records),
                "skipped_no_comments": skipped_no_comments,
                "include_empty_comments": include_empty_comments,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    corpus: list[list[str]] = []
    for item in train_records:
        corpus.append(_s6_tokenize(_non_empty_text(item.get("video_description"))))
        for index in range(1, 6):
            corpus.append(_s6_tokenize(_non_empty_text(item.get(f"comment_{index}"))))
    for item in desc_records:
        corpus.append(_s6_tokenize(_non_empty_text(item.get("video_description"))))

    idf = _s6_build_idf(corpus)
    index = _S6TrainingIndex(train_records, idf)
    rules = _platform_rules(platform)

    merged = dict(existing)
    added = 0
    skipped_existing = 0
    label_counter: Counter[str] = Counter()

    for desc_record in eligible_records:
        video_id = _normalize_id(desc_record.get("id"))
        top5_record = top5_map.get(video_id)
        comments = _comment_slots(top5_record)
        if video_id in merged:
            merged[video_id] = _merge_existing_record(merged[video_id], desc_record, top5_record, comments)
            skipped_existing += 1
            continue

        video_desc = _non_empty_text(desc_record.get("video_description"))
        desc_vec = _s6_tfidf(_s6_tokenize(video_desc), idf)
        labels: list[str] = []
        for comment in comments:
            clabel = _s6_predict_clabel(
                comment=comment,
                video_desc=video_desc,
                video_label=_non_empty_text(desc_record.get("label")),
                idf=idf,
                index=index,
                rules=rules,
                desc_vec=desc_vec,
            )
            labels.append(clabel)
            if clabel:
                label_counter[clabel] += 1

        merged[video_id] = _build_output_record(desc_record, top5_record, comments, labels)
        added += 1

    output_records = _sort_records_by_id(list(merged.values()))
    _dump_json_atomic(config["output_path"], output_records)

    summary = {
        "platform": platform,
        "generated_at": _now_iso(),
        "output_path": str(config["output_path"]),
        "description_records": len(desc_records),
        "top5_records": len(top5_records),
        "eligible_records": len(eligible_records),
        "added": added,
        "skipped_existing": skipped_existing,
        "skipped_no_comments": skipped_no_comments,
        "output_count": len(output_records),
        "label_distribution": dict(label_counter.most_common()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_platform(raw: str) -> list[str]:
    raw = (raw or "all").strip().lower()
    if raw == "all":
        return ["douyin", "youtube"]
    if raw not in PLATFORM_CONFIGS:
        raise SystemExit(f"Unsupported platform: {raw}")
    return [raw]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build codex-native Step6 sample outputs from codex Step5 descriptions and top5 comments."
    )
    parser.add_argument("--platform", choices=("douyin", "youtube", "all"), default="all")
    parser.add_argument(
        "--include-empty-comments",
        action="store_true",
        help="Also include videos that do not have any non-empty comment slots.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore existing *_sample_codex.json outputs and rebuild from scratch.",
    )
    args = parser.parse_args()

    summaries = []
    for platform in _parse_platform(args.platform):
        summaries.append(_run_platform(platform, args.include_empty_comments, args.rebuild))

    print("\n=== Step6 codex summary ===")
    print(json.dumps({"platforms": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
