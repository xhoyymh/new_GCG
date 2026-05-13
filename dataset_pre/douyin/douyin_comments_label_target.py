import argparse
import sys
from typing import Iterable, List

import douyin_dataset_all_in_one_ollama as mod


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_LABELS = [
    "Comedy Skits",
    "Daily Life Jokes",
    "Funny Animal Videos",
    "Humorous Commentary",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_labels(raw_values: Iterable[str]) -> List[str]:
    labels: List[str] = []
    for raw in raw_values:
        for part in raw.split(","):
            label = part.strip()
            if label and label not in labels:
                labels.append(label)
    return labels


def run_comments_for_labels(labels: List[str]) -> None:
    video_list = mod.load_json_data(mod.VIDEO_INTRO_JSON)
    if not isinstance(video_list, list):
        raise SystemExit(f"VIDEO_INTRO_JSON 格式异常: {mod.VIDEO_INTRO_JSON}")

    label_set = set(labels)
    candidates = [
        item for item in mod.sort_records_by_id(video_list)
        if item.get("label", "") in label_set
    ]

    existing_top5 = mod.load_existing_output(mod.TOP5_COMMENTS_JSON)
    existing_all = mod.load_all_comments_output()
    label_counts, normalized_existing = mod.normalize_existing_comment_outputs(existing_top5, existing_all)
    if normalized_existing:
        mod.save_comment_outputs(existing_top5, existing_all)

    _log(f"准备抓取评论的 label: {', '.join(labels)}")
    _log(f"候选视频数: {len(candidates)}")
    _log(f"已存在 top5 记录: {len(existing_top5)}")

    comment_api = "aweme/v1/web/comment/list/"
    driver = mod.build_comment_driver()
    skipped = added = failed = 0

    try:
        for item in candidates:
            video_id = str(item.get("id", "")).strip()
            label = item.get("label", "")
            if not video_id:
                failed += 1
                continue

            if video_id in existing_top5:
                comments = mod.normalize_comment_sample(existing_all.get(video_id, []))
                existing_all[video_id] = comments
                merged_item = dict(existing_top5[video_id])
                mod.assign_record_value(merged_item, "video_url", item.get("video_url", ""), force=True)
                mod.assign_record_value(merged_item, "video_introduction", item.get("video_introduction", ""), force=True)
                mod.assign_record_value(merged_item, "label", label, force=True)
                mod.assign_record_value(merged_item, "video_path", item.get("video_path", ""), force=True)
                existing_top5[video_id] = mod.build_comment_record(merged_item, comments)
                skipped += 1
                continue

            if label and label_counts.get(label, 0) >= mod.COMMENT_LABEL_TARGET:
                skipped += 1
                continue

            _log(f"开始抓评论: ID={video_id} | {label}")
            try:
                comments = mod.fetch_comment_sample(driver, item.get("video_url", ""), comment_api=comment_api)
            except Exception as e:
                failed += 1
                _log(f"[失败] ID={video_id} 评论抓取异常: {e}")
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = mod.build_comment_driver()
                continue

            existing_all[video_id] = comments
            existing_top5[video_id] = mod.build_comment_record(item, comments)
            mod.save_comment_outputs(existing_top5, existing_all)
            added += 1
            if label:
                label_counts[label] += 1
            _log(f"[完成] ID={video_id} | 评论数={len(comments)} | {label_counts.get(label, 0)}/{mod.COMMENT_LABEL_TARGET}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    mod.save_comment_outputs(existing_top5, existing_all)
    _log(
        f"评论抓取结束: 新增 {added} 条 | 跳过 {skipped} 条 | 失败 {failed} 条 | 当前总计 {len(existing_top5)} 条"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="按指定 Douyin label 抓取评论并实时断点写盘")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=DEFAULT_LABELS,
        help="要抓评论的 label，支持空格分隔或逗号分隔；默认前四类",
    )
    args = parser.parse_args()

    labels = _parse_labels(args.labels)
    if not labels:
        raise SystemExit("至少需要一个有效的 label")

    run_comments_for_labels(labels)


if __name__ == "__main__":
    main()
