import argparse
import asyncio
import sys

import douyin_dataset_all_in_one_ollama as mod


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


async def download_label_until_target(label: str, target_count: int) -> None:
    video_list = mod.load_json_data(mod.VIDEO_URL_JSON)
    existing = mod.load_existing_output(mod.VIDEO_INTRO_JSON)

    current_count = 0
    for record in existing.values():
        if record.get("label", "") != label:
            continue
        resolved = mod.resolve_video_path(record)
        if resolved and mod.os.path.isfile(resolved):
            current_count += 1

    print(f"当前 {label} 已下载 {current_count}/{target_count} 条")
    if current_count >= target_count:
        print("已达到目标数量，无需继续下载。")
        return

    candidates = [item for item in video_list if item.get("label", "") == label]
    skipped = added = failed = 0

    for item in candidates:
        if current_count >= target_count:
            break

        video_id = str(item.get("id", "")).strip()
        video_url = item.get("video_url", "")
        if not video_id or not video_url:
            failed += 1
            continue

        existing_item = existing.get(video_id)
        if existing_item:
            resolved_existing = mod.resolve_video_path({**item, **existing_item})
            if resolved_existing and mod.os.path.isfile(resolved_existing):
                skipped += 1
                continue

        print(f"\n▶ 定向下载 {label} | ID={video_id} | 当前 {current_count}/{target_count}")

        try:
            aweme = await mod._get_aweme_detail(video_url)
        except Exception as e:
            print(f"[失败] ID={video_id} 获取 aweme_detail 异常：{e}")
            failed += 1
            continue

        if not aweme:
            print(f"[失败] ID={video_id} 无法获取 aweme_detail")
            failed += 1
            continue

        play_urls = aweme.get("video", {}).get("play_addr", {}).get("url_list", [])
        if not play_urls:
            print(f"[失败] ID={video_id} 未找到播放地址")
            failed += 1
            continue

        filepath = mod.get_video_output_path(video_id, label)
        mod.os.makedirs(mod.os.path.dirname(filepath), exist_ok=True)
        success = mod._download_video_file(play_urls[0], filepath)
        if not success:
            print(f"[失败] ID={video_id} 视频下载失败")
            failed += 1
            continue

        record = dict(existing_item or item)
        record["id"] = video_id
        record["video_url"] = video_url
        record["label"] = label
        record["video_introduction"] = aweme.get("desc", "")
        record["video_path"] = mod.to_repo_relative(filepath)
        existing[video_id] = record
        mod.dump_json_atomic(mod.VIDEO_INTRO_JSON, mod.sort_records_by_id(list(existing.values())))

        added += 1
        current_count += 1
        print(f"[完成] ID={video_id} -> {filepath}")

    print(
        f"\n{label} 下载结束：新增 {added} 条 | 跳过 {skipped} 条 | 失败 {failed} 条 | 当前总计 {current_count} 条"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="定向补齐指定 Douyin label 的下载数量")
    parser.add_argument("--label", required=True, help="要下载的 label")
    parser.add_argument("--target", type=int, required=True, help="目标下载数量")
    args = parser.parse_args()

    if args.target <= 0:
        raise SystemExit("--target 必须为正整数")

    asyncio.run(download_label_until_target(args.label, args.target))


if __name__ == "__main__":
    main()
