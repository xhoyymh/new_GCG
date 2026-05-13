import argparse
import contextlib
import os
import time

from youtube_dataset_all_in_one_ollama import (
    LABEL_SLUG_MAP,
    URL_JSON,
    VIDEO_INTRO_JSON,
    assign_record_value,
    build_youtube_client,
    download_video,
    dump_json_atomic,
    extract_video_id,
    get_video_info,
    get_video_output_path,
    is_valid_media_file,
    load_json_data,
    resolve_video_path,
    sort_records_by_id,
    to_repo_relative,
)


LOCK_PATH = VIDEO_INTRO_JSON + ".lock"
LABEL_BY_SLUG = {slug: label for label, slug in LABEL_SLUG_MAP.items()}


@contextlib.contextmanager
def json_lock(lock_path: str = LOCK_PATH, timeout: float = 300.0, poll_interval: float = 0.25):
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("ascii", "ignore"))
            break
        except FileExistsError:
            if time.time() - start >= timeout:
                raise TimeoutError(f"timed out waiting for lock: {lock_path}")
            time.sleep(poll_interval)
    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


def load_video_url_records_for_label(label: str) -> list:
    if not os.path.exists(URL_JSON):
        return []
    data = load_json_data(URL_JSON)
    if not isinstance(data, list):
        return []
    return [item for item in sort_records_by_id(data) if isinstance(item, dict) and item.get("label", "") == label]


def load_video_intro_map() -> dict:
    if not os.path.exists(VIDEO_INTRO_JSON):
        return {}
    data = load_json_data(VIDEO_INTRO_JSON)
    if not isinstance(data, list):
        return {}
    return {str(item["id"]): dict(item) for item in data if isinstance(item, dict) and "id" in item}


def save_video_intro_map(output_map: dict) -> None:
    dump_json_atomic(VIDEO_INTRO_JSON, sort_records_by_id(list(output_map.values())))


def is_completed_record(source: dict, output_map: dict) -> bool:
    record_id = str(source.get("id", "")).strip()
    if not record_id:
        return False
    existing_record = output_map.get(record_id)
    if not existing_record:
        return False
    if not is_valid_media_file(resolve_video_path({**source, **existing_record})):
        return False
    assign_record_value(existing_record, "label", source.get("label", ""), force=True)
    assign_record_value(existing_record, "video_url", source.get("video_url", ""), force=True)
    resolved_path = resolve_video_path({**source, **existing_record})
    if resolved_path:
        assign_record_value(existing_record, "video_path", to_repo_relative(resolved_path), force=True)
    output_map[record_id] = existing_record
    return True


def download_label(label: str) -> None:
    records = load_video_url_records_for_label(label)
    if not records:
        print(f"[提示] 未找到 label={label} 的 URL 记录。")
        return

    youtube_client = build_youtube_client()
    skipped = added = failed = 0
    total = len(records)

    print(f"[开始] label={label} | 待扫描 {total} 条")

    for index, source in enumerate(records, start=1):
        record_id = str(source.get("id", "")).strip()
        video_url = source.get("video_url", "")

        if not record_id:
            failed += 1
            print(f"[失败] {index}/{total} | 缺少 id，已跳过")
            continue

        with json_lock():
            output_map = load_video_intro_map()
            if is_completed_record(source, output_map):
                skipped += 1
                continue

        video_id = extract_video_id(video_url)
        if not video_id:
            failed += 1
            print(f"[失败] {index}/{total} | ID={record_id} 无法解析 video_id")
            continue

        title, api_description = get_video_info(youtube_client, video_id, video_url=video_url)
        if not title and not api_description:
            failed += 1
            print(f"[失败] {index}/{total} | ID={record_id} 无法获取视频信息")
            continue

        output_path = get_video_output_path(record_id, label)
        if not download_video(video_url, output_path):
            failed += 1
            print(f"[失败] {index}/{total} | ID={record_id} 下载失败")
            continue

        with json_lock():
            output_map = load_video_intro_map()
            if is_completed_record(source, output_map):
                skipped += 1
                continue

            record = dict(output_map.get(record_id, {}))
            record["id"] = source.get("id", record_id)
            record["label"] = label
            record["video_url"] = video_url
            record["video_introduction"] = title
            record["video_api_description"] = api_description
            record["video_path"] = to_repo_relative(output_path)
            output_map[record_id] = record
            save_video_intro_map(output_map)
            added += 1
            print(f"[完成] {index}/{total} | ID={record_id} | 新增={added} | 跳过={skipped} | 失败={failed}")

    print(f"[结束] label={label} | 新增={added} | 跳过={skipped} | 失败={failed}")


def resolve_label(label: str, label_slug: str) -> str:
    if label:
        return label
    slug = (label_slug or "").strip()
    if not slug:
        return ""
    return LABEL_BY_SLUG.get(slug, "")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 label 并行下载 YouTube 视频，并用文件锁保护共享 JSON 断点。")
    parser.add_argument("--label", default="", help="要处理的 label 名称")
    parser.add_argument("--label-slug", default="", help="要处理的 label slug")
    args = parser.parse_args()
    label = resolve_label(args.label, args.label_slug)
    if not label:
        raise SystemExit("--label 或 --label-slug 必须提供有效值")
    download_label(label)


if __name__ == "__main__":
    main()
