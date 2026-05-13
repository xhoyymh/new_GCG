"""
抖音视频全流程一键脚本
Pipeline:
  Step 1: 采集视频 URL            → douyin_video_url.json
  Step 2: 下载视频 + 获取简介     → douyin_video_introduction.json + video/*.mp4
  Step 3: 抓取评论                → douyin_top5_comments.json + douyin_all_comments.json
  Step 4: 抽帧 + 音频转录         → douyin_chouzhen.json + image/<id>/frames/
  Step 5: Ollama 生成视频描述      → douyin_video_description.json
  Step 6: C_label 自动标注        → douyin_sample.json

依赖安装：
  pip install selenium playwright DrissionPage tqdm opencv-python openai-whisper pydub requests
  playwright install chromium
"""

import os
import re
import json
import time
import asyncio
import hashlib
import sys
import requests
import subprocess
import tempfile
import traceback
import argparse
from collections import Counter
from urllib.parse import quote

import cv2
import whisper
from tqdm import tqdm


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from DrissionPage import ChromiumOptions, ChromiumPage
from ollama import Client as OllamaClient


def normalize_ollama_host():
    raw_host = (os.environ.get("OLLAMA_HOST") or "").strip()
    if not raw_host:
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
        return
    if "://" not in raw_host:
        raw_host = f"http://{raw_host}"
    if "0.0.0.0" in raw_host:
        raw_host = raw_host.replace("0.0.0.0", "127.0.0.1")
    scheme, rest = raw_host.split("://", 1)
    if "/" not in rest and ":" not in rest:
        raw_host = f"{scheme}://{rest}:11434"
    os.environ["OLLAMA_HOST"] = raw_host


normalize_ollama_host()
OLLAMA_CLIENT = OllamaClient(host=os.environ["OLLAMA_HOST"])

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════
#  ★ 全局配置区 — 所有路径和参数在此统一修改 ★
# ════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# 中间/输出文件路径
VIDEO_URL_JSON          = os.path.join(BASE_DIR, "data_pre","json", "douyin", "data_pre", "douyin_video_url.json")
VIDEO_INTRO_JSON        = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre","douyin_video_introduction.json")
ALL_COMMENTS_JSON       = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre", "douyin_all_comments.json")
TOP5_COMMENTS_JSON      = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre", "douyin_top5_comments.json")
CHOUZHEN_JSON           = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre", "douyin_chouzhen.json")
VIDEO_DESCRIPTION_JSON  = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre", "douyin_video_description.json")

# Step 6: C_label 标注
SAMPLE_TRAIN_JSON       = os.path.join(BASE_DIR, "data_pre", "json", "sample", "douyin_comments_sample.json")
DOUYIN_SAMPLE_JSON      = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "sample", "douyin_sample.json")

# 参考样本 JSON（用于 label/tag 分析，Step 1）
SAMPLE_JSON_PATH        = os.path.join(SCRIPT_DIR, "douyin_video_sample.json")

# 文件夹路径
VIDEO_DIR               = os.path.join(BASE_DIR, "data_pre", "video", "douyin")
IMAGE_DIR               = os.path.join(BASE_DIR, "data_pre", "douyin_image")
USERDATA_DIR            = os.path.join(BASE_DIR, "data_pre", "userdata_douyin")
CHROME_BINARY_PATH      = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CHROME_PROFILE_DIR      = "Default"

# Step 3: 评论参数
SCROLL_ROUNDS = 8  # 评论页最大滚动次数
COMMENT_CAPTURE_MODE = "full_source"
COMMENT_CAPTURE_CAP = 50
COMMENT_LABEL_TARGET = 0  # <= 0 表示按来源文件全量补抓，不做每个 label 的数量上限
COMMENT_TOP_K = 5
COMMENT_EMPTY_ROUND_LIMIT = 2
COMMENT_INITIAL_WAIT_TIMEOUT = 3.0
COMMENT_SCROLL_PAUSE = 0.75
COMMENT_RESPONSE_TIMEOUT = 0.75

# Step 4: 抽帧参数
FRAME_FPS               = 1                        # 每秒抽帧数
WHISPER_MODEL_NAME      = "base"                   # Whisper 模型大小：tiny / base / small / medium
FFMPEG_PATH             = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "ffmpeg-8.1-essentials_build",
    "bin",
    "ffmpeg.exe",
)  # 可改为空，走 PATH/自动查找

# Step 5: Ollama 参数
OLLAMA_MODEL            = "qwen3.5:9b"   # 本地 Ollama 多模态模型名称
MAX_IMAGES_PER_BATCH    = 5           # 每批传入的最大帧数
FRAME_INTERVAL          = 3           # 传给模型的帧间隔（每隔 N 帧取一张）

# Step 1: Label → 自动搜索词映射
# 选定 Label 后，对应的搜索词会自动加入 tag 搜索，提升抖音搜索命中率
# 可按需增删，key 与 douyin_video_sample.json 中的 label 字段保持一致
LABEL_AUTO_TAGS_MAP = {
    "Comedy Skits":                          ["搞笑短剧", "情景喜剧"],
    "Funny Animal Videos":                   ["搞笑动物"],
    "Daily Life Jokes":                      ["生活搞笑"],
    "Humorous Commentary":                   ["搞笑吐槽"],
    "Talk Shows / Stand-Up Comedy / Cross-Talk": ["脱口秀", "单口喜剧", "相声", "小品"],
}

LABEL_SLUG_MAP = {
    "Comedy Skits": "comedy_skits",
    "Funny Animal Videos": "funny_animal_videos",
    "Daily Life Jokes": "daily_life_jokes",
    "Humorous Commentary": "humorous_commentary",
    "Talk Shows / Stand-Up Comedy / Cross-Talk": "talk_show_standup_crosstalk",
}

LABEL_QUERY_HINTS_MAP = {
    "Comedy Skits": [
        "搞笑短剧", "情景喜剧", "搞笑", "短剧",
        "反转短剧", "沙雕短剧", "下饭短剧", "剧情反转",
    ],
    "Daily Life Jokes": [
        "生活搞笑", "搞笑日常", "日常搞笑", "校园搞笑",
        "女大学生", "男大学生", "crush", "校园vlog",
    ],
    "Funny Animal Videos": [
        "搞笑动物", "动物的迷惑行为", "萌宠", "猫咪",
        "狗狗", "傻狗搞笑日常", "猫咪的迷惑行为", "专治不开心",
    ],
    "Humorous Commentary": [
        "搞笑吐槽", "搞笑解说", "吐槽", "动物世界",
        "动物世界搞笑解说", "电影解说", "废话文学", "反转",
    ],
    "Talk Shows / Stand-Up Comedy / Cross-Talk": [
        "脱口秀", "单口喜剧", "相声", "小品",
        "脱口秀爆梗名场面", "脱口秀搞笑视频", "搞笑脱口秀", "脱口秀互动",
        "脱口秀大会", "吐槽大会", "单口喜剧专场", "脱口秀演员",
        "相声合集", "相声名场面", "相声小品", "爆笑小品",
        "喜剧小品", "春晚小品", "德云社", "郭德纲",
        "岳云鹏", "孟鹤堂", "烧饼曹鹤阳", "付航脱口秀",
        "徐志胜", "何广智", "李雪琴", "庞博",
    ],
}

BATCH_COLLECTION_STAGES = [
    {"max_duration_sec": 300, "tag_limit": 2},
    {"max_duration_sec": 300, "tag_limit": 5},
    {"max_duration_sec": 480, "tag_limit": 8},
    {"max_duration_sec": 720, "tag_limit": 12},
    {"max_duration_sec": 1200, "tag_limit": 20},
    {"max_duration_sec": 1800, "tag_limit": 28},
    {"max_duration_sec": 3600, "tag_limit": 36},
]

VERIFICATION_KEYWORDS = (
    "验证码中间页",
    "验证码",
    "安全验证",
    "验证后继续访问",
    "请完成下列验证",
    "抖音安全验证",
)

# ── pydub / ffmpeg 初始化（配置区末尾，勿移动）─────────────────────
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", RuntimeWarning)
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg    = FFMPEG_PATH
    AudioSegment.ffprobe   = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    _ffmpeg_bin_dir = os.path.dirname(FFMPEG_PATH)
    os.environ["PATH"] = _ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")


# ════════════════════════════════════════════════════════════════
#  STEP 1: 采集视频 URL
# ════════════════════════════════════════════════════════════════

def parse_duration(time_str: str) -> int:
    parts = time_str.strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return 0
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def build_search_url(tag: str) -> str:
    return f"https://www.douyin.com/search/{quote(tag)}"


def load_json_data(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_output(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(item["id"]): item for item in data if "id" in item}
    except Exception as e:
        print(f"  [警告] 读取已有输出文件失败：{e}，将视为空文件处理。")
        return {}


def dump_json_atomic(filepath: str, data) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=os.path.dirname(filepath),
        suffix=".tmp",
        delete=False,
    ) as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        temp_path = tf.name
    os.replace(temp_path, filepath)


def dedupe_keep_order(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def sort_records_by_id(records: list) -> list:
    return sorted(
        records,
        key=lambda x: int(x["id"]) if str(x.get("id", "")).isdigit() else float("inf")
    )


def sort_id_keys(keys) -> list:
    def _sort_key(value):
        try:
            return int(str(value))
        except Exception:
            return float("inf")
    return sorted(keys, key=_sort_key)


def get_label_slug(label: str) -> str:
    label = (label or "").strip()
    if not label:
        return "unknown"
    if label in LABEL_SLUG_MAP:
        return LABEL_SLUG_MAP[label]

    slug = label.lower().replace("&", " and ")
    slug = re.sub(r"[\\/]+", " ", slug)
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    if slug:
        return slug

    digest = hashlib.md5(label.encode("utf-8")).hexdigest()[:8]
    return f"label_{digest}"


def to_repo_path(path_value: str) -> str:
    path_value = (path_value or "").strip()
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(BASE_DIR, path_value))


def to_repo_relative(path_value: str) -> str:
    abs_path = os.path.abspath(path_value)
    try:
        if os.path.commonpath([BASE_DIR, abs_path]) == BASE_DIR:
            return os.path.relpath(abs_path, BASE_DIR)
    except ValueError:
        pass
    return abs_path


def build_video_relpath(video_id: str, label: str) -> str:
    return os.path.join("data_pre", "video", "douyin", get_label_slug(label), f"{video_id}.mp4")


def get_video_output_path(video_id: str, label: str) -> str:
    return os.path.join(BASE_DIR, build_video_relpath(video_id, label))


def build_image_root_relpath(video_id: str, label: str) -> str:
    return os.path.join("data_pre", "douyin_image", get_label_slug(label), str(video_id))


def get_image_output_root(video_id: str, label: str) -> str:
    return os.path.join(BASE_DIR, build_image_root_relpath(video_id, label))


def resolve_video_path(record: dict) -> str:
    video_id = str(record.get("id", "")).strip()
    label = record.get("label", "")
    candidates = []

    for key in ("video_path", "video_file"):
        value = record.get(key, "")
        if value:
            candidates.append(to_repo_path(value))

    if video_id:
        candidates.append(get_video_output_path(video_id, label))
        candidates.append(os.path.join(VIDEO_DIR, f"{video_id}.mp4"))

    candidates = dedupe_keep_order([c for c in candidates if c])
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0] if candidates else ""


def resolve_image_root(record: dict) -> str:
    video_id = str(record.get("id", "")).strip()
    label = record.get("label", "")
    candidates = []

    image_root = record.get("image_root", "")
    if image_root:
        candidates.append(to_repo_path(image_root))
    if video_id:
        candidates.append(get_image_output_root(video_id, label))
        candidates.append(os.path.join(IMAGE_DIR, video_id))

    candidates = dedupe_keep_order([c for c in candidates if c])
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0] if candidates else ""


def load_all_comments_output() -> dict:
    if not os.path.exists(ALL_COMMENTS_JSON):
        return {}
    try:
        with open(ALL_COMMENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
    except Exception as e:
        print(f"  [警告] 读取已有评论详情失败：{e}，将视为空文件处理。")
    return {}


def normalize_comment_sample(comments, like_key: str = "digg_count") -> list:
    normalized = []
    for item in comments or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        like_value = item.get(like_key, item.get("likeCount", item.get("digg_count", 0)))
        try:
            digg_count = int(like_value or 0)
        except Exception:
            digg_count = 0
        normalized.append({
            "text": text,
            "digg_count": digg_count,
        })
    normalized.sort(key=lambda x: x["digg_count"], reverse=True)
    return normalized[:COMMENT_CAPTURE_CAP]


def normalize_comment_label_target(label_target=None) -> int:
    raw_value = COMMENT_LABEL_TARGET if label_target is None else label_target
    try:
        normalized = int(raw_value)
    except (TypeError, ValueError):
        return 0
    return normalized if normalized > 0 else 0


def get_comment_capture_mode(label_target=None) -> str:
    return "label_target" if normalize_comment_label_target(label_target) > 0 else COMMENT_CAPTURE_MODE


def describe_comment_collection_mode(label_target=None) -> str:
    normalized_target = normalize_comment_label_target(label_target)
    if normalized_target > 0:
        return f"按标签上限模式（每类最多 {normalized_target} 条）"
    return "全量模式"


def build_comment_record(source: dict, comments: list, comment_capture_mode: str | None = None) -> dict:
    current_capture_mode = comment_capture_mode or COMMENT_CAPTURE_MODE
    record = {
        "id": str(source.get("id", "")).strip(),
        "video_url": source.get("video_url", ""),
        "video_introduction": source.get("video_introduction", ""),
        "label": source.get("label", ""),
        "comment_capture_mode": current_capture_mode,
        "comment_capture_cap": COMMENT_CAPTURE_CAP,
        "comment_capture_count": len(comments),
    }
    if source.get("video_path", ""):
        record["video_path"] = source["video_path"]
    for index in range(1, COMMENT_TOP_K + 1):
        record[f"comment_{index}"] = comments[index - 1]["text"] if index - 1 < len(comments) else ""
    return record


def save_comment_outputs(top5_map: dict, all_comments_map: dict) -> None:
    ordered_top5 = sort_records_by_id(list(top5_map.values()))
    dump_json_atomic(TOP5_COMMENTS_JSON, ordered_top5)

    ordered_all = {}
    all_keys = set(all_comments_map) | {str(item.get("id", "")) for item in ordered_top5 if item.get("id", "") != ""}
    for key in sort_id_keys(all_keys):
        ordered_all[str(key)] = normalize_comment_sample(all_comments_map.get(str(key), []))
    dump_json_atomic(ALL_COMMENTS_JSON, ordered_all)


def normalize_existing_comment_outputs(top5_map: dict, all_comments_map: dict, comment_capture_mode: str | None = None) -> tuple[Counter, bool]:
    label_counts = Counter()
    changed = False

    for record_id, record in list(top5_map.items()):
        comments = normalize_comment_sample(all_comments_map.get(record_id, []))
        updated_record = build_comment_record(record, comments, comment_capture_mode=comment_capture_mode)
        if comments != all_comments_map.get(record_id, []):
            all_comments_map[record_id] = comments
            changed = True
        if updated_record != record:
            top5_map[record_id] = updated_record
            changed = True
        label = top5_map[record_id].get("label", "")
        if label:
            label_counts[label] += 1

    return label_counts, changed


def build_comment_driver():
    co = ChromiumOptions()
    co.headless(True)
    co.set_user_data_path(USERDATA_DIR)
    return ChromiumPage(co)


def fetch_comment_sample(driver, video_url: str, comment_api: str = 'aweme/v1/web/comment/list/') -> list:
    seen_ids = set()
    collected = []
    empty_rounds = 0

    def _consume_comment_events(timeout: float) -> int:
        end_time = time.time() + max(timeout, 0)
        while time.time() < end_time:
            try:
                resp = driver.listen.wait(timeout=min(COMMENT_RESPONSE_TIMEOUT, max(end_time - time.time(), 0.05)))
            except Exception:
                continue
            if comment_api not in getattr(resp, "url", ""):
                continue
            try:
                data = resp.response.body
            except Exception:
                data = {}
            comments = data.get("comments", []) if isinstance(data, dict) else []
            before = len(collected)
            if comments:
                collected.extend(_collect_comments(comments, seen_ids))
            return len(collected) - before
        return 0

    driver.listen.start(comment_api)
    try:
        driver.get(video_url, timeout=10, retry=0, interval=0)
        _consume_comment_events(COMMENT_INITIAL_WAIT_TIMEOUT)

        for _ in range(SCROLL_ROUNDS):
            if len(collected) >= COMMENT_CAPTURE_CAP or empty_rounds >= COMMENT_EMPTY_ROUND_LIMIT:
                break
            _scroll_comment_container(driver)
            time.sleep(COMMENT_SCROLL_PAUSE)
            added = _consume_comment_events(COMMENT_RESPONSE_TIMEOUT)
            if added > 0:
                empty_rounds = 0
            else:
                empty_rounds += 1
    finally:
        try:
            driver.listen.stop()
        except Exception:
            pass

    return normalize_comment_sample(collected)


def resolve_frame_paths(record: dict) -> list:
    frame_paths = []
    for frame_path in record.get("image", []) or []:
        resolved = to_repo_path(frame_path)
        if os.path.isfile(resolved):
            frame_paths.append(resolved)
    if frame_paths:
        return sorted(frame_paths)

    image_root = resolve_image_root(record)
    if not image_root:
        return []

    for candidate_dir in (os.path.join(image_root, "frames"), image_root):
        if os.path.isdir(candidate_dir):
            frame_paths = sorted(
                os.path.join(candidate_dir, name)
                for name in os.listdir(candidate_dir)
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if frame_paths:
                return frame_paths
    return []


def assign_record_value(record: dict, key: str, value, force: bool = False) -> None:
    if value in ("", None, [], {}):
        return
    if force or record.get(key) in ("", None, [], {}):
        record[key] = value


def extract_douyin_video_id(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""

    match = re.search(r"/video/(\d+)", value)
    if match:
        return match.group(1)

    match = re.search(r"(?:^|[\\/])(\d+)\.(?:mp4|mov|avi)$", value, re.IGNORECASE)
    if match:
        return match.group(1)

    return value if value.isdigit() else ""


def load_video_url_records() -> list:
    if not os.path.exists(VIDEO_URL_JSON):
        return []
    data = load_json_data(VIDEO_URL_JSON)
    return [data] if isinstance(data, dict) else data


def save_video_url_records(records: list) -> None:
    dump_json_atomic(VIDEO_URL_JSON, sort_records_by_id(records))


def get_next_numeric_id(records: list) -> int:
    numeric_ids = [int(str(item.get("id", "")).strip()) for item in records if str(item.get("id", "")).strip().isdigit()]
    return max(numeric_ids, default=0) + 1


def build_seen_video_ids(records: list) -> set:
    seen = set()
    for item in records:
        video_id = extract_douyin_video_id(item.get("video_url", ""))
        if video_id:
            seen.add(video_id)
    return seen


def count_records_for_label(records: list, label: str) -> int:
    count = 0
    seen_ids = set()
    for item in records:
        if item.get("label", "") != label:
            continue
        video_id = extract_douyin_video_id(item.get("video_url", "")) or f"id::{item.get('id', '')}"
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        count += 1
    return count


def build_label_query_candidates(data: list, label: str, sample_top_n: int = 20) -> list:
    auto_tags = LABEL_AUTO_TAGS_MAP.get(label, [])
    fallback_tags = LABEL_QUERY_HINTS_MAP.get(label, [])
    sample_tags = [tag for tag, _ in get_top_tags(data, label, top_n=sample_top_n)] if data else []
    query_candidates = dedupe_keep_order(auto_tags + fallback_tags + sample_tags)
    return [tag for tag in query_candidates if tag]


def normalize_search_query(tag: str) -> str:
    tag = (tag or "").strip()
    if not tag:
        return ""
    return tag if tag.startswith("#") else f"#{tag}"


def collect_urls_for_label_target(label: str, data: list, target_count: int) -> int:
    records = load_video_url_records()
    current_count = count_records_for_label(records, label)
    if current_count >= target_count:
        print(f"  [跳过] Label={label} 已有 {current_count}/{target_count} 条 URL")
        return current_count

    query_candidates = build_label_query_candidates(data, label)
    if not query_candidates:
        print(f"  [警告] Label={label} 没有可用搜索 tag，跳过。")
        return current_count

    print("\n" + "=" * 60)
    print(f"  自动采集 Label：{label}")
    print(f"  当前已有：{current_count} / 目标：{target_count}")
    print(f"  候选 Query：{', '.join(query_candidates[:12])}")
    print("=" * 60)

    seen_video_ids = build_seen_video_ids(records)
    existing_ids = {str(item.get("id", "")).strip() for item in records}
    next_id = get_next_numeric_id(records)
    used_queries = set()
    driver = init_selenium_driver(headless=False)

    try:
        for stage in BATCH_COLLECTION_STAGES:
            if current_count >= target_count:
                break

            stage_queries = query_candidates[:stage["tag_limit"]]
            if len(stage_queries) < len(query_candidates) and stage["tag_limit"] >= len(query_candidates):
                stage_queries = list(query_candidates)

            print(f"\n  [阶段] Label={label} | 时长上限={stage['max_duration_sec']} 秒 | 查询数={len(stage_queries)}")

            for raw_query in stage_queries:
                if current_count >= target_count:
                    break
                if raw_query in used_queries:
                    continue

                used_queries.add(raw_query)
                remaining = target_count - current_count
                search_query = normalize_search_query(raw_query)

                print(f"\n  [查询] Label={label} | Query={search_query} | 还需 {remaining} 条")

                try:
                    found_ids = crawl_tag(
                        driver,
                        search_query,
                        stage["max_duration_sec"],
                        remaining,
                        interactive_verification=False,
                    )
                except SystemExit as e:
                    print(f"  [中止] Label={label} 采集被取消：{e}")
                    return current_count
                except Exception as e:
                    print(f"  [失败] Label={label} Query={search_query} 抓取异常：{e}")
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    driver = init_selenium_driver(headless=False)
                    continue

                new_unique_ids = []
                for vid in found_ids:
                    vid = str(vid).strip()
                    if not vid or vid in seen_video_ids:
                        continue
                    seen_video_ids.add(vid)
                    new_unique_ids.append(vid)

                if not new_unique_ids:
                    print(f"  [结果] Query={search_query} 没有新增 URL")
                    continue

                for vid in new_unique_ids:
                    while str(next_id) in existing_ids:
                        next_id += 1
                    new_record = {
                        "id": str(next_id),
                        "video_url": f"https://www.douyin.com/video/{vid}",
                        "label": label,
                    }
                    records.append(new_record)
                    existing_ids.add(str(next_id))
                    next_id += 1
                    current_count += 1
                    if current_count >= target_count:
                        break

                save_video_url_records(records)
                print(f"  [进度] Label={label} 当前 {current_count}/{target_count}")

            if current_count < target_count:
                print(f"  [阶段结束] Label={label} 当前仅 {current_count}/{target_count}，继续扩大 tag 或放宽时长")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    return current_count


def step1_batch_collect_all_labels(target_count: int = 500) -> None:
    if not os.path.exists(SAMPLE_JSON_PATH):
        raise SystemExit(f"样本文件不存在：{SAMPLE_JSON_PATH}")

    data = load_json_data(SAMPLE_JSON_PATH)
    labels = [label for label in get_labels(data) if label in LABEL_AUTO_TAGS_MAP]
    summary = {}

    print("\n" + "★" * 60)
    print(f"  STEP 1（批量）：五类 Label 自动采集，每类目标 {target_count} 条 URL")
    print("★" * 60)

    for label in labels:
        summary[label] = collect_urls_for_label_target(label, data, target_count)

    print("\n" + "=" * 60)
    print("  批量采集汇总")
    for label in labels:
        print(f"  {label}: {summary.get(label, 0)}/{target_count}")
    print(f"  输出文件：{VIDEO_URL_JSON}")
    print("=" * 60)


def get_labels(data: list) -> list:
    counter = Counter(item.get("label", "") for item in data)
    return [label for label, _ in counter.most_common() if label]


def get_top_tags(data: list, label: str, top_n: int = 30) -> list:
    tags = []
    for item in data:
        if item.get("label", "") == label:
            found = re.findall(r"#(\S+)", item.get("video_introduction", ""))
            tags.extend(found)
    return Counter(tags).most_common(top_n)


def select_label(labels: list) -> tuple:
    print("\n┌─────────────────────────────────────────┐")
    print("│           请选择视频 Label               │")
    print("├─────────────────────────────────────────┤")
    for i, label in enumerate(labels, 1):
        print(f"│  {i:>2}. {label:<37}│")
    print(f"│   N. 输入全新 Label（自定义）           │")
    print("└─────────────────────────────────────────┘")
    while True:
        choice = input("请输入序号或 N：").strip()
        if choice.upper() == "N":
            new_label = input("请输入新 Label 名称：").strip()
            if new_label:
                return new_label, True
            print("  Label 不能为空，请重新输入。")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(labels):
                    return labels[idx], False
            except ValueError:
                pass
            print(f"  无效输入，请输入 1~{len(labels)} 或 N。")


def select_tags(top_tags: list, is_new_label: bool) -> list:
    chosen = []
    if not is_new_label and top_tags:
        print("\n┌─────────────────────────────────────────────────────────┐")
        print("│       该 Label 下出现次数最多的 Tag（Top 30）            │")
        print("├────┬──────────────────────────────────────┬─────────────┤")
        print("│ 序号│ Tag                                  │ 出现次数    │")
        print("├────┼──────────────────────────────────────┼─────────────┤")
        for i, (tag, cnt) in enumerate(top_tags, 1):
            print(f"│{i:>3} │ {tag:<36} │ {cnt:<11} │")
        print("└────┴──────────────────────────────────────┴─────────────┘")
        print("\n请选择要搜索的 Tag（可多选）：")
        print("  • 输入序号，多个用逗号分隔，例如：1,3,5")
        print("  • 直接回车跳过，只使用自定义 Tag")
        print("  • 输入 A 全选")
        sel_input = input("选择序号：").strip()
        if sel_input.upper() == "A":
            chosen = [tag for tag, _ in top_tags]
            print(f"  已全选 {len(chosen)} 个 Tag。")
        elif sel_input:
            for part in sel_input.split(","):
                part = part.strip()
                try:
                    idx = int(part) - 1
                    if 0 <= idx < len(top_tags):
                        chosen.append(top_tags[idx][0])
                    else:
                        print(f"  序号 {part} 超出范围，已忽略。")
                except ValueError:
                    print(f"  无效序号 '{part}'，已忽略。")

    if is_new_label:
        print("\n（全新 Label，请输入自定义搜索 Tag）")
    else:
        print("\n如需追加自定义 Tag，请输入（多个用逗号分隔，直接回车跳过）：")
    custom_input = input("自定义 Tag：").strip()
    if custom_input:
        custom_tags = [t.strip() for t in custom_input.split(",") if t.strip()]
        chosen.extend(custom_tags)

    seen = set()
    result = []
    for t in chosen:
        if t not in seen:
            seen.add(t)
            result.append(t)
    if not result:
        print("  未选择手动 Tag。")
    return result


def init_selenium_driver(headless: bool = False, chromedriver_path: str = "") -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    if CHROME_BINARY_PATH and os.path.isfile(CHROME_BINARY_PATH):
        options.binary_location = CHROME_BINARY_PATH
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f"--user-data-dir={USERDATA_DIR}")
    options.add_argument(f"--profile-directory={CHROME_PROFILE_DIR}")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--lang=zh-CN")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(executable_path=chromedriver_path) if chromedriver_path else Service()
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def is_verification_page(driver) -> tuple[bool, str]:
    try:
        title = driver.title or ""
    except Exception:
        title = ""
    try:
        current_url = driver.current_url or ""
    except Exception:
        current_url = ""
    try:
        page_source = driver.page_source or ""
    except Exception:
        page_source = ""

    haystack = "\n".join([title, current_url, page_source[:50000]])
    for keyword in VERIFICATION_KEYWORDS:
        if keyword in haystack:
            return True, keyword
    return False, ""


def wait_for_manual_verification(driver, context: str = "", interactive: bool = True) -> None:
    while True:
        is_verifying, keyword = is_verification_page(driver)
        if not is_verifying:
            return

        print("\n  [提示] 检测到抖音验证码/安全验证页面。")
        if context:
            print(f"  场景：{context}")
        if keyword:
            print(f"  识别关键字：{keyword}")
        print("  浏览器将保持打开，请在浏览器中手动完成验证。")

        if interactive:
            user_input = input("  完成后按回车继续，输入 q 退出 Step 1：").strip().lower()
            if user_input == "q":
                raise SystemExit("用户取消 Step 1：验证码未完成。")
            time.sleep(2)
        else:
            print("  [等待] 请在浏览器中手动完成验证，脚本将每 5 秒自动重试。")
            time.sleep(5)


def extract_videos(driver, max_duration_sec: int, max_count: int, seen_ids: set, interactive_verification: bool = True) -> list:
    """从当前页面提取符合时长条件的视频 ID，返回 video_id 字符串列表"""
    results = []
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[id^='waterfall_item_']"))
        )
    except Exception:
        is_verifying, _ = is_verification_page(driver)
        if is_verifying:
            wait_for_manual_verification(driver, context="等待搜索结果卡片", interactive=interactive_verification)
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[id^='waterfall_item_']"))
                )
            except Exception:
                print("  [警告] 验证完成后仍未找到视频卡片，页面结构可能已变化。")
                return results
        else:
            print("  [警告] 未找到视频卡片，页面可能需要登录或结构已变化。")
            return results

    cards = driver.find_elements(By.CSS_SELECTOR, "div[id^='waterfall_item_']")
    print(f"  当前页面共发现 {len(cards)} 个视频卡片")

    for card in cards:
        if len(results) >= max_count:
            break

        card_id = card.get_attribute("id")
        match = re.search(r"waterfall_item_(\d+)", card_id)
        if not match:
            continue
        video_id = match.group(1)
        if video_id in seen_ids:
            continue

        # 提取时长（使用新版 CSS selector）
        duration_str = ""
        for sel in ["div.FnM1bbIQ", "div[class*='FnM1bbIQ']",
                    "span[class*='duration']", "div[class*='duration']"]:
            try:
                el = card.find_element(By.CSS_SELECTOR, sel)
                duration_str = el.text.strip()
                if duration_str:
                    break
            except Exception:
                continue

        # 时长过滤
        if duration_str:
            if parse_duration(duration_str) > max_duration_sec:
                print(f"  跳过 {video_id}（时长 {duration_str} 超过限制）")
                continue
        else:
            print(f"  警告：未获取到 {video_id} 的时长，默认保留")

        seen_ids.add(video_id)
        results.append(video_id)
        print(f"  ✓ {video_id}  时长={duration_str or '未知'}")

    return results


def scroll_and_collect(driver, max_duration_sec: int, max_count: int,
                       scroll_pause: float = 2.5, interactive_verification: bool = True) -> list:
    """
    持续滚动页面采集视频，直到满足以下任一条件才停止：
      1. 已采集到 max_count 条符合条件的视频
      2. 页面连续 3 次高度不变，确认已到底
    不设滚动次数上限，确保用户指定的数量能被满足。
    """
    all_ids = []
    seen_ids: set = set()
    no_new_content_count = 0
    scroll_i = 0

    while len(all_ids) < max_count:
        scroll_i += 1
        print(f"\n[第 {scroll_i} 次扫描] 已采集 {len(all_ids)}/{max_count} 条")
        batch = extract_videos(
            driver,
            max_duration_sec,
            max_count - len(all_ids),
            seen_ids,
            interactive_verification=interactive_verification,
        )
        all_ids.extend(batch)

        if len(all_ids) >= max_count:
            print(f"  [完成] 已达到目标数量 {max_count} 条，停止采集。")
            break

        prev_h = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        new_h = driver.execute_script("return document.body.scrollHeight")

        if new_h == prev_h:
            no_new_content_count += 1
            if no_new_content_count >= 3:
                print(f"  [提示] 页面连续 3 次无新内容，搜索结果已耗尽。")
                print(f"  [提示] 最终采集到 {len(all_ids)}/{max_count} 条。")
                break
            else:
                print(f"  [等待] 页面高度未变，稍等后重试（{no_new_content_count}/3）...")
                time.sleep(scroll_pause)
        else:
            no_new_content_count = 0

    return all_ids


def crawl_tag(driver, tag: str, max_duration_sec: int, max_count: int, interactive_verification: bool = True) -> list:
    """打开搜索页并采集"""
    search_url = build_search_url(tag)
    print(f"\n  搜索 URL：{search_url}")
    driver.get(search_url)
    time.sleep(2)
    wait_for_manual_verification(driver, context=f"打开搜索页 {tag}", interactive=interactive_verification)
    return scroll_and_collect(
        driver,
        max_duration_sec,
        max_count,
        interactive_verification=interactive_verification,
    )


def step1_crawl_urls():
    """Step 1: 采集视频 URL，保存至 VIDEO_URL_JSON"""
    print("\n" + "═"*60)
    print("  STEP 1：采集抖音视频 URL")
    print("═"*60)

    # ── 加载样本 JSON 用于 label/tag 分析 ──
    data, labels = [], []
    if os.path.exists(SAMPLE_JSON_PATH):
        data = load_json_data(SAMPLE_JSON_PATH)
        labels = get_labels(data)
        print(f"  已读取样本文件 {len(data)} 条，共 {len(labels)} 个 Label。")

    # ── 选择 Label ──
    if labels:
        chosen_label, is_new_label = select_label(labels)
    else:
        chosen_label = input("\n请输入 Label 名称：").strip() or "unknown"
        is_new_label = True

    # ── 选择 / 输入 Tag ──
    top_tags = get_top_tags(data, chosen_label, top_n=30) if not is_new_label and data else []
    manual_tags = select_tags(top_tags, is_new_label)

    # ── 自动将 Label 的搜索词加入搜索 tag ──
    auto_tags = LABEL_AUTO_TAGS_MAP.get(chosen_label, [])
    chosen_tags = dedupe_keep_order(list(auto_tags) + manual_tags)
    if auto_tags:
        print('  [自动] Label "' + chosen_label + '" → 搜索词 "' + ', '.join(auto_tags) + '" 已加入 tag 列表')
    elif not is_new_label:
        print('  [提示] Label "' + chosen_label + '" 在 LABEL_AUTO_TAGS_MAP 中无对应搜索词，'
              '如需可在配置区手动添加')
    if not chosen_tags:
        raise SystemExit("未选择任何 Tag，终止。")

    # ── 其他参数交互 ──
    max_dur_str = input("\n最大视频时长（格式 MM:SS，直接回车默认 5:00）：").strip() or "5:00"
    max_duration_sec = parse_duration(max_dur_str)
    if max_duration_sec <= 0:
        print("  输入格式有误，使用默认值 5:00")
        max_dur_str = "5:00"
        max_duration_sec = 300

    try:
        count = int(input("需要采集多少条视频 URL（总数）：").strip())
        if count <= 0:
            raise ValueError
    except ValueError:
        count = 30
        print(f"  输入无效，默认 {count} 条")

    # ── 加载已有输出，自动推算默认 id 起始 ──
    existing = load_existing_output(VIDEO_URL_JSON)
    if existing:
        print(f"  检测到已有输出文件，共 {len(existing)} 条记录，重复 id 将保留原内容。")
    default_start_id = max((int(k) for k in existing.keys()), default=0) + 1 if existing else 1
    try:
        start_id = int(input(f"id 从多少开始（直接回车默认 {default_start_id}）：").strip() or str(default_start_id))
        if start_id <= 0:
            raise ValueError
    except ValueError:
        start_id = default_start_id

    # ── 汇总确认 ──
    print(f"\n{'═'*55}")
    print(f"  Label    : {chosen_label}")
    print(f"  Tags     : {', '.join(chosen_tags)}")
    print(f"  时长上限 : {max_dur_str}（{max_duration_sec} 秒）")
    print(f"  采集总数 : {count} 条")
    print(f"  id 起始  : {start_id}")
    print(f"  保存路径 : {VIDEO_URL_JSON}")
    print(f"{'═'*55}")

    # ── 启动浏览器 ──
    driver = init_selenium_driver(headless=False)
    new_video_ids = []

    try:
        combined_tag = "#" + "#".join(chosen_tags)
        print(f"\n{'─'*55}")
        print(f"  搜索词：{combined_tag}")
        print(f"{'─'*55}")
        new_video_ids = crawl_tag(driver, combined_tag, max_duration_sec, count)
        print(f"  采集完成：{len(new_video_ids)} 条")
    finally:
        driver.quit()

    # ── 合并新旧数据（id 冲突时保留原内容） ──
    skipped = 0
    added = 0
    merged = dict(existing)
    seen_video_ids = build_seen_video_ids(list(merged.values()))
    next_id = start_id

    for vid in new_video_ids:
        vid = str(vid).strip()
        if not vid or vid in seen_video_ids:
            skipped += 1
            continue

        while str(next_id) in merged:
            next_id += 1

        merged[str(next_id)] = {
            "id": str(next_id),
            "video_url": f"https://www.douyin.com/video/{vid}",
            "label": chosen_label
        }
        seen_video_ids.add(vid)
        next_id += 1
        added += 1

    output_list = sort_records_by_id(list(merged.values()))
    save_video_url_records(output_list)

    print(f"\n{'═'*55}")
    print(f"  新增条数 : {added} 条")
    print(f"  跳过条数 : {skipped} 条（id 重复，保留原内容）")
    print(f"  文件总计 : {len(output_list)} 条")
    print(f"  Label    : {chosen_label}")
    print(f"  保存路径 : {VIDEO_URL_JSON}")
    print(f"{'═'*55}")
    print(f"\n  ✅ Step 1 完成 → {VIDEO_URL_JSON}")


# ════════════════════════════════════════════════════════════════
#  STEP 2: 下载视频 + 获取视频简介
# ════════════════════════════════════════════════════════════════

async def _get_aweme_detail(share_url):
    aweme_detail = None
    found_event = asyncio.Event()

    async def intercept_aweme_response(response):
        nonlocal aweme_detail
        try:
            if 'application/json' in response.headers.get('content-type', '').lower():
                json_body = await response.json()
                if isinstance(json_body, dict) and 'aweme_detail' in json_body:
                    aweme_detail = json_body['aweme_detail']
                    found_event.set()
        except:
            pass

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(java_script_enabled=True)
        page = await context.new_page()
        page.on("response", intercept_aweme_response)

        async def navigate_and_wait():
            try:
                await page.goto(share_url, wait_until="networkidle", timeout=60000)
            except TimeoutError:
                pass

        task = asyncio.create_task(navigate_and_wait())
        try:
            await asyncio.wait_for(found_event.wait(), timeout=15)
        except asyncio.TimeoutError:
            print(f"[警告] 未从网络中获取到 aweme_detail: {share_url}")

        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        for obj in (page, context, browser):
            try:
                await obj.close()
            except:
                pass

    return aweme_detail


def _download_video_file(video_url, filename):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(video_url, headers=headers, stream=True, timeout=30)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return os.path.exists(filename) and os.path.getsize(filename) > 0
    except Exception as e:
        print(f"[错误] 下载失败: {e}")
        return False


async def step2_download_videos():
    """Step 2: 下载视频并提取简介，保存至 VIDEO_INTRO_JSON。已存在的 id 直接保留，不重复下载。"""
    print("\n" + "═"*60)
    print("  STEP 2：下载视频 + 获取视频简介")
    print("═"*60)

    os.makedirs(VIDEO_DIR, exist_ok=True)
    with open(VIDEO_URL_JSON, "r", encoding="utf-8") as f:
        video_list = json.load(f)

    # 加载已有输出，key = id 字符串
    existing: dict = load_existing_output(VIDEO_INTRO_JSON)
    if existing:
        print(f"  检测到已有输出文件，共 {len(existing)} 条记录，已存在的 id 将直接保留。")

    skipped, added, failed = 0, 0, 0

    print(f"\n📥 共需处理 {len(video_list)} 个视频...\n")

    for item in tqdm(video_list, desc="下载进度", unit="视频"):
        video_url = item.get("video_url", "")
        video_id  = str(item.get("id", "")).strip()

        # ── 已存在则跳过 ──
        if video_id in existing:
            existing_item = existing[video_id]
            assign_record_value(existing_item, "video_url", video_url, force=True)
            assign_record_value(existing_item, "label", item.get("label", ""), force=True)
            assign_record_value(existing_item, "video_introduction", item.get("video_introduction", ""))
            resolved_existing_path = resolve_video_path({**item, **existing_item})
            if resolved_existing_path and os.path.isfile(resolved_existing_path):
                assign_record_value(existing_item, "video_path", to_repo_relative(resolved_existing_path), force=True)
            tqdm.write(f"  [跳过] ID={video_id} 已存在，保留原内容。")
            skipped += 1
            continue

        tqdm.write(f"\n▶ 正在处理: ID={video_id} URL={video_url}")

        aweme = await _get_aweme_detail(video_url)
        if not aweme:
            tqdm.write("[失败] 无法获取 aweme_detail")
            failed += 1
            continue

        item["video_introduction"] = aweme.get("desc", "")
        item["label"] = item.get("label", "")

        play_urls = aweme.get("video", {}).get("play_addr", {}).get("url_list", [])
        if not play_urls:
            tqdm.write("[失败] 未找到播放地址")
            failed += 1
            continue

        filepath = get_video_output_path(video_id, item.get("label", ""))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        success = _download_video_file(play_urls[0], filepath)
        if success:
            tqdm.write(f"[完成] 视频已保存到: {filepath}")
        else:
            tqdm.write(f"[失败] 视频下载失败")
            failed += 1
            continue

        item["video_path"] = to_repo_relative(filepath)
        existing[video_id] = dict(item)
        result_list = sort_records_by_id(list(existing.values()))
        dump_json_atomic(VIDEO_INTRO_JSON, result_list)
        added += 1

    # ── 最终按 id 排序写出，确保空结果时也能落盘 ──
    result_list = sort_records_by_id(list(existing.values()))
    dump_json_atomic(VIDEO_INTRO_JSON, result_list)

    print(f"\n  新增：{added} 条 | 跳过（已存在）：{skipped} 条 | 失败：{failed} 条 | 文件总计：{len(result_list)} 条")
    print(f"  ✅ Step 2 完成 → {VIDEO_INTRO_JSON}")


# ════════════════════════════════════════════════════════════════
#  STEP 3: 抓取评论
# ════════════════════════════════════════════════════════════════

def _collect_comments(comments, seen_ids):
    result = []
    for comment in comments:
        try:
            cid = comment['cid']
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            result.append({'text': comment['text'], 'digg_count': comment['digg_count']})
        except Exception as e:
            print(f"⚠️ Failed to process comment: {e}")
    return result


def _scroll_comment_container(driver):
    scroll_script = '''
        let ele = document.querySelector('div[data-e2e="comment-list"]');
        while (ele && ele.scrollHeight <= ele.clientHeight) {
            ele = ele.parentElement;
        }
        if (ele) {
            ele.scrollTo(0, ele.scrollHeight);
            ele.dispatchEvent(new Event('scroll'));
            return ele.scrollHeight;
        } else { return -1; }
    '''
    return driver.run_js(scroll_script)


def step3_download_comments(comment_label_target=None):
    """Step 3: 抓取每个视频的评论，保存到 Top5 评论与评论样本 JSON。"""
    normalized_label_target = normalize_comment_label_target(comment_label_target)
    comment_capture_mode = get_comment_capture_mode(normalized_label_target)
    mode_text = describe_comment_collection_mode(normalized_label_target)
    skip_reason_text = "已存在/已达 label 上限/无效记录" if normalized_label_target > 0 else "已存在/无效记录"

    print("\n" + "═" * 60)
    print(f"  STEP 3：抓取视频评论（{mode_text}）")
    print("═" * 60)

    source_json = VIDEO_INTRO_JSON if os.path.exists(VIDEO_INTRO_JSON) else VIDEO_URL_JSON
    with open(source_json, 'r', encoding='utf-8') as f:
        video_list = json.load(f)
        if isinstance(video_list, dict):
            video_list = [video_list]

    existing_top5 = load_existing_output(TOP5_COMMENTS_JSON)
    existing_all = load_all_comments_output()
    label_counts, normalized_existing = normalize_existing_comment_outputs(
        existing_top5,
        existing_all,
        comment_capture_mode=comment_capture_mode,
    )
    if existing_top5:
        print(f"  检测到已有输出文件，共 {len(existing_top5)} 条记录，已存在的 id 将直接保留。")
    if normalized_existing:
        save_comment_outputs(existing_top5, existing_all)

    driver = build_comment_driver()
    comment_api = 'aweme/v1/web/comment/list/'
    skipped, added, failed = 0, 0, 0

    for item in tqdm(video_list, desc="抓取评论", unit="video"):
        vid = str(item.get('id', '')).strip()
        url = item.get('video_url', '')
        label = item.get("label", "")
        if not vid or not url:
            failed += 1
            continue

        if vid in existing_top5:
            comments = normalize_comment_sample(existing_all.get(vid, []))
            existing_all[vid] = comments
            merged_item = dict(existing_top5[vid])
            assign_record_value(merged_item, "video_url", url, force=True)
            assign_record_value(merged_item, "video_introduction", item.get("video_introduction", ""), force=True)
            assign_record_value(merged_item, "label", label, force=True)
            assign_record_value(merged_item, "video_path", item.get("video_path", ""), force=True)
            existing_top5[vid] = build_comment_record(
                merged_item,
                comments,
                comment_capture_mode=comment_capture_mode,
            )
            skipped += 1
            continue

        if normalized_label_target > 0 and label and label_counts.get(label, 0) >= normalized_label_target:
            skipped += 1
            continue

        try:
            comments = fetch_comment_sample(driver, url, comment_api=comment_api)
            existing_all[vid] = comments
            existing_top5[vid] = build_comment_record(
                item,
                comments,
                comment_capture_mode=comment_capture_mode,
            )
            save_comment_outputs(existing_top5, existing_all)
            added += 1
            if label:
                label_counts[label] += 1
        except Exception as e:
            failed += 1
            tqdm.write(f"  [失败] ID={vid} 评论抓取异常：{e}")
            try:
                driver.quit()
            except Exception:
                pass
            driver = build_comment_driver()
        finally:
            try:
                driver.listen.stop()
            except Exception:
                pass

    try:
        driver.quit()
    except Exception:
        pass

    save_comment_outputs(existing_top5, existing_all)

    print(f"\n  新增：{added} 条 | 跳过（{skip_reason_text}）：{skipped} 条 | 失败：{failed} 条 | 文件总计：{len(existing_top5)} 条")
    print(f"  ✅ Step 3 完成 → {ALL_COMMENTS_JSON}")
    print(f"            → {TOP5_COMMENTS_JSON}")


# ════════════════════════════════════════════════════════════════
#  STEP 4: 抽帧 + 音频转录
# ════════════════════════════════════════════════════════════════

_ffmpeg_exe = None  # 缓存，避免重复查找

def _find_ffmpeg() -> str:
    """
    按优先级查找 ffmpeg 可执行文件，并同步告知 pydub：
      1. 配置区 FFMPEG_PATH（手动指定）
      2. 系统 PATH（shutil.which）
      3. Windows 常见安装目录
    找不到则抛出 FileNotFoundError 并给出安装提示。
    """
    global _ffmpeg_exe
    if _ffmpeg_exe:
        return _ffmpeg_exe

    import shutil
    from pydub import AudioSegment

    def _set(path: str) -> str:
        """找到后同步设置 pydub、注入 PATH（供 Whisper 等直接调用 ffmpeg 的库使用），并缓存"""
        global _ffmpeg_exe
        _ffmpeg_exe = path
        # 告知 pydub
        AudioSegment.converter = path
        AudioSegment.ffmpeg    = path
        AudioSegment.ffprobe   = path.replace("ffmpeg.exe", "ffprobe.exe")
        # 将 ffmpeg 所在目录注入当前进程的 PATH
        # Whisper / subprocess 调用 "ffmpeg" 时会从 PATH 里找，这样就能找到
        ffmpeg_dir = os.path.dirname(path)
        current_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in current_path:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
        return path

    # 1. 手动指定
    if FFMPEG_PATH:
        if os.path.isfile(FFMPEG_PATH):
            return _set(FFMPEG_PATH)
        raise FileNotFoundError(
            f"配置区 FFMPEG_PATH 指定的路径不存在：{FFMPEG_PATH}\n"
            "请检查路径是否正确。"
        )

    # 2. 系统 PATH
    found = shutil.which("ffmpeg")
    if found:
        return _set(found)

    # 3. Windows 常见目录
    common_dirs = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""), "ffmpeg", "bin", "ffmpeg.exe"),
    ]
    for path in common_dirs:
        if os.path.isfile(path):
            print(f"  [ffmpeg] 在 {path} 找到，建议将其加入系统 PATH 或在配置区设置 FFMPEG_PATH。")
            return _set(path)

    raise FileNotFoundError(
        "\n[错误] 找不到 ffmpeg！请按以下任一方式解决：\n"
        "  方式 A（推荐）：下载 ffmpeg 并加入系统 PATH\n"
        "    下载地址：https://www.gyan.dev/ffmpeg/builds/  （选 ffmpeg-release-essentials.zip）\n"
        "    解压后将 bin 目录加入环境变量 PATH，重启终端即可。\n"
        "  方式 B：在脚本顶部配置区填写完整路径，例如：\n"
        "    FFMPEG_PATH = r'C:\\ffmpeg\\bin\\ffmpeg.exe'\n"
    )


def _extract_audio(video_path):
    ffmpeg = _find_ffmpeg()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = tmp.name
    tmp.close()
    cmd = [ffmpeg, "-i", video_path, "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-y", audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def _transcribe(audio_path, model):
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]


def _find_ffprobe() -> str:
    ffmpeg = _find_ffmpeg()
    ffprobe = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
    if os.path.isfile(ffprobe):
        return ffprobe

    import shutil

    found = shutil.which("ffprobe")
    if found:
        return found

    common_dirs = [
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "ffmpeg", "bin", "ffprobe.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""), "ffmpeg", "bin", "ffprobe.exe"),
    ]
    for path in common_dirs:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError("ffprobe not found alongside ffmpeg or in PATH")


def _ensure_video_stream(video_path):
    ffprobe = _find_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed: {stderr or 'unknown error'}")
    if not (result.stdout or "").strip():
        raise RuntimeError("missing video stream")


def _save_frames(video_path, output_dir, fps):
    _ensure_video_stream(video_path)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(video_fps / fps), 1)
    frame_id, saved = 0, 0
    frame_paths = []
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
            img_path = os.path.join(output_dir, f"{saved+1}.jpg")
            cv2.imwrite(img_path, frame)
            frame_paths.append(os.path.abspath(img_path))
            saved += 1
        frame_id += 1

    cap.release()
    return frame_paths


def step4_chouzhen():
    """Step 4: 抽帧 + Whisper 转录，保存至 CHOUZHEN_JSON。已存在的 id 直接保留，不重复处理。"""
    print("\n" + "═"*60)
    print("  STEP 4：视频抽帧 + 音频转录")
    print("═"*60)

    # 提前定位 ffmpeg，同步告知 pydub，消除 RuntimeWarning
    ffmpeg_exe = _find_ffmpeg()
    print(f"  [ffmpeg] 使用路径：{ffmpeg_exe}")

    model = whisper.load_model(WHISPER_MODEL_NAME)

    with open(VIDEO_INTRO_JSON, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        if isinstance(original_data, dict):
            original_data = [original_data]
    # 以文件名（id.mp4）为 key 建立映射
    label_map = {f"{item['id']}.mp4": item for item in original_data}
    # 同时支持以 video_url 为 key
    for item in original_data:
        label_map[item.get("video_url", "")] = item

    # 加载已有输出，key = id 字符串
    existing: dict = load_existing_output(CHOUZHEN_JSON)
    if existing:
        print(f"  检测到已有输出文件，共 {len(existing)} 条记录，已存在的 id 将直接保留。")

    os.makedirs(IMAGE_DIR, exist_ok=True)
    videos = original_data
    print(f"  共发现 {len(videos)} 个视频文件")

    new_items: dict = {}
    skipped, added, failed = 0, 0, 0

    for video_file in tqdm(videos, desc="📦 处理视频", unit="个"):
        video_name = str(video_file.get("id", "")).strip()
        if not video_name:
            continue

        # ── 已存在则跳过 ──
        if video_name in existing:
            existing_item = existing[video_name]
            assign_record_value(existing_item, "video_url", video_file.get("video_url", ""), force=True)
            assign_record_value(existing_item, "video_introduction", video_file.get("video_introduction", ""), force=True)
            assign_record_value(existing_item, "label", video_file.get("label", ""), force=True)
            resolved_existing_path = resolve_video_path(video_file)
            if resolved_existing_path and os.path.isfile(resolved_existing_path):
                assign_record_value(existing_item, "video_path", to_repo_relative(resolved_existing_path), force=True)
            existing_image_root = resolve_image_root(existing_item)
            if existing_image_root and os.path.isdir(existing_image_root):
                assign_record_value(existing_item, "image_root", to_repo_relative(existing_image_root), force=True)
            tqdm.write(f"  [跳过] ID={video_name} 已存在，保留原内容。")
            skipped += 1
            continue

        try:
            video_path = resolve_video_path(video_file)
            if not video_path or not os.path.isfile(video_path):
                tqdm.write(f"  [失败] ID={video_name} 未找到本地视频：{video_path or 'N/A'}")
                failed += 1
                continue

            image_root = get_image_output_root(video_name, video_file.get("label", ""))
            frame_dir = os.path.join(image_root, "frames")

            main_frames = _save_frames(video_path, frame_dir, FRAME_FPS)

            audio_path = _extract_audio(video_path)
            try:
                full_transcript = _transcribe(audio_path, model)
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

            os.makedirs(image_root, exist_ok=True)
            txt_path = os.path.join(image_root, "transcription.txt")
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(full_transcript)

            new_items[video_name] = {
                "id": video_name,
                "video_url": video_file.get("video_url", f"https://www.douyin.com/video/{video_name}"),
                "video_path": video_file.get("video_path") or to_repo_relative(video_path),
                "video_introduction": video_file.get("video_introduction", ""),
                "label": video_file.get("label", ""),
                "image_root": to_repo_relative(image_root),
                "image": main_frames,
                "all_transcription": full_transcript
            }
            added += 1
        except Exception as e:
            failed += 1
            tqdm.write(f"  [失败] ID={video_name} 抽帧/转录异常：{e}")

    # ── 合并旧数据 + 新数据，按 id 排序写出 ──
    id_to_meta = {str(v.get("id", "")).strip(): v for v in original_data}
    for vid, record in existing.items():
        src = id_to_meta.get(vid)
        if not src:
            continue
        assign_record_value(record, "video_url", src.get("video_url", ""), force=True)
        assign_record_value(record, "video_introduction", src.get("video_introduction", ""), force=True)
        assign_record_value(record, "label", src.get("label", ""), force=True)
        resolved_existing_path = resolve_video_path(src)
        if resolved_existing_path and os.path.isfile(resolved_existing_path):
            assign_record_value(record, "video_path", to_repo_relative(resolved_existing_path), force=True)
        existing_image_root = resolve_image_root(record)
        if existing_image_root and os.path.isdir(existing_image_root):
            assign_record_value(record, "image_root", to_repo_relative(existing_image_root), force=True)

    merged = {**existing, **new_items}
    result_json = sort_records_by_id(list(merged.values()))
    dump_json_atomic(CHOUZHEN_JSON, result_json)

    print(f"\n  新增：{added} 条 | 跳过（已存在）：{skipped} 条 | 失败：{failed} 条 | 文件总计：{len(result_json)} 条")
    print(f"  ✅ Step 4 完成 → {CHOUZHEN_JSON}")


# ════════════════════════════════════════════════════════════════
#  STEP 5: Ollama 本地模型生成视频描述
# ════════════════════════════════════════════════════════════════

def _detect_language(texts: list) -> str:
    combined = " ".join(texts)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', combined)
    return "zh" if len(chinese_chars) / max(len(combined), 1) > 0.2 else "en"


def _call_ollama_with_images(transcription, video_intro, frames, lang="zh", max_retries=3):
    """调用本地 Ollama 多模态模型，直接传入本地图像路径列表，逐批生成描述。"""
    full_description = ""

    if lang == "zh":
        system_prompt = (
            "你是一位视频内容叙述专家，你的任务是根据视频的关键帧图像和音频转录内容，"
            "用中文写出一段完整的故事性描述，帮助没有看过视频的读者完全理解视频讲了什么。"
            "你的描述应自然流畅、像讲故事一样，结合画面和声音的信息，真实、细腻地呈现"
            "视频中的人物、动作、场景、情节发展和情绪变化，使读者仿佛亲眼看过这个视频一样。"
        )
        text_template = (
            "以下是该视频的简介、音频转录文本和部分关键帧图像（第 {batch_idx} 批）：\n\n"
            "视频简介：{video_intro}\n\n"
            "音频转录文本：{transcription}\n\n"
            "每张图像的文件名数值越小表示越靠近视频开头。"
            "请结合图像和音频，写出自然连贯、像讲故事一样的视频内容叙述。"
        )
    else:
        system_prompt = (
            "You are a video content narration expert. Your task is to describe the video story "
            "based on the key frame images and the audio transcription. Write a complete story-like "
            "video description in English, combining the visual and audio content, so that someone "
            "who hasn't seen the video can fully understand what it is about. Be vivid, coherent, "
            "and realistic in describing the characters, actions, scenes, plot, and emotional development."
        )
        text_template = (
            "Below is the video's introduction, audio transcript, and some keyframe images (batch {batch_idx}):\n\n"
            "Video introduction: {video_intro}\n\n"
            "Audio transcript: {transcription}\n\n"
            "The smaller the image filename number, the earlier it appears in the video. "
            "Please write a natural, coherent, story-like description of the video content."
        )

    for batch_idx in range(0, len(frames), MAX_IMAGES_PER_BATCH):
        image_batch = frames[batch_idx:batch_idx + MAX_IMAGES_PER_BATCH]
        # 过滤不存在的文件
        valid_images = [p for p in image_batch if os.path.isfile(p)]

        if not valid_images:
            print(f"⚠️ 批次 {batch_idx // MAX_IMAGES_PER_BATCH + 1} 无有效图像，跳过。")
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": text_template.format(
                    batch_idx=batch_idx // MAX_IMAGES_PER_BATCH + 1,
                    video_intro=video_intro,
                    transcription=transcription,
                ),
                "images": valid_images,   # Ollama 直接接受本地路径列表
            },
        ]

        for attempt in range(1, max_retries + 1):
            try:
                response = OLLAMA_CLIENT.chat(model=OLLAMA_MODEL, messages=messages)
                full_description += response.message.content.strip() + "\n"
                break
            except Exception as e:
                print(f"⚠️ Ollama 调用失败（批次 {batch_idx // MAX_IMAGES_PER_BATCH + 1}，"
                      f"第 {attempt}/{max_retries} 次）: {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    print("⏭️ 已达最大重试次数，跳过此批次。")

    return full_description.strip()


def step5_generate_descriptions():
    """Step 5: 调用本地 Ollama 模型生成每个视频的描述，保存至 VIDEO_DESCRIPTION_JSON。已存在的 id 直接保留。"""
    print("\n" + "═"*60)
    print(f"  STEP 5：Ollama（{OLLAMA_MODEL}）生成视频描述")
    print("═"*60)

    with open(CHOUZHEN_JSON, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # 加载已有输出，key = id 字符串
    existing: dict = load_existing_output(VIDEO_DESCRIPTION_JSON)
    if existing:
        print(f"  检测到已有输出文件，共 {len(existing)} 条记录，已存在的 id 将直接保留。")

    new_items: dict = {}
    skipped, added = 0, 0

    for video in tqdm(input_data, desc="Processing videos"):
        video_id      = str(video.get("id", "")).strip()
        transcription = video.get("all_transcription", "")
        video_intro   = video.get("video_introduction", "No introduction provided.")

        # ── 已存在则跳过 ──
        if video_id in existing:
            tqdm.write(f"  [跳过] ID={video_id} 已存在，保留原内容。")
            skipped += 1
            continue

        lang = _detect_language([transcription, video_intro])

        all_frames = resolve_frame_paths(video)
        frames = all_frames[::FRAME_INTERVAL]

        if not frames:
            tqdm.write(f"⚠️ 视频 {video_id} 无有效帧，跳过。")
            continue

        description = _call_ollama_with_images(transcription, video_intro, frames, lang=lang)
        if not description:
            tqdm.write(f"⚠️ 视频 {video_id} 描述为空，跳过。")
            continue

        new_items[video_id] = {
            "id":                 video_id,
            "video_url":          video.get("video_url", f"https://www.douyin.com/video/{video_id}"),
            "video_path":         video.get("video_path", ""),
            "video_introduction": video_intro,
            "label":              video.get("label", ""),
            "all_transcription":  transcription,
            "video_description":  description,
        }
        added += 1

    # ── 合并旧数据 + 新数据，按 id 排序写出 ──
    id_to_input = {str(v.get("id", "")).strip(): v for v in input_data}
    for vid, record in existing.items():
        src = id_to_input.get(vid)
        if not src:
            continue
        assign_record_value(record, "video_url", src.get("video_url", ""), force=True)
        assign_record_value(record, "video_path", src.get("video_path", ""), force=True)
        assign_record_value(record, "video_introduction", src.get("video_introduction", ""), force=True)
        assign_record_value(record, "label", src.get("label", ""), force=True)
        assign_record_value(record, "all_transcription", src.get("all_transcription", ""))

    merged = {**existing, **new_items}
    output_data = sort_records_by_id(list(merged.values()))
    dump_json_atomic(VIDEO_DESCRIPTION_JSON, output_data)

    print(f"\n  新增：{added} 条 | 跳过（已存在）：{skipped} 条 | 文件总计：{len(output_data)} 条")
    print(f"  ✅ Step 5 完成 → {VIDEO_DESCRIPTION_JSON}")




# ════════════════════════════════════════════════════════════════
#  STEP 6: C_label 自动标注（评论 × 视频描述 → douyin_sample.json）
# ════════════════════════════════════════════════════════════════

# ── 6a. 文本工具 ──────────────────────────────────────────────────
import math as _math
from collections import defaultdict as _defaultdict

def _s6_clean(text):
    text = re.sub(r"\[[\w\u4e00-\u9fff]+\]", " ", text)
    text = re.sub(r"[#@][\w\u4e00-\u9fff🐱🤪]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _s6_tokenize(text):
    text = _s6_clean(text)
    tokens = [text[i:i+2] for i in range(len(text) - 1)]
    tokens += [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    return tokens

def _s6_build_idf(corpus):
    df = _defaultdict(int)
    N  = len(corpus)
    for doc in corpus:
        for tok in set(doc):
            df[tok] += 1
    return {tok: _math.log((N+1)/(cnt+1)) + 1.0 for tok, cnt in df.items()}

def _s6_tfidf(tokens, idf):
    if not tokens:
        return {}
    tf = Counter(tokens)
    n  = len(tokens)
    return {tok: (cnt/n) * idf.get(tok, 1.0) for tok, cnt in tf.items()}

def _s6_cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0)*v for k, v in b.items())
    ma  = _math.sqrt(sum(v*v for v in a.values()))
    mb  = _math.sqrt(sum(v*v for v in b.values()))
    return dot/(ma*mb) if ma and mb else 0.0

# ── 6b. 情绪检测 ──────────────────────────────────────────────────
_S6_EMOTION_RULES = [
    (r"\[泪奔\]|\[哭\]|\[流泪\]|\[泣不成声\]|\[大哭\]",                              "deep_empathy"),
    (r"\[发怒\]|\[愤怒\]|\[鄙视\]",                                                    "anger"),
    (r"\[捂脸\]|\[尬笑\]|\[黑脸\]|\[白眼\]",                                        "speechless"),
    (r"哈哈|笑死|笑发财|笑喷|颠|太逗|太好笑|搞笑|好笑|绝了|\[大笑\]|\[呲牙\]|\[憨笑\]","humor"),
    (r"\[赞\]|\[鼓掌\]|好看|牛啊|厉害|超棒|真棒|666|绝绝子",                             "admiration"),
    (r"\[玫瑰\]|\[心\]|\[比心\]|\[爱心\]|可爱|萌|爱了",                             "affection"),
    (r"原来|宁可|结果|所以说|说白了|分明|这哪|不过是|罢了|才发现",                           "irony"),
    (r"有没有|请问|为什么|怎么|吗[？?]|啊[？?]",                                             "curiosity"),
    (r"\[耶\]|\[微笑\]|不错|还行|挺好",                                                  "mild_positive"),
]

def _s6_detect_emotion(text):
    for pattern, label in _S6_EMOTION_RULES:
        if re.search(pattern, text):
            return label
    return "empty" if not text.strip() else "neutral"

# ── 6c. 关键词规则 ────────────────────────────────────────────────
_S6_KW_RULES = [
    (1, r"(电影|风格|国家).*(电影|风格|国家)|招队友|还差.*位|已有.*位",                    "Rhyming"),
    (1, r"=|谐音|其实是|读作|[\u4e00-\u9fff]{1,4}[=＝][\u4e00-\u9fff]{1,4}|你有.*了么", "Puns (Homophones)"),
    (2, r"第[二三四五六七八九十\d]天|下[集章节]|上[集章节]|霸总|总裁|女主|男主|男二|女二"
        r"|医[生院][:：]|[她他][:：]|扣\s?[1一]|集合|来[报名演]|有没有想",              "Content Extraction"),
    (2, r"原来我|原来是|竟然|居然|真的在想|我脑子|代入|以前.*现在"
        r"|幻想|临死前|你知道的|没想到|怪不得|难怪|这不就是|这不是.*吗",                 "Meme Application"),
    (3, r"这才是.*死法|有本事.*翻拍|宁可.*都不|服了|麻了|绷不住|破防|算了|随便|不是吧|就这","Sarcasm (Irony)"),
    (4, r"哈哈|笑死|笑发财|颠|太逗|好玩|[来快]来|一起|冲|开冲",                          "Plain Humor"),
]

def _s6_kw_label(text):
    for _, pattern, label in sorted(_S6_KW_RULES, key=lambda x: x[0]):
        if re.search(pattern, text):
            return label
    return None

# ── 6d. KNN索引 ───────────────────────────────────────────────────
class _S6TrainingIndex:
    def __init__(self, records, idf):
        self.entries = []
        self.prior   = _defaultdict(Counter)
        for item in records:
            vlabel = item.get("label", "").strip()
            for i in range(1, 6):
                comment = (item.get(f"comment_{i}") or "").strip()
                clabel  = (item.get(f"C{i}_label")  or "").strip()
                if comment and clabel:
                    vec = _s6_tfidf(_s6_tokenize(comment), idf)
                    self.entries.append((vec, clabel, vlabel))
                    self.prior[vlabel][clabel] += 1

    def predict(self, vec, video_label="", k=5):
        if not self.entries:
            return "Plain Humor"
        scored = sorted((_s6_cosine(vec, e[0]), e[1]) for e in self.entries)
        scored.reverse()
        top = scored[:k]
        if top[0][0] < 0.05 and video_label in self.prior:
            return self.prior[video_label].most_common(1)[0][0]
        return Counter(c for _, c in top).most_common(1)[0][0]

_S6_EMOTION_TO_LABEL = {
    "deep_empathy":  "Meme Application",
    "speechless":    "Meme Application",
    "irony":         "Sarcasm (Irony)",
    "anger":         "Sarcasm (Irony)",
    "humor":         "Plain Humor",
    "admiration":    "Plain Humor",
    "affection":     "Plain Humor",
    "mild_positive": "Plain Humor",
    "curiosity":     "Content Extraction",
}

def _s6_predict_clabel(comment, video_desc, video_label, idf, index):
    comment = (comment or "").strip()
    if not comment:
        return ""
    label = _s6_kw_label(comment)
    if label:
        return label
    comment_vec = _s6_tfidf(_s6_tokenize(comment), idf)
    sim = _s6_cosine(comment_vec, _s6_tfidf(_s6_tokenize(video_desc), idf))
    if sim >= 0.10:
        return "Content Extraction"
    emotion = _s6_detect_emotion(comment)
    if emotion in _S6_EMOTION_TO_LABEL:
        return _S6_EMOTION_TO_LABEL[emotion]
    return index.predict(comment_vec, video_label)

# ── 6e. 主函数 ────────────────────────────────────────────────────
def step6_label_comments():
    """Step 6: 对 top5 评论进行 C_label 标注，合并视频描述，写出 douyin_sample.json。
    已存在于输出文件中的 id 直接保留，不重复处理。同时检测并报告重复 id。"""
    print("\n" + "═"*60)
    print("  STEP 6：评论 C_label 自动标注")
    print("═"*60)

    # ── 载入三个输入文件 ──────────────────────────────────────────
    with open(SAMPLE_TRAIN_JSON, "r", encoding="utf-8") as f:
        train_records = json.load(f)

    with open(VIDEO_DESCRIPTION_JSON, "r", encoding="utf-8") as f:
        desc_records = json.load(f)

    with open(TOP5_COMMENTS_JSON, "r", encoding="utf-8") as f:
        top5_raw = json.load(f)
    # top5 可能是 list 或 dict
    top5_records = list(top5_raw.values()) if isinstance(top5_raw, dict) else top5_raw

    print(f"  训练集 {len(train_records)} 条 | "
          f"视频描述 {len(desc_records)} 条 | "
          f"Top5评论 {len(top5_records)} 条")

    # ── 重复 id 检测（三个输入文件各自内部 + 描述与评论跨文件） ──
    def _check_dup(records, name):
        ids = [str(r.get("id","")) for r in records]
        seen, dups = set(), set()
        for i in ids:
            if i in seen:
                dups.add(i)
            seen.add(i)
        if dups:
            print(f"  ⚠️  [{name}] 内部重复 id（共 {len(dups)} 个）：{sorted(dups)}")
        else:
            print(f"  ✅  [{name}] 无重复 id")
        return seen, dups

    desc_ids,  desc_dups  = _check_dup(desc_records,  "douyin_video_description")
    top5_ids,  top5_dups  = _check_dup(top5_records,  "douyin_top5_comments")
    train_ids, train_dups = _check_dup(train_records, "douyin_comments_sample (训练集)")

    # 描述文件与评论文件的 id 对齐检查
    only_in_desc = desc_ids - top5_ids
    only_in_top5 = top5_ids - desc_ids
    if only_in_desc:
        print(f"  ⚠️  仅在视频描述中存在（无对应评论）的 id：{sorted(only_in_desc)}")
    if only_in_top5:
        print(f"  ⚠️  仅在评论文件中存在（无对应描述）的 id：{sorted(only_in_top5)}")
    if not only_in_desc and not only_in_top5:
        print(f"  ✅  视频描述与评论文件 id 完全对齐")

    # ── 加载已有输出，已存在的 id 直接保留 ──────────────────────
    existing: dict = load_existing_output(DOUYIN_SAMPLE_JSON)
    if existing:
        print(f"\n  检测到已有输出文件 {DOUYIN_SAMPLE_JSON}")
        print(f"  共 {len(existing)} 条记录，已存在的 id 将直接保留，不重复处理。")
        # 检测已有输出内部是否有重复 id（理论上不应有，但做保险检查）
        out_ids = list(existing.keys())
        out_dup = {i for i in out_ids if out_ids.count(i) > 1}
        if out_dup:
            print(f"  ⚠️  输出文件中存在重复 id（共 {len(out_dup)} 个）：{sorted(out_dup)}")
            print(f"      将以最后出现的记录为准，建议手动排查。")

    # ── 构建 TF-IDF + KNN 索引 ───────────────────────────────────
    print("\n  构建 TF-IDF + KNN 索引...")
    corpus = []
    for item in train_records:
        corpus.append(_s6_tokenize(item.get("video_description", "")))
        for i in range(1, 6):
            corpus.append(_s6_tokenize(item.get(f"comment_{i}") or ""))
    for item in desc_records:
        corpus.append(_s6_tokenize(item.get("video_description", "")))

    idf   = _s6_build_idf(corpus)
    index = _S6TrainingIndex(train_records, idf)
    print(f"  词表 {len(idf):,} tokens | 训练样本 {len(index.entries)} 条评论")

    # ── 逐条预测 ─────────────────────────────────────────────────
    id_to_top5  = {str(r.get("id","")): r for r in top5_records}
    new_items   = {}
    all_clabels = []
    skipped, added = 0, 0

    for vd in tqdm(desc_records, desc="标注评论"):
        vid_id      = str(vd.get("id", "")).strip()
        video_url   = vd.get("video_url", "")
        video_intro = vd.get("video_introduction", "")
        video_label = vd.get("label", "").strip()
        video_desc  = vd.get("video_description", "")

        # 已存在则跳过
        if vid_id in existing:
            tqdm.write(f"  [跳过] ID={vid_id} 已存在，保留原内容。")
            skipped += 1
            continue

        top5 = id_to_top5.get(vid_id)
        if top5 is None:
            tqdm.write(f"  ⚠️  ID={vid_id} 在 top5 评论文件中找不到对应记录，评论将为空。")

        record = {
            "id":                 vid_id,
            "video_url":          video_url,
            "video_path":         vd.get("video_path", ""),
            "video_introduction": video_intro,
        }
        for i in range(1, 6):
            text   = ((top5.get(f"comment_{i}") or "") if top5 else "").strip()
            clabel = _s6_predict_clabel(text, video_desc, video_label, idf, index)
            record[f"comment_{i}"] = text
            record[f"C{i}_label"]  = clabel
            if clabel:
                all_clabels.append(clabel)

        record["label"]             = video_label
        record["video_description"] = video_desc
        new_items[vid_id] = record
        added += 1

    # ── 合并旧数据 + 新数据，按 id 排序写出 ─────────────────────
    merged = {**existing, **new_items}
    id_to_desc = {str(v.get("id", "")).strip(): v for v in desc_records}
    for vid, record in existing.items():
        src = id_to_desc.get(vid)
        if not src:
            continue
        assign_record_value(record, "video_url", src.get("video_url", ""), force=True)
        assign_record_value(record, "video_path", src.get("video_path", ""), force=True)
        assign_record_value(record, "video_introduction", src.get("video_introduction", ""), force=True)
        assign_record_value(record, "label", src.get("label", ""), force=True)
        assign_record_value(record, "video_description", src.get("video_description", ""), force=True)

    merged = {**existing, **new_items}
    output_list = sort_records_by_id(list(merged.values()))
    dump_json_atomic(DOUYIN_SAMPLE_JSON, output_list)

    print(f"\n  新增：{added} 条 | 跳过（已存在）：{skipped} 条 | 文件总计：{len(output_list)} 条")
    print(f"  ✅ Step 6 完成 → {DOUYIN_SAMPLE_JSON}")

    if all_clabels:
        total = len(all_clabels)
        print("\n  ── 本次新增标签分布 " + "─"*30)
        for label, cnt in Counter(all_clabels).most_common():
            bar = "█" * round(cnt/total*20)
            print(f"    {label:25s} {cnt:3d}  {bar}")

# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="抖音视频全流程一键脚本",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--steps", type=str, default="1,2,3,4,5,6",
        help="指定要执行的步骤（逗号分隔），例如 --steps 2,3,4\n"
             "  1: 采集视频 URL\n"
             "  2: 下载视频 + 获取简介\n"
             "  3: 抓取评论\n"
             "  4: 抽帧 + 音频转录\n"
             "  5: Ollama 生成视频描述\n"
             "  6: C_label 自动标注\n"
             "（默认全部执行）"
    )
    parser.add_argument(
        "--auto-label-target",
        type=int,
        default=0,
        help="Step 1 批量模式：自动抓取 5 个预设 Label，每类至少目标 N 条 URL。"
    )
    parser.add_argument(
        "--comment-label-target",
        type=int,
        default=COMMENT_LABEL_TARGET,
        help="Step 3 评论抓取：>0 时限制每个 label 最多抓 N 条；<=0 时按来源文件全量补抓。"
    )
    args = parser.parse_args()

    steps = set()
    for s in args.steps.split(","):
        s = s.strip()
        if s.isdigit():
            steps.add(int(s))

    print("\n" + "★"*60)
    print("  抖音视频全流程 Pipeline")
    print(f"  将执行步骤：{sorted(steps)}")
    print("★"*60)

    if 1 in steps:
        if args.auto_label_target > 0:
            step1_batch_collect_all_labels(args.auto_label_target)
        else:
            step1_crawl_urls()

    if 2 in steps:
        asyncio.run(step2_download_videos())

    if 3 in steps:
        step3_download_comments(args.comment_label_target)

    if 4 in steps:
        step4_chouzhen()

    if 5 in steps:
        step5_generate_descriptions()

    if 6 in steps:
        step6_label_comments()

    print("\n\n" + "★"*60)
    print("  ✅ 所有指定步骤已完成！")
    print("★"*60)


if __name__ == "__main__":
    main()
