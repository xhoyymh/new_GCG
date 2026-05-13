"""
YouTube 视频数据采集与描述生成 — 一体化流水线
=================================================
流程：
  Step 1  搜索 YouTube Shorts URL 并保存
  Step 2  下载视频
  Step 3  抓取评论
  Step 4  抽帧 + Whisper 语音转录
  Step 5  调用 Ollama 多模态模型生成视频描述
  Step 6  C_label 自动标注

依赖：
  pip install google-api-python-client isodate yt-dlp opencv-python
              openai-whisper pydub tqdm ollama
  ffmpeg 需在系统 PATH 中或通过 FFMPEG_PATH 指定
"""

import argparse
import json
import math as _math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict as _defaultdict
from urllib.parse import parse_qs, urlparse

import cv2
import isodate
import whisper
import yt_dlp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from ollama import Client as OllamaClient
from tqdm import tqdm


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

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════
#  全局配置（按需修改）
# ═══════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
BASE_DIR = os.path.join(REPO_ROOT, "data_pre")

API_KEY = os.environ.get("YOUTUBE_API_KEY", "AIzaSyAp0cKrDn6M3--UQaSHlfJF1UcGfanWsug")

JSON_DIR = os.path.join(BASE_DIR, "json", "youtube", "data_pre")
VIDEO_DIR = os.path.join(BASE_DIR, "video", "youtube")
IMAGE_DIR = os.path.join(BASE_DIR, "youtube_image")

# 各步骤产出 JSON 路径
URL_JSON = os.path.join(JSON_DIR, "youtube_video_url.json")
VIDEO_INTRO_JSON = os.path.join(JSON_DIR, "youtube_video_introduction.json")
VIDEO_SAMPLE_JSON = os.path.join(SCRIPT_DIR, "youtube_video_sample.json")
TOP5_COMMENT_JSON = os.path.join(JSON_DIR, "youtube_top5_comment.json")
ALL_COMMENT_JSON = os.path.join(JSON_DIR, "youtube_all_comments.json")
CHOUZHEN_JSON = os.path.join(JSON_DIR, "youtube_chouzhen.json")
DESCRIPTION_JSON = os.path.join(JSON_DIR, "youtube_video_description.json")

# Step 5 输出 / Step 6 输入输出路径
SAMPLE_TRAIN_JSON = os.path.join(BASE_DIR, "json", "sample", "youtube_comments_sample.json")
YOUTUBE_SAMPLE_JSON = os.path.join(BASE_DIR, "json", "youtube", "sample", "youtube_sample.json")

# 抽帧 / Whisper / Ollama 参数
FRAMES_PER_SECOND = 1
WHISPER_MODEL_NAME = "base"
OLLAMA_MODEL = "qwen3.5:9b"
MAX_IMAGES_PER_BATCH = 5
FRAME_INTERVAL = 1
DEFAULT_BATCH_TARGET = 200
YOUTUBE_API_QUOTA_EXCEEDED = False
COMMENT_CAPTURE_MODE = "full_source"
COMMENT_CAPTURE_CAP = 50
COMMENT_LABEL_TARGET = 0
COMMENT_TOP_K = 5
COMMENT_API_PAGE_SIZE = 50
COMMENT_API_MAX_PAGES = 1


def normalize_comment_label_target(label_target: int | None = None) -> int:
    if label_target is None:
        label_target = COMMENT_LABEL_TARGET
    try:
        value = int(label_target)
    except (TypeError, ValueError):
        value = COMMENT_LABEL_TARGET
    return value if value > 0 else 0


def get_comment_capture_mode(label_target: int | None = None) -> str:
    return "balanced" if normalize_comment_label_target(label_target) > 0 else COMMENT_CAPTURE_MODE


def describe_comment_collection_mode(label_target: int | None = None) -> str:
    normalized_target = normalize_comment_label_target(label_target)
    if normalized_target > 0:
        return f"按标签上限模式（每个 label 最多 {normalized_target} 条）"
    return "全量模式"

# ffmpeg 路径（留空则自动从系统 PATH 中搜索；找不到时会报错提示）
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

LABEL_AUTO_TAGS_MAP = {
    "Comedy Skits": ["comedy skits", "sketch comedy"],
    "Daily Life Jokes": ["daily life jokes", "relatable humor"],
    "Funny Animal Videos": ["funny animal videos", "funny pets"],
    "Humorous Commentary": ["humorous commentary", "funny commentary"],
    "Talk Shows / Stand-Up Comedy / Cross-Talk": ["stand up comedy", "talk show comedy"],
}

LABEL_SLUG_MAP = {
    "Comedy Skits": "comedy_skits",
    "Daily Life Jokes": "daily_life_jokes",
    "Funny Animal Videos": "funny_animal_videos",
    "Humorous Commentary": "humorous_commentary",
    "Talk Shows / Stand-Up Comedy / Cross-Talk": "talk_show_standup_crosstalk",
}

LABEL_SEARCH_PREFIX_MAP = {
    "Comedy Skits": "comedy skits",
    "Daily Life Jokes": "daily life jokes",
    "Funny Animal Videos": "funny animal videos",
    "Humorous Commentary": "humorous commentary",
    "Talk Shows / Stand-Up Comedy / Cross-Talk": "stand up comedy",
}

LABEL_QUERY_HINTS_MAP = {
    "Comedy Skits": ["funny skit", "situational comedy", "funny sketch", "short comedy", "plot twist skit", "parody skit", "sitcom clips"],
    "Daily Life Jokes": ["daily comedy", "relatable shorts", "school jokes", "family jokes", "relationship humor", "office humor", "daily routine comedy"],
    "Funny Animal Videos": ["funny cats", "funny dogs", "animal memes", "cute animals", "animal fails", "funny pet compilation", "funny wildlife"],
    "Humorous Commentary": ["satirical commentary", "comedy commentary", "funny reaction", "humorous review", "reaction commentary", "roast commentary"],
    "Talk Shows / Stand-Up Comedy / Cross-Talk": ["funny monologue", "comedy clips", "late night comedy", "cross talk comedy", "stand up clips", "crowd work comedy", "late night monologue"],
}

BATCH_COLLECTION_STAGES = [
    {"max_duration_sec": 60, "tag_limit": 2},
    {"max_duration_sec": 180, "tag_limit": 4},
    {"max_duration_sec": 300, "tag_limit": 8},
    {"max_duration_sec": 600, "tag_limit": 12},
    {"max_duration_sec": 1200, "tag_limit": 16},
    {"max_duration_sec": 1800, "tag_limit": 24},
    {"max_duration_sec": 3600, "tag_limit": 36},
]

# ── pydub / ffmpeg 初始化（保持在配置区末尾）──────────────────────
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", RuntimeWarning)
    from pydub import AudioSegment
if (not FFMPEG_PATH) or (not os.path.isfile(FFMPEG_PATH)):
    FFMPEG_PATH = shutil.which("ffmpeg") or FFMPEG_PATH
if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg = FFMPEG_PATH
    ffprobe_path = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    if os.path.isfile(ffprobe_path):
        AudioSegment.ffprobe = ffprobe_path
    _ffmpeg_bin_dir = os.path.dirname(FFMPEG_PATH)
    os.environ["PATH"] = _ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")


# ═══════════════════════════════════════════════════════════
#  共用工具
# ═══════════════════════════════════════════════════════════

def load_json_data(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)


def is_quota_exceeded_error(exc: Exception) -> bool:
    if isinstance(exc, HttpError):
        try:
            if exc.resp.status == 403 and "quota" in str(exc).lower():
                return True
        except Exception:
            pass
    message = str(exc).lower()
    return "quotaexceeded" in message or "exceeded your" in message


def mark_quota_exceeded() -> None:
    global YOUTUBE_API_QUOTA_EXCEEDED
    if not YOUTUBE_API_QUOTA_EXCEEDED:
        print("  [切换] YouTube Data API 配额已耗尽，后续改用 yt-dlp 兜底。")
    YOUTUBE_API_QUOTA_EXCEEDED = True


def find_nodejs_path() -> str:
    candidates = [
        os.environ.get("YT_DLP_NODE_PATH", "").strip(),
        os.path.join(os.path.dirname(sys.executable), "node.exe"),
        os.path.join(sys.prefix, "node.exe"),
        shutil.which("node"),
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "nodejs", "node.exe"),
    ]
    for candidate in dedupe_keep_order([path for path in candidates if path]):
        if os.path.isfile(candidate):
            return candidate
    return ""


def build_yt_dlp_options(get_comments: bool = False) -> dict:
    opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "ignoreerrors": True,
        "getcomments": get_comments,
        "extractor_args": {
            "youtube": {
                "comment_sort": ["top"],
            }
        },
    }
    node_path = find_nodejs_path()
    if node_path:
        opts["js_runtimes"] = {"node": {"path": node_path}}
        opts["remote_components"] = ["ejs:github"]
    if get_comments:
        opts["extractor_args"]["youtube"]["max_comments"] = [
            str(COMMENT_CAPTURE_CAP),
            str(COMMENT_CAPTURE_CAP),
            "0",
            "0",
            "1",
        ]
    return opts


def fetch_yt_dlp_info(video_url: str, get_comments: bool = False) -> dict:
    opts = build_yt_dlp_options(get_comments=get_comments)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return info or {}


def load_existing_output(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    try:
        data = load_json_data(filepath)
        if isinstance(data, list):
            return {str(item["id"]): item for item in data if isinstance(item, dict) and "id" in item}
    except Exception as e:
        print(f"  [警告] 读取已有输出文件失败：{e}，将视为空文件处理。")
    return {}


def dump_json_atomic(filepath: str, data) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=os.path.dirname(filepath), delete=False, suffix=".tmp") as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, filepath)


def sort_records_by_id(records: list) -> list:
    def _sort_key(item):
        try:
            return int(str(item.get("id", "")))
        except Exception:
            return float("inf")
    return sorted(records, key=_sort_key)


def dedupe_keep_order(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def assign_record_value(record: dict, key: str, value, force: bool = False) -> None:
    if value in (None, ""):
        return
    if force or not record.get(key):
        record[key] = value


def to_repo_relative(path: str) -> str:
    if not path:
        return ""
    if not os.path.isabs(path):
        return path
    try:
        return os.path.relpath(path, REPO_ROOT).replace("/", os.sep)
    except ValueError:
        return path


def to_repo_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def slugify_label(label: str) -> str:
    if label in LABEL_SLUG_MAP:
        return LABEL_SLUG_MAP[label]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "unknown"


def build_video_output_relpath(video_id, label: str) -> str:
    return os.path.join("data_pre", "video", "youtube", slugify_label(label), f"{video_id}.mp4")


def get_video_output_path(video_id, label: str) -> str:
    return os.path.join(REPO_ROOT, build_video_output_relpath(video_id, label))


def build_image_root_relpath(video_id, label: str) -> str:
    return os.path.join("data_pre", "youtube_image", slugify_label(label), str(video_id))


def get_image_output_root(video_id, label: str) -> str:
    return os.path.join(REPO_ROOT, build_image_root_relpath(video_id, label))


# ═══════════════════════════════════════════════════════════
#  Step 1  搜索 YouTube URL
# ═══════════════════════════════════════════════════════════

def extract_video_id(video_url: str) -> str:
    if not video_url:
        return ""
    if "shorts/" in video_url:
        return video_url.rstrip("/").split("/")[-1]
    parsed = urlparse(video_url)
    return parse_qs(parsed.query).get("v", [""])[0]


def build_seen_video_ids(records: list) -> set:
    seen = set()
    for item in records:
        video_id = extract_video_id(item.get("video_url", ""))
        if video_id:
            seen.add(video_id)
    return seen


def load_video_url_records() -> list:
    if not os.path.exists(URL_JSON):
        return []
    data = load_json_data(URL_JSON)
    return data if isinstance(data, list) else []


def save_video_url_records(records: list) -> None:
    dump_json_atomic(URL_JSON, sort_records_by_id(records))


def get_next_numeric_id(records: list) -> int:
    max_id = 0
    for item in records:
        try:
            max_id = max(max_id, int(str(item.get("id", 0))))
        except Exception:
            continue
    return max_id + 1


def count_records_for_label(records: list, label: str) -> int:
    return sum(1 for item in records if item.get("label", "") == label)

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
    print("│   N. 输入全新 Label（自定义）           │")
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
        print("  • 直接回车跳过，只使用自动 / 自定义 Tag")
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
        chosen.extend([t.strip() for t in custom_input.split(",") if t.strip()])

    result = dedupe_keep_order(chosen)
    if not result:
        print("  未选择任何手动 Tag。")
    return result


def clean_query_tag(tag: str) -> str:
    tag = (tag or "").strip().lstrip("#")
    tag = tag.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", tag).strip()


def normalize_search_query(label: str, tag: str, is_new_label: bool = False) -> str:
    query = clean_query_tag(tag)
    if not query:
        return ""
    search_prefix = LABEL_SEARCH_PREFIX_MAP.get(label, clean_query_tag(label))
    if not is_new_label and search_prefix and search_prefix.lower() not in query.lower():
        query = f"{search_prefix} {query}"
    if "shorts" not in query.lower():
        query = f"{query} shorts"
    return re.sub(r"\s+", " ", query).strip()


def build_label_query_candidates(data: list, label: str) -> list:
    auto_tags = LABEL_AUTO_TAGS_MAP.get(label, [])
    top_tags = [clean_query_tag(tag) for tag, _ in get_top_tags(data, label, top_n=30)]
    hint_tags = LABEL_QUERY_HINTS_MAP.get(label, [])
    candidates = []
    for tag in auto_tags + top_tags + hint_tags:
        cleaned = clean_query_tag(tag)
        if not cleaned:
            continue
        if cleaned.lower() in {"shorts", "short", "youtube", "ytshorts"}:
            continue
        candidates.append(cleaned)
    return dedupe_keep_order(candidates)


def _read_int_input(prompt: str, default_value: int) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default_value
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        print(f"  输入无效，使用默认值 {default_value}。")
        return default_value


def _format_duration(seconds: int) -> str:
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:d}:{remainder:02d}"


def _chunked(items: list, chunk_size: int):
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def search_ytdlp_videos(query: str, max_duration_sec: int, limit: int) -> list:
    search_size = min(max(limit * 3, 50), 500)
    opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
        "noplaylist": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{search_size}:{query}", download=False) or {}

    results = []
    seen = set()
    for entry in info.get("entries") or []:
        if not entry:
            continue
        video_id = entry.get("id")
        duration = entry.get("duration")
        if not video_id or video_id in seen or duration is None:
            continue
        if duration <= max_duration_sec:
            seen.add(video_id)
            results.append(video_id)
        if len(results) >= limit:
            break
    return results


def search_youtube_videos(youtube_client, query: str, max_duration_sec: int, limit: int, page_limit: int = 8) -> list:
    if limit <= 0:
        return []

    if YOUTUBE_API_QUOTA_EXCEEDED:
        return search_ytdlp_videos(query, max_duration_sec, limit)

    results = []
    seen = set()
    next_page_token = None
    pages_fetched = 0
    while len(results) < limit and pages_fetched < page_limit:
        pages_fetched += 1
        try:
            response = youtube_client.search().list(
                q=query,
                part="id",
                type="video",
                maxResults=50,
                pageToken=next_page_token,
            ).execute()
        except Exception as e:
            if is_quota_exceeded_error(e):
                mark_quota_exceeded()
                return search_ytdlp_videos(query, max_duration_sec, limit)
            raise
        video_ids = []
        for item in response.get("items", []):
            video_id = item.get("id", {}).get("videoId", "")
            if video_id and video_id not in seen:
                seen.add(video_id)
                video_ids.append(video_id)
        for batch_ids in _chunked(video_ids, 50):
            try:
                details = youtube_client.videos().list(id=",".join(batch_ids), part="contentDetails").execute()
            except Exception as e:
                if is_quota_exceeded_error(e):
                    mark_quota_exceeded()
                    merged = list(results)
                    for fallback_id in search_ytdlp_videos(query, max_duration_sec, limit):
                        if fallback_id not in merged:
                            merged.append(fallback_id)
                        if len(merged) >= limit:
                            break
                    return merged[:limit]
                raise
            for video in details.get("items", []):
                try:
                    duration = isodate.parse_duration(video["contentDetails"]["duration"]).total_seconds()
                except Exception:
                    continue
                if duration <= max_duration_sec:
                    results.append(video["id"])
                if len(results) >= limit:
                    return results
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return results


def collect_urls_for_label_target(label: str, data: list, target_count: int) -> int:
    records = load_video_url_records()
    current_count = count_records_for_label(records, label)
    if current_count >= target_count:
        print(f"  [跳过] Label={label} 已达到 {current_count}/{target_count}")
        return current_count

    query_candidates = build_label_query_candidates(data, label)
    if not query_candidates:
        print(f"  [警告] Label={label} 没有可用查询词，跳过。")
        return current_count

    seen_video_ids = build_seen_video_ids(records)
    youtube_client = build_youtube_client()

    print("\n" + "─" * 60)
    print(f"  Label={label} | 当前 {current_count}/{target_count}")
    print(f"  查询候选数：{len(query_candidates)}")
    print("─" * 60)

    for stage in BATCH_COLLECTION_STAGES:
        if current_count >= target_count:
            break
        stage_queries = query_candidates[: stage["tag_limit"]]
        stage_duration = stage["max_duration_sec"]
        print(f"\n  [阶段] Label={label} | 时长上限={_format_duration(stage_duration)} | 查询数={len(stage_queries)}")
        for raw_tag in stage_queries:
            if current_count >= target_count:
                break
            query = normalize_search_query(label, raw_tag)
            if not query:
                continue
            print(f"\n  [查询] Label={label} | Query={query} | 还需 {target_count - current_count} 条")
            try:
                ids = search_youtube_videos(youtube_client, query, stage_duration, target_count - current_count)
            except Exception as e:
                print(f"  [失败] Label={label} Query={query} 搜索异常：{e}")
                continue
            if not ids:
                print("  [结果] 本查询未命中符合条件的新视频。")
                continue
            next_id = get_next_numeric_id(records)
            added = 0
            for video_id in ids:
                if video_id in seen_video_ids:
                    continue
                records.append({"id": next_id, "label": label, "video_url": f"https://www.youtube.com/watch?v={video_id}"})
                seen_video_ids.add(video_id)
                next_id += 1
                current_count += 1
                added += 1
                if current_count >= target_count:
                    break
            save_video_url_records(records)
            print(f"  [进度] Label={label} 当前 {current_count}/{target_count}（本查询新增 {added}）")
        if current_count < target_count:
            print(f"  [阶段结束] Label={label} 当前仅 {current_count}/{target_count}，继续扩大 tag 或放宽时长")
    return current_count


def step1_batch_collect_all_labels(target_count: int = DEFAULT_BATCH_TARGET) -> None:
    if not os.path.exists(VIDEO_SAMPLE_JSON):
        raise SystemExit(f"样本文件不存在：{VIDEO_SAMPLE_JSON}")
    data = load_json_data(VIDEO_SAMPLE_JSON)
    labels = [label for label in get_labels(data) if label in LABEL_AUTO_TAGS_MAP]
    summary = {}
    print("\n" + "═" * 60)
    print(f"  Step 1（批量）— 五类 Label 自动采集，每类目标 {target_count} 条 URL")
    print("═" * 60)
    for label in labels:
        summary[label] = collect_urls_for_label_target(label, data, target_count)
    print("\n" + "=" * 60)
    print("  批量采集汇总")
    for label in labels:
        print(f"  {label}: {summary.get(label, 0)}/{target_count}")
    print(f"  输出文件：{URL_JSON}")
    print("=" * 60)


def step1_search_urls():
    print("\n" + "═" * 60)
    print("  Step 1 — 搜索 YouTube Shorts URL")
    print("═" * 60)
    os.makedirs(JSON_DIR, exist_ok=True)

    video_data = []
    if os.path.exists(VIDEO_SAMPLE_JSON):
        print(f"正在加载本地样本数据：{VIDEO_SAMPLE_JSON}")
        video_data = load_json_data(VIDEO_SAMPLE_JSON)
        print(f"  共加载 {len(video_data)} 条视频记录。")
    else:
        print("  [提示] 未找到本地样本文件，将改为手动输入。")

    if video_data:
        labels = get_labels(video_data)
        selected_label, is_new_label = select_label(labels)
        print(f"\n已选择 Label：{selected_label}（{'全新' if is_new_label else '已有'}）")
        top_tags = get_top_tags(video_data, selected_label) if not is_new_label else []
        manual_tags = select_tags(top_tags, is_new_label)
    else:
        selected_label = input("请输入 Label 名称：").strip() or "unknown"
        is_new_label = True
        tag_input = input("请输入搜索 Tag（多个用逗号分隔）：").strip()
        manual_tags = [t.strip() for t in tag_input.split(",") if t.strip()]

    auto_tags = [] if is_new_label else LABEL_AUTO_TAGS_MAP.get(selected_label, [])
    selected_tags = dedupe_keep_order(list(auto_tags) + manual_tags)
    if auto_tags:
        print(f'  [自动] Label "{selected_label}" → 搜索词 "{", ".join(auto_tags)}" 已加入 tag 列表')
    if not selected_tags:
        print("未选择任何 Tag，跳过 Step 1。")
        return

    max_minutes = _read_int_input("\n请输入最大时长（分钟，默认 5）：", 5)
    total_count = _read_int_input("请输入需要采集多少条 URL（总数，默认 30）：", 30)

    merged = sort_records_by_id(load_video_url_records())
    seen_video_ids = build_seen_video_ids(merged)
    next_id = get_next_numeric_id(merged)
    youtube_client = build_youtube_client()
    all_video_ids = []

    for tag in selected_tags:
        if len(all_video_ids) >= total_count:
            break
        query = normalize_search_query(selected_label, tag, is_new_label=is_new_label)
        print(f"\n正在搜索 '{query}'，时长 ≤ {max_minutes} 分钟，目标总数 {total_count} 条…")
        try:
            ids = search_youtube_videos(youtube_client, query, max_minutes * 60, total_count - len(all_video_ids))
        except Exception as e:
            print(f"  [失败] 查询异常：{e}")
            continue
        for video_id in ids:
            if video_id not in seen_video_ids and video_id not in all_video_ids:
                all_video_ids.append(video_id)

    if not all_video_ids:
        print("\n未找到任何符合条件的新视频。")
        return

    for video_id in all_video_ids:
        merged.append({"id": next_id, "label": selected_label, "video_url": f"https://www.youtube.com/watch?v={video_id}"})
        next_id += 1
    save_video_url_records(merged)
    print(f"\n✅ 已新增 {len(all_video_ids)} 条记录，保存至 {URL_JSON}")


# ═══════════════════════════════════════════════════════════
#  Step 2  下载视频
# ═══════════════════════════════════════════════════════════

def is_valid_media_file(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0


def resolve_video_path(record: dict) -> str:
    raw_path = record.get("video_path", "")
    if raw_path:
        resolved = to_repo_path(raw_path)
        if os.path.isfile(resolved):
            return resolved
    video_id = record.get("id", "")
    label = record.get("label", "")
    if video_id != "":
        new_path = get_video_output_path(video_id, label)
        if os.path.isfile(new_path):
            return new_path
        legacy_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if os.path.isfile(legacy_path):
            return legacy_path
        return new_path
    return ""


def load_video_intro_records() -> list:
    if not os.path.exists(VIDEO_INTRO_JSON):
        return []
    data = load_json_data(VIDEO_INTRO_JSON)
    return data if isinstance(data, list) else []


def save_video_intro_records(records: list) -> None:
    dump_json_atomic(VIDEO_INTRO_JSON, sort_records_by_id(records))


def download_video(url: str, output_path: str) -> bool:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ydl_opts = build_yt_dlp_options(get_comments=False)
    ydl_opts.update(
        {
            "skip_download": False,
            "format": "bv*+ba/b",
            "merge_output_format": "mp4",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": False,
            "overwrites": True,
        }
    )
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"  [失败] 下载异常：{url} | {e}")
        return False
    if not is_valid_media_file(output_path):
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass
        print(f"  [失败] 下载后文件无效：{output_path}")
        return False
    return True


def get_video_info(youtube_client, video_id: str, video_url: str = "") -> tuple:
    if not YOUTUBE_API_QUOTA_EXCEEDED:
        try:
            response = youtube_client.videos().list(part="snippet", id=video_id).execute()
            if response.get("items"):
                snippet = response["items"][0].get("snippet", {})
                return snippet.get("title", ""), snippet.get("description", "")
        except Exception as e:
            if is_quota_exceeded_error(e):
                mark_quota_exceeded()
            else:
                print(f"  [失败] 获取视频信息失败：{video_id} | {e}")
    fallback_url = video_url or f"https://www.youtube.com/watch?v={video_id}"
    try:
        info = fetch_yt_dlp_info(fallback_url, get_comments=False)
        return info.get("title", ""), info.get("description", "")
    except Exception as e:
        print(f"  [失败] yt-dlp 获取视频信息失败：{video_id} | {e}")
    return "", ""


def step2_download_videos():
    print("\n" + "═" * 60)
    print("  Step 2 — 下载视频（单条成功即 checkpoint）")
    print("═" * 60)

    if not os.path.exists(URL_JSON):
        print(f"  [错误] 未找到 URL 文件：{URL_JSON}，请先运行 Step 1。")
        return

    source_records = sort_records_by_id(load_video_url_records())
    if not source_records:
        print("  [提示] URL 文件为空，无需下载。")
        return

    youtube_client = build_youtube_client()
    output_map = {
        str(item["id"]): dict(item)
        for item in load_video_intro_records()
        if isinstance(item, dict) and "id" in item
    }
    skipped = added = failed = 0

    for source in tqdm(source_records, desc="下载视频"):
        record_id = str(source.get("id", "")).strip()
        if not record_id:
            failed += 1
            tqdm.write("  [失败] 遇到缺少 id 的 URL 记录，已跳过。")
            continue

        label = source.get("label", "")
        video_url = source.get("video_url", "")
        existing_record = output_map.get(record_id)
        if existing_record and is_valid_media_file(resolve_video_path({**source, **existing_record})):
            assign_record_value(existing_record, "label", label)
            assign_record_value(existing_record, "video_url", video_url, force=True)
            assign_record_value(existing_record, "video_path", to_repo_relative(resolve_video_path({**source, **existing_record})))
            output_map[record_id] = existing_record
            skipped += 1
            continue

        video_id = extract_video_id(video_url)
        if not video_id:
            failed += 1
            tqdm.write(f"  [失败] ID={record_id} 无法解析 video_id，已跳过。")
            continue

        title, api_description = get_video_info(youtube_client, video_id, video_url=video_url)
        if not title and not api_description:
            failed += 1
            tqdm.write(f"  [失败] ID={record_id} 无法获取 snippet，已跳过。")
            continue

        output_path = get_video_output_path(record_id, label)
        if not download_video(video_url, output_path):
            failed += 1
            continue

        record = dict(existing_record or {})
        record["id"] = source.get("id", record_id)
        record["label"] = label
        record["video_url"] = video_url
        record["video_introduction"] = title
        record["video_api_description"] = api_description
        record["video_path"] = to_repo_relative(output_path)
        output_map[record_id] = record
        save_video_intro_records(list(output_map.values()))
        added += 1

    if output_map:
        save_video_intro_records(list(output_map.values()))
    print(f"\n  新增下载：{added} | 已存在跳过：{skipped} | 失败跳过：{failed}")
    print(f"  ✅ Step 2 完成 → {VIDEO_INTRO_JSON}")


# ═══════════════════════════════════════════════════════════
#  Step 3  抓取评论
# ═══════════════════════════════════════════════════════════

def load_all_comments_output() -> dict:
    if not os.path.exists(ALL_COMMENT_JSON):
        return {}
    try:
        data = load_json_data(ALL_COMMENT_JSON)
    except Exception as e:
        print(f"  [警告] 读取已有评论详情失败：{e}，将视为空文件处理。")
        return {}
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    return {}


def sort_id_keys(keys) -> list:
    def _sort_key(value):
        try:
            return int(str(value))
        except Exception:
            return float("inf")
    return sorted(keys, key=_sort_key)


def save_comment_outputs(top5_map: dict, all_comments_map: dict) -> None:
    ordered_top5 = sort_records_by_id(list(top5_map.values()))
    dump_json_atomic(TOP5_COMMENT_JSON, ordered_top5)

    ordered_all = {}
    for key in sort_id_keys(set(all_comments_map) | {str(item.get("id", "")) for item in ordered_top5 if item.get("id", "") != ""}):
        ordered_all[str(key)] = normalize_comment_sample(all_comments_map.get(str(key), []))
    dump_json_atomic(ALL_COMMENT_JSON, ordered_all)


def normalize_comment_sample(comments: list) -> list:
    normalized = []
    for item in comments or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        like_value = item.get("likeCount", item.get("digg_count", 0))
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


def build_comment_record(
    video_data: dict,
    comments: list,
    existing_record: dict | None = None,
    comment_capture_mode: str | None = None,
) -> dict:
    record = dict(existing_record or {})
    record["id"] = video_data.get("id", record.get("id", ""))
    record["label"] = video_data.get("label", record.get("label", ""))
    record["video_url"] = video_data.get("video_url", record.get("video_url", ""))
    record["video_introduction"] = video_data.get("video_introduction", record.get("video_introduction", ""))
    record["video_api_description"] = video_data.get("video_api_description", record.get("video_api_description", ""))
    record["video_path"] = video_data.get("video_path", record.get("video_path", ""))
    record["comment_capture_mode"] = comment_capture_mode or COMMENT_CAPTURE_MODE
    record["comment_capture_cap"] = COMMENT_CAPTURE_CAP
    record["comment_capture_count"] = len(comments)
    for idx in range(1, COMMENT_TOP_K + 1):
        text = comments[idx - 1]["text"] if idx - 1 < len(comments) else ""
        record[f"comment_{idx}"] = text
        if text:
            record[f"C{idx}_label"] = record.get(f"C{idx}_label", "") or "待标注"
        else:
            record[f"C{idx}_label"] = ""
    return record


def normalize_existing_comment_outputs(
    top5_map: dict,
    all_comments_map: dict,
    comment_capture_mode: str | None = None,
) -> tuple[Counter, bool]:
    label_counts = Counter()
    changed = False
    for record_id, record in list(top5_map.items()):
        comments = normalize_comment_sample(all_comments_map.get(record_id, []))
        updated_record = build_comment_record(
            record,
            comments,
            existing_record=record,
            comment_capture_mode=comment_capture_mode,
        )
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


def get_top_comments(
    youtube_client,
    video_id: str,
    video_url: str = "",
    max_count: int = COMMENT_TOP_K,
    max_total: int = COMMENT_CAPTURE_CAP,
    max_api_pages: int = COMMENT_API_MAX_PAGES,
) -> tuple:
    if not YOUTUBE_API_QUOTA_EXCEEDED:
        comments = []
        next_page_token = None
        try_count = 0
        page_count = 0
        while True:
            try:
                response = youtube_client.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(COMMENT_API_PAGE_SIZE, max_total),
                    pageToken=next_page_token,
                    textFormat="plainText",
                    order="relevance",
                ).execute()
                for item in response.get("items", []):
                    snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                    comments.append({
                        "text": snippet.get("textDisplay", ""),
                        "likeCount": snippet.get("likeCount", 0),
                    })
                    if len(comments) >= max_total:
                        break
                next_page_token = response.get("nextPageToken")
                page_count += 1
                if not next_page_token or len(comments) >= max_total or page_count >= max_api_pages:
                    break
            except Exception as e:
                if is_quota_exceeded_error(e):
                    mark_quota_exceeded()
                    break
                try_count += 1
                if try_count >= 3:
                    raise RuntimeError(f"评论抓取失败：{video_id} | {e}") from e
                time.sleep(1)
        if comments:
            sample_comments = normalize_comment_sample(comments)
            top_comments = [
                {"text": item["text"], "likeCount": item["digg_count"]}
                for item in sample_comments[:max_count]
            ]
            all_comments = [
                {"text": item["text"], "likeCount": item["digg_count"]}
                for item in sample_comments
            ]
            return top_comments, all_comments

    fallback_url = video_url or f"https://www.youtube.com/watch?v={video_id}"
    try:
        info = fetch_yt_dlp_info(fallback_url, get_comments=True)
    except Exception as e:
        raise RuntimeError(f"yt-dlp 评论抓取失败：{video_id} | {e}") from e

    comments = []
    for item in info.get("comments") or []:
        text = (item.get("text") or item.get("text_display") or "").strip()
        if not text:
            continue
        comments.append({
            "text": text,
            "likeCount": item.get("like_count") or item.get("vote_count") or 0,
        })
        if len(comments) >= max_total:
            break
    sample_comments = normalize_comment_sample(comments)
    top_comments = [
        {"text": item["text"], "likeCount": item["digg_count"]}
        for item in sample_comments[:max_count]
    ]
    all_comments = [
        {"text": item["text"], "likeCount": item["digg_count"]}
        for item in sample_comments
    ]
    return top_comments, all_comments


def step3_fetch_comments(comment_label_target: int | None = None):
    normalized_label_target = normalize_comment_label_target(comment_label_target)
    comment_capture_mode = get_comment_capture_mode(normalized_label_target)
    mode_text = describe_comment_collection_mode(normalized_label_target)
    skip_reason_text = (
        "已存在/已达 label 上限"
        if normalized_label_target > 0
        else "已存在/无效记录"
    )
    print("\n" + "═" * 60)
    print(f"  Step 3 — 抓取评论（{mode_text}，单条成功即 checkpoint）")
    print("═" * 60)

    if not os.path.exists(VIDEO_INTRO_JSON):
        print(f"  [错误] 未找到视频信息文件：{VIDEO_INTRO_JSON}，请先运行 Step 2。")
        return

    input_records = sort_records_by_id(load_video_intro_records())
    if not input_records:
        print("  [提示] 视频信息文件为空，无需抓取评论。")
        return

    youtube_client = build_youtube_client()
    top5_map = load_existing_output(TOP5_COMMENT_JSON)
    all_comments_map = load_all_comments_output()
    label_counts, normalized_existing = normalize_existing_comment_outputs(
        top5_map,
        all_comments_map,
        comment_capture_mode=comment_capture_mode,
    )
    if normalized_existing:
        save_comment_outputs(top5_map, all_comments_map)
    skipped = added = failed = 0

    for video_data in tqdm(input_records, desc="抓取评论"):
        record_id = str(video_data.get("id", "")).strip()
        if not record_id:
            failed += 1
            continue

        label = video_data.get("label", "")
        existing_top5 = top5_map.get(record_id)
        if existing_top5 and record_id in all_comments_map:
            comments = normalize_comment_sample(all_comments_map.get(record_id, []))
            all_comments_map[record_id] = comments
            top5_map[record_id] = build_comment_record(
                video_data,
                comments,
                existing_record=existing_top5,
                comment_capture_mode=comment_capture_mode,
            )
            skipped += 1
            continue

        if normalized_label_target > 0 and label and label_counts.get(label, 0) >= normalized_label_target:
            skipped += 1
            continue

        video_url = video_data.get("video_url", "")
        video_id = extract_video_id(video_url)
        if not video_id:
            failed += 1
            tqdm.write(f"  [失败] ID={record_id} 无法解析评论对应的 video_id，已跳过。")
            continue

        try:
            top5_comments, all_comments = get_top_comments(
                youtube_client,
                video_id,
                video_url=video_url,
                max_count=COMMENT_TOP_K,
                max_total=COMMENT_CAPTURE_CAP,
                max_api_pages=COMMENT_API_MAX_PAGES,
            )
        except Exception as e:
            failed += 1
            tqdm.write(f"  [失败] ID={record_id} 抓取评论失败，已跳过。{e}")
            continue

        normalized_comments = normalize_comment_sample([
            {"text": item.get("text", ""), "digg_count": item.get("likeCount", 0)}
            for item in all_comments
        ])
        top5_map[record_id] = build_comment_record(
            video_data,
            normalized_comments,
            existing_record=existing_top5,
            comment_capture_mode=comment_capture_mode,
        )
        all_comments_map[record_id] = normalized_comments
        save_comment_outputs(top5_map, all_comments_map)
        added += 1
        if label:
            label_counts[label] += 1

    if top5_map or all_comments_map:
        save_comment_outputs(top5_map, all_comments_map)
    print(f"\n  新增评论：{added} | 跳过（{skip_reason_text}）：{skipped} | 失败跳过：{failed}")
    print(f"  ✅ Step 3 完成 → {TOP5_COMMENT_JSON}")
    print(f"  ✅ 评论详情 → {ALL_COMMENT_JSON}")


# ═══════════════════════════════════════════════════════════
#  Step 4  抽帧 + Whisper 转录
# ═══════════════════════════════════════════════════════════

def extract_audio(video_path: str) -> str:
    ffmpeg_cmd = FFMPEG_PATH if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH) else shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        raise RuntimeError("未找到 ffmpeg，请先安装 ffmpeg 或修正 FFMPEG_PATH。")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = tmp.name
    tmp.close()
    subprocess.run(
        [ffmpeg_cmd, "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-y", audio_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return audio_path


def transcribe(audio_path: str, model) -> str:
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]


def save_frames(video_path: str, output_dir: str, fps: int) -> list:
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    interval = max(int(video_fps / fps), 1) if fps > 0 else 1
    frame_id = 0
    saved = 0
    frame_paths = []
    os.makedirs(output_dir, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
            img_path = os.path.join(output_dir, f"{saved + 1}.jpg")
            cv2.imwrite(img_path, frame)
            frame_paths.append(os.path.abspath(img_path))
            saved += 1
        frame_id += 1
    cap.release()
    return frame_paths


def step4_extract_frames_transcribe():
    print("\n" + "═" * 60)
    print("  Step 4 — 抽帧 & Whisper 语音转录")
    print("═" * 60)

    if not os.path.exists(VIDEO_INTRO_JSON):
        print(f"  [错误] 未找到视频信息文件：{VIDEO_INTRO_JSON}，请先运行 Step 2。")
        return

    input_records = sort_records_by_id(load_video_intro_records())
    if not input_records:
        print("  [提示] 视频信息文件为空，无需抽帧。")
        return

    output_map = load_existing_output(CHOUZHEN_JSON)
    model = whisper.load_model(WHISPER_MODEL_NAME)
    skipped = added = failed = 0

    for video_data in tqdm(input_records, desc="抽帧转写"):
        record_id = str(video_data.get("id", "")).strip()
        if not record_id:
            failed += 1
            continue

        image_root_rel = video_data.get("image_root") or build_image_root_relpath(record_id, video_data.get("label", ""))
        image_root_abs = to_repo_path(image_root_rel)
        frames_dir = os.path.join(image_root_abs, "frames")
        existing_record = output_map.get(record_id)
        if existing_record and existing_record.get("all_transcription") and os.path.isdir(frames_dir):
            assign_record_value(existing_record, "label", video_data.get("label", ""))
            assign_record_value(existing_record, "video_url", video_data.get("video_url", ""), force=True)
            assign_record_value(existing_record, "video_introduction", video_data.get("video_introduction", ""))
            assign_record_value(existing_record, "video_api_description", video_data.get("video_api_description", ""))
            assign_record_value(existing_record, "video_path", video_data.get("video_path", ""))
            assign_record_value(existing_record, "image_root", to_repo_relative(image_root_abs), force=True)
            output_map[record_id] = existing_record
            skipped += 1
            continue

        video_path = resolve_video_path(video_data)
        if not is_valid_media_file(video_path):
            failed += 1
            tqdm.write(f"  [失败] ID={record_id} 本地视频不存在，已跳过。")
            continue

        os.makedirs(image_root_abs, exist_ok=True)
        try:
            frame_paths = save_frames(video_path, frames_dir, FRAMES_PER_SECOND)
            if not frame_paths:
                raise RuntimeError("未抽取到有效帧。")
            audio_path = extract_audio(video_path)
            try:
                full_transcript = transcribe(audio_path, model)
            finally:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
        except Exception as e:
            failed += 1
            tqdm.write(f"  [失败] ID={record_id} 抽帧或转写失败，已跳过。{e}")
            continue

        txt_path = os.path.join(image_root_abs, "transcription.txt")
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(full_transcript)

        record = dict(existing_record or {})
        record["id"] = video_data.get("id", record_id)
        record["label"] = video_data.get("label", "")
        record["video_url"] = video_data.get("video_url", "")
        record["video_introduction"] = video_data.get("video_introduction", "")
        record["video_api_description"] = video_data.get("video_api_description", "")
        record["video_path"] = to_repo_relative(video_path)
        record["image_root"] = to_repo_relative(image_root_abs)
        record["all_transcription"] = full_transcript
        output_map[record_id] = record
        dump_json_atomic(CHOUZHEN_JSON, sort_records_by_id(list(output_map.values())))
        added += 1

    if output_map:
        dump_json_atomic(CHOUZHEN_JSON, sort_records_by_id(list(output_map.values())))
    print(f"\n  新增抽帧转写：{added} | 已存在跳过：{skipped} | 失败跳过：{failed}")
    print(f"  ✅ Step 4 完成 → {CHOUZHEN_JSON}")


# ═══════════════════════════════════════════════════════════
#  Step 5  Ollama 多模态视频描述生成
# ═══════════════════════════════════════════════════════════

def detect_language(texts: list) -> str:
    combined = " ".join(texts)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", combined)
    return "zh" if len(chinese_chars) / max(len(combined), 1) > 0.2 else "en"


def get_images_from_folder(folder_path: str) -> list:
    if not os.path.isdir(folder_path):
        return []
    return sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )


def call_ollama_with_images(transcription: str, video_intro: str, frames: list, lang: str = "zh", max_retries: int = 3) -> str:
    if lang == "zh":
        system_prompt = (
            "你是一位视频内容叙述专家，你的任务是根据视频的关键帧图像和音频转录内容，"
            "用中文写出一段完整的故事性描述，帮助没有看过视频的读者完全理解视频讲了什么。"
            "你的描述应自然流畅、像讲故事一样，结合画面和声音的信息，真实、细腻地呈现"
            "视频中的人物、动作、场景、情节发展和情绪变化，使读者仿佛亲眼看过这个视频一样。"
        )
        text_prompt_template = (
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
            "video description in English, combining the visual and audio content."
        )
        text_prompt_template = (
            "Below is the video's introduction, audio transcript, and some keyframe images (batch {batch_idx}):\n\n"
            "Video introduction: {video_intro}\n\n"
            "Audio transcript: {transcription}\n\n"
            "The smaller the image filename number, the earlier it appears in the video. "
            "Please write a natural, coherent, story-like description of the video content."
        )

    full_description = ""
    for batch_idx in range(0, len(frames), MAX_IMAGES_PER_BATCH):
        image_batch = frames[batch_idx : batch_idx + MAX_IMAGES_PER_BATCH]
        valid_images = [p for p in image_batch if os.path.isfile(p)]
        if not valid_images:
            print(f"⚠️ Batch {batch_idx // MAX_IMAGES_PER_BATCH + 1} 无有效图像，跳过。")
            continue

        user_content = text_prompt_template.format(
            batch_idx=batch_idx // MAX_IMAGES_PER_BATCH + 1,
            video_intro=video_intro,
            transcription=transcription,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content, "images": valid_images},
        ]
        for attempt in range(1, max_retries + 1):
            try:
                response = OLLAMA_CLIENT.chat(model=OLLAMA_MODEL, messages=messages)
                full_description += response.message.content.strip() + "\n"
                break
            except Exception as e:
                print(
                    f"⚠️ Ollama 调用失败（batch {batch_idx // MAX_IMAGES_PER_BATCH + 1}，"
                    f"第 {attempt}/{max_retries} 次）：{e}"
                )
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    print("⏭️ 已达最大重试次数，跳过本批次。")
    return full_description.strip()


def step5_generate_descriptions():
    print("\n" + "═" * 60)
    print("  Step 5 — Ollama 多模态视频描述生成")
    print("═" * 60)

    if not os.path.exists(CHOUZHEN_JSON):
        print(f"  [错误] 未找到抽帧文件：{CHOUZHEN_JSON}，请先运行 Step 4。")
        return

    input_data = sort_records_by_id(load_json_data(CHOUZHEN_JSON))
    output_map = load_existing_output(DESCRIPTION_JSON)
    skipped = added = failed = 0

    for video in tqdm(input_data, desc="生成视频描述"):
        video_id = str(video.get("id", "")).strip()
        if not video_id:
            failed += 1
            continue

        existing = output_map.get(video_id)
        if existing and existing.get("video_description"):
            assign_record_value(existing, "label", video.get("label", ""))
            assign_record_value(existing, "video_url", video.get("video_url", ""), force=True)
            assign_record_value(existing, "video_introduction", video.get("video_introduction", ""))
            assign_record_value(existing, "video_api_description", video.get("video_api_description", ""))
            assign_record_value(existing, "video_path", video.get("video_path", ""))
            assign_record_value(existing, "image_root", video.get("image_root", ""))
            assign_record_value(existing, "all_transcription", video.get("all_transcription", ""))
            output_map[video_id] = existing
            skipped += 1
            continue

        transcription = video.get("all_transcription", "")
        video_intro = video.get("video_introduction", "No introduction provided.")
        lang = detect_language([transcription, video_intro])
        image_root_abs = to_repo_path(video.get("image_root") or build_image_root_relpath(video_id, video.get("label", "")))
        frames_dir = os.path.join(image_root_abs, "frames") if os.path.isdir(os.path.join(image_root_abs, "frames")) else image_root_abs
        all_frames = get_images_from_folder(frames_dir)
        frames = all_frames[::FRAME_INTERVAL]

        if not frames:
            failed += 1
            tqdm.write(f"  [失败] ID={video_id} 无有效帧，已跳过。")
            continue

        description = call_ollama_with_images(transcription, video_intro, frames, lang=lang)
        if not description:
            failed += 1
            tqdm.write(f"  [失败] ID={video_id} 描述为空，已跳过。")
            continue

        record = dict(existing or {})
        record["id"] = video.get("id", video_id)
        record["video_url"] = video.get("video_url", "")
        record["video_introduction"] = video.get("video_introduction", "")
        record["video_api_description"] = video.get("video_api_description", "")
        record["label"] = video.get("label", "")
        record["video_path"] = video.get("video_path", "")
        record["image_root"] = video.get("image_root", "")
        record["all_transcription"] = transcription
        record["video_description"] = description
        output_map[video_id] = record
        dump_json_atomic(DESCRIPTION_JSON, sort_records_by_id(list(output_map.values())))
        added += 1

    if output_map:
        dump_json_atomic(DESCRIPTION_JSON, sort_records_by_id(list(output_map.values())))
    print(f"\n  新增描述：{added} | 已存在跳过：{skipped} | 失败跳过：{failed}")
    print(f"  ✅ Step 5 完成 → {DESCRIPTION_JSON}")


# ═══════════════════════════════════════════════════════════
#  Step 6  C_label 自动标注
# ═══════════════════════════════════════════════════════════

# ── 5a. 文本工具 ──────────────────────────────────────────────────

def _s6_clean(text):
    text = re.sub(r"\[\w\u4e00-\u9fff]+\]", " ", text)
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

# ── 5b. 情绪检测（English rules） ────────────────────────────────
# Each tuple: (regex_pattern, emotion_label)
# Covers both pure-English comments and mixed English/emoji comments.
_S6_EMOTION_RULES = [
    # ── deep sadness / empathy ──────────────────────────────────
    (r"(?i)\bcry(ing)?\b|😭|😢|🥺|\bi('m| am) (crying|sobbing|in tears)\b"
     r"|\b(broke my heart|so sad|made me cry|tear(s)? up|heartbroken|moved me)\b",
     "deep_empathy"),

    # ── anger / disgust ─────────────────────────────────────────
    (r"(?i)\b(angry|furious|outraged|disgusting|disgusted|appalled|wtf|what the hell"
     r"|unacceptable|ridiculous|pathetic)\b|😡|🤬|🖕",
     "anger"),

    # ── speechless / awkward ────────────────────────────────────
    (r"(?i)\b(speechless|i can('t| not) even|awkward|cringe|facepalm|embarrassing"
     r"|why would (you|they|he|she)|i('m| am) dead)\b|🤦|😬|🙄|💀",
     "speechless"),

    # ── humor / laughter ────────────────────────────────────────
    (r"(?i)\b(lol|lmao|lmfao|haha|hahaha|rofl|dying|i('m| am) dead"
     r"|so funny|hilarious|can('t| not) stop laughing|this is gold|sending me"
     r"|im weak|i wheezed|lost it|cracked (me )?up)\b|😂|🤣",
     "humor"),

    # ── admiration / praise ─────────────────────────────────────
    (r"(?i)\b(amazing|awesome|incredible|genius|brilliant|well done|great job"
     r"|respect|talented|goat|legendary|fire|banger|underrated|this deserves more"
     r"|take my like|10/10|100/100|chef'?s kiss)\b|👏|🔥|💯|🤩|😍",
     "admiration"),

    # ── affection / cuteness ────────────────────────────────────
    (r"(?i)\b(adorable|so cute|precious|love (this|you|it|them)|i love"
     r"|obsessed|wholesome|my heart|aww+|omg so sweet)\b|❤️|🥰|😻|💕|💖|🫶",
     "affection"),

    # ── sarcasm / irony (verbal signals) ───────────────────────
    (r"(?i)\b(oh sure|yeah right|totally|of course|absolutely|wow thanks"
     r"|great job (genius|buddy|champ|einstein)|nice one|very helpful"
     r"|because that('s| is) (totally |definitely |definitely)?normal"
     r"|no way|sure jan|ok boomer|not like|as if|shocking|who would('ve)? thought)\b"
     r"|🙃|😒",
     "irony"),

    # ── curiosity / questions ───────────────────────────────────
    (r"(?i)\b(what (is|are|was|were|did|does|happened)|how (do|does|did|can|could)"
     r"|(can|could) (you|someone) (explain|tell me|help)|i('m| am) confused"
     r"|wait (what|why|how)|anyone (know|else)|where (is|can|do)|why (is|does|did|would)"
     r"|(does|do) (anyone|somebody) know)\b|\?{2,}|❓",
     "curiosity"),

    # ── mild positive ────────────────────────────────────────────
    (r"(?i)\b(nice|good|pretty good|not bad|decent|okay|ok|alright|fine"
     r"|i like (this|it|that)|looks good|sounds good|cool)\b|👍|🙂",
     "mild_positive"),
]

def _s6_detect_emotion(text: str) -> str:
    """Return emotion label for the comment text, or 'empty'/'neutral'."""
    if not text.strip():
        return "empty"
    for pattern, label in _S6_EMOTION_RULES:
        if re.search(pattern, text):
            return label
    return "neutral"

# ── 5c. 关键词规则 ────────────────────────────────────────────────
_S6_KW_RULES = [
    (1, r"(movie|style|country).*(movie|style|country)|recruiting teammates|still need.* positions|already have.* positions", "Rhyming"),
    (1, r"=|homophonic|actually|read as|[\u4e00-\u9fff]{1,4}[=＝][\u4e00-\u9fff]{1,4}|have you.*?", "Puns (Homophones)"),
    (2, r"day[two three four five six seven eight nine ten\d]|next[episode chapter]|previous[episode chapter]|domineering president|CEO|female lead|male lead|male second|female second"
    r"|medical[hospital][:：]|[she he][:：]|debit\s?[1 one]|gather|come[sign up]|have you thought", "Content Extraction"),
    (2, r"So it turns out I|So it was|Actually|Really|Really thinking|My brain|Substitute|Before.*Now"
    r"|Fantasy|Before dying|You know|Unexpected|No wonder|No wonder|Isn't this|Isn't this.*", "Meme Application"),
    (3, r"This is the way to die|If you're so capable, remake it|I'd rather.*than|I'm not convinced|Numb|Can't hold back|Breaks through|Forget it|Whatever|No way|That's it","Sarcasm (Irony)"),
    (4, r"Haha|Laughing to death|Laughing to get rich|Crazy|Too funny|Fun|[Come on]|Come on|Let's|Charge|Let's go", "Plain Humor"),
]

def _s6_kw_label(text):
    for _, pattern, label in sorted(_S6_KW_RULES, key=lambda x: x[0]):
        if re.search(pattern, text):
            return label
    return None

# ── 5d. KNN 索引 ──────────────────────────────────────────────────
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

def _s6_load_existing(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(item["id"]): item for item in data if "id" in item}
    except Exception as e:
        print(f"  [警告] 读取已有输出文件失败：{e}，将视为空文件处理。")
        return {}

def _s6_check_dup(records, name):
    ids = [str(r.get("id", "")) for r in records]
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

# ── 5e. 主函数 ────────────────────────────────────────────────────
def step6_label_comments():
    """Step 6: 对 top5 评论进行 C_label 标注，合并视频描述，写出 youtube_sample.json。"""
    print("\n" + "═" * 60)
    print("  Step 6 — 评论 C_label 自动标注")
    print("═" * 60)

    for path, desc in [
        (SAMPLE_TRAIN_JSON, "训练集 JSON（SAMPLE_TRAIN_JSON）"),
        (DESCRIPTION_JSON, "视频描述 JSON（DESCRIPTION_JSON）"),
        (TOP5_COMMENT_JSON, "Top5评论 JSON（TOP5_COMMENT_JSON）"),
    ]:
        if not os.path.exists(path):
            print(f"  [错误] 未找到 {desc}：{path}")
            return

    with open(SAMPLE_TRAIN_JSON, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(DESCRIPTION_JSON, "r", encoding="utf-8") as f:
        desc_records = json.load(f)
    with open(TOP5_COMMENT_JSON, "r", encoding="utf-8") as f:
        top5_raw = json.load(f)
    top5_records = list(top5_raw.values()) if isinstance(top5_raw, dict) else top5_raw

    print(
        f"  训练集 {len(train_records)} 条 | "
        f"视频描述 {len(desc_records)} 条 | "
        f"Top5评论 {len(top5_records)} 条"
    )

    desc_ids, _ = _s6_check_dup(desc_records, "youtube_video_description")
    top5_ids, _ = _s6_check_dup(top5_records, "youtube_top5_comment")
    _, _ = _s6_check_dup(train_records, "youtube_comments_sample（训练集）")

    only_in_desc = desc_ids - top5_ids
    only_in_top5 = top5_ids - desc_ids
    if only_in_desc:
        print(f"  ⚠️  仅在视频描述中存在（无对应评论）的 id：{sorted(only_in_desc)}")
    if only_in_top5:
        print(f"  ⚠️  仅在评论文件中存在（无对应描述）的 id：{sorted(only_in_top5)}")
    if not only_in_desc and not only_in_top5:
        print("  ✅  视频描述与评论文件 id 完全对齐")

    existing = _s6_load_existing(YOUTUBE_SAMPLE_JSON)
    if existing:
        print(f"\n  检测到已有输出文件，共 {len(existing)} 条，已存在 id 将直接保留。")

    print("\n  构建 TF-IDF + KNN 索引...")
    corpus = []
    for item in train_records:
        corpus.append(_s6_tokenize(item.get("video_description", "")))
        for i in range(1, 6):
            corpus.append(_s6_tokenize(item.get(f"comment_{i}") or ""))
    for item in desc_records:
        corpus.append(_s6_tokenize(item.get("video_description", "")))

    idf = _s6_build_idf(corpus)
    index = _S6TrainingIndex(train_records, idf)
    print(f"  词表 {len(idf):,} tokens | 训练样本 {len(index.entries)} 条评论")

    id_to_top5 = {str(r.get("id", "")): r for r in top5_records}
    merged = dict(existing)
    all_clabels = []
    skipped = added = 0

    for vd in tqdm(desc_records, desc="标注评论"):
        vid_id = str(vd.get("id", "")).strip()
        if not vid_id:
            continue

        video_url = vd.get("video_url", "")
        video_intro = vd.get("video_introduction", "")
        video_label = vd.get("label", "").strip()
        video_desc = vd.get("video_description", "")
        video_api_description = vd.get("video_api_description", "")
        video_path = vd.get("video_path", "")
        image_root = vd.get("image_root", "")

        if not video_intro:
            top5_meta = id_to_top5.get(vid_id, {})
            video_intro = top5_meta.get("video_introduction", "")

        if vid_id in merged:
            old = merged[vid_id]
            updated = False
            for key, value, force in [
                ("label", video_label, False),
                ("video_url", video_url, True),
                ("video_introduction", video_intro, False),
                ("video_api_description", video_api_description, False),
                ("video_description", video_desc, False),
                ("video_path", video_path, False),
                ("image_root", image_root, False),
            ]:
                before = old.get(key)
                assign_record_value(old, key, value, force=force)
                if old.get(key) != before:
                    updated = True
            if updated:
                tqdm.write(f"  [补齐] ID={vid_id} 已存在，补充缺失字段。")
            else:
                tqdm.write(f"  [跳过] ID={vid_id} 已存在，保留原内容。")
            merged[vid_id] = old
            skipped += 1
            continue

        top5 = id_to_top5.get(vid_id)
        if top5 is None:
            tqdm.write(f"  ⚠️  ID={vid_id} 在 top5 评论文件中找不到对应记录，评论将为空。")

        record = {
            "id": vd.get("id", vid_id),
            "video_url": video_url,
            "video_introduction": video_intro,
            "video_api_description": video_api_description,
            "label": video_label,
            "video_description": video_desc,
            "video_path": video_path,
            "image_root": image_root,
        }
        for i in range(1, 6):
            text = ((top5.get(f"comment_{i}") or "") if top5 else "").strip()
            clabel = _s6_predict_clabel(text, video_desc, video_label, idf, index)
            record[f"comment_{i}"] = text
            record[f"C{i}_label"] = clabel
            if clabel:
                all_clabels.append(clabel)

        merged[vid_id] = record
        dump_json_atomic(YOUTUBE_SAMPLE_JSON, sort_records_by_id(list(merged.values())))
        added += 1

    output_list = sort_records_by_id(list(merged.values()))
    dump_json_atomic(YOUTUBE_SAMPLE_JSON, output_list)

    print(f"\n  新增：{added} 条 | 跳过（已存在）：{skipped} 条 | 文件总计：{len(output_list)} 条")
    print(f"  ✅ Step 6 完成 → {YOUTUBE_SAMPLE_JSON}")
    print("     每条记录均含 label、video_introduction、video_api_description、video_description 字段")

    if all_clabels:
        total = len(all_clabels)
        print("\n  ── 本次新增标签分布 " + "─" * 30)
        for label, cnt in Counter(all_clabels).most_common():
            bar = "█" * round(cnt / total * 20)
            print(f"    {label:25s} {cnt:3d}  {bar}")


# ═══════════════════════════════════════════════════════════
#  主入口 — CLI + 交互式菜单
# ═══════════════════════════════════════════════════════════

STEP_FUNCS = {
    "1": "搜索 YouTube URL",
    "2": "下载视频",
    "3": "抓取评论",
    "4": "抽帧 & Whisper 语音转录",
    "5": "Ollama 多模态视频描述生成",
    "6": "C_label 自动标注",
}


def run_step(
    step_key: str,
    auto_label_target: int,
    comment_label_target: int | None = None,
    interactive_menu: bool = False,
) -> None:
    if step_key == "1":
        if interactive_menu:
            print("\nStep 1 采集模式：")
            print("  1. 五类 Label 批量补齐")
            print("  2. 单个 Label 交互采集")
            mode = input("请选择（默认 1）：").strip() or "1"
            if mode == "2":
                step1_search_urls()
            else:
                target = _read_int_input(
                    f"请输入每个 Label 的目标数量（默认 {auto_label_target}）：",
                    auto_label_target,
                )
                step1_batch_collect_all_labels(target)
        else:
            step1_batch_collect_all_labels(auto_label_target)
    elif step_key == "2":
        step2_download_videos()
    elif step_key == "3":
        step3_fetch_comments(comment_label_target)
    elif step_key == "4":
        step4_extract_frames_transcribe()
    elif step_key == "5":
        step5_generate_descriptions()
    elif step_key == "6":
        step6_label_comments()
    else:
        raise SystemExit(f"未知步骤：{step_key}")


def parse_steps_arg(raw_steps: str) -> list:
    raw = (raw_steps or "").strip().lower()
    if raw == "all":
        return list(STEP_FUNCS.keys())
    steps = [step.strip() for step in raw_steps.split(",") if step.strip()]
    invalid = [step for step in steps if step not in STEP_FUNCS]
    if invalid:
        raise SystemExit(f"无效步骤：{', '.join(invalid)}；可选值为 {', '.join(STEP_FUNCS)} 或 all")
    return steps


def build_arg_parser():
    parser = argparse.ArgumentParser(description="YouTube 数据采集与描述生成一体化流水线")
    parser.add_argument(
        "--steps",
        help="要执行的步骤，使用逗号分隔，例如 1,2,3；或使用 all 执行全部 6 步。",
    )
    parser.add_argument(
        "--auto-label-target",
        type=int,
        default=DEFAULT_BATCH_TARGET,
        help=f"Step 1 批量采集时每个 Label 的目标数量，默认 {DEFAULT_BATCH_TARGET}。",
    )
    parser.add_argument(
        "--comment-label-target",
        type=int,
        default=COMMENT_LABEL_TARGET,
        help="Step 3 每个 Label 的评论抓取上限；传入大于 0 的值时启用按标签限额，传入 0 或负数时全量补抓。",
    )
    return parser


def show_menu(auto_label_target: int) -> None:
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       YouTube 视频数据采集与描述生成 — 一体化流水线      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for key, desc in STEP_FUNCS.items():
        print(f"║  {key}. {desc:<52}║")
    print("║  A. 顺序执行全部步骤（1 → 2 → 3 → 4 → 5 → 6）          ║")
    print("║  Q. 退出                                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    choice = input("\n请选择要执行的步骤：").strip().upper()
    if choice == "Q":
        print("已退出。")
        return
    if choice == "A":
        batch_target = _read_int_input(
            f"请输入 Step 1 批量模式每个 Label 的目标数量（默认 {auto_label_target}）：",
            auto_label_target,
        )
        for key in STEP_FUNCS:
            run_step(key, batch_target, COMMENT_LABEL_TARGET, interactive_menu=False)
        return
    if choice in STEP_FUNCS:
        run_step(choice, auto_label_target, COMMENT_LABEL_TARGET, interactive_menu=True)
        return
    print(f"无效选项：{choice}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.auto_label_target <= 0:
        raise SystemExit("--auto-label-target 必须为正整数。")

    if args.steps:
        for step in parse_steps_arg(args.steps):
            run_step(
                step,
                args.auto_label_target,
                args.comment_label_target,
                interactive_menu=False,
            )
        return

    show_menu(args.auto_label_target)


if __name__ == "__main__":
    main()
