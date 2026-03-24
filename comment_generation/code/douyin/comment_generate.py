"""
抖音视频全流程 + 评论生成 一体化脚本
=====================================
执行顺序：
  Phase 1 — 视频处理 Pipeline（原 allinone.py）
    Step 1: 采集视频 URL（Tag 搜索 或 直接输入 URL）  → douyin_video_url.json
    Step 2: 下载视频 + 获取简介                       → douyin_video_introduction.json + video/*.mp4
    Step 4: 抽帧 + 音频转录                           → douyin_chouzhen.json + image/<id>/frames/
    Step 5: Ollama 多模态模型生成视频描述              → douyin_video_description.json
    Step 3: Ollama LLM 自动 label 分类（最后执行）    → 回写上述 JSON

  Phase 2 — 评论生成（原 test.py）
    对 Phase 1 输出的 douyin_video_description.json 中每条视频：
      1. 语义检索学习样本中 top-k 相似条目
      2. 确定 c_label 及示例评论
      3. Ollama 本地模型生成评论
    → douyin_output_comments.json

依赖安装：
  pip install selenium playwright DrissionPage tqdm opencv-python openai-whisper pydub
              requests ollama numpy jieba
  playwright install chromium
"""

import os
import re
import json
import time
import asyncio
import pickle
import requests
import subprocess
import tempfile
import argparse
import collections
from collections import Counter
from urllib.parse import quote

import cv2
import whisper
import numpy as np
import jieba
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from playwright.async_api import async_playwright
from ollama import chat as ollama_chat
import ollama


# ════════════════════════════════════════════════════════════════
#  ★ 全局配置区 — 所有路径和参数在此统一修改 ★
# ════════════════════════════════════════════════════════════════

BASE_DIR = r"D:\Desktop\video_comment_generation\ALLinone"

# ── Phase 1 中间 / 输出文件路径 ──────────────────────────────────
VIDEO_URL_JSON          = os.path.join(BASE_DIR, "comment_generation", "json", "douyin", "douyin_video_url.json")
VIDEO_INTRO_JSON        = os.path.join(BASE_DIR, "comment_generation", "json", "douyin", "douyin_video_introduction.json")
CHOUZHEN_JSON           = os.path.join(BASE_DIR, "comment_generation", "json", "douyin", "douyin_chouzhen.json")
VIDEO_DESCRIPTION_JSON  = os.path.join(BASE_DIR, "comment_generation", "json", "douyin", "douyin_video_description.json")

# ── Phase 1 文件夹路径 ────────────────────────────────────────────
VIDEO_DIR               = os.path.join(BASE_DIR, "comment_generation", "video", "douyin")
IMAGE_DIR               = os.path.join(BASE_DIR, "comment_generation", "image", "douyin")
USERDATA_DIR            = os.path.join(BASE_DIR, "comment_generation", "userdata_douyin")

# ── Step 3: Label 分类参数 ────────────────────────────────────────
SAMPLE_JSON_PATH  = os.path.join(BASE_DIR, "data_pre", "code", "douyin", "douyin_video_sample.json")
OLLAMA_TEXT_MODEL = "qwen3.5:latest"
LABEL_FEW_SHOT_N  = 20

# ── 支持的 label 列表（Phase 1 & Phase 2 共用）───────────────────
LABELS_ZH = [
    "搞笑短剧类",
    "日常生活段子类",
    "动物搞笑类",
    "幽默解说类",
    "脱口秀表演相声表演类",
    "其他",
]
LABELS_EN = [
    "Comedy Skit",
    "Funny Everyday Moments",
    "Animal Comedy",
    "Humorous Commentary",
    "Talk Show / Crosstalk Performance",
    "Other",
]
# label 双向映射（Phase 2 使用）
LABEL_MAP: dict = {zh: en for zh, en in zip(LABELS_ZH, LABELS_EN)}
LABEL_MAP.update({en: zh for zh, en in zip(LABELS_ZH, LABELS_EN)})

# ── Step 4: 抽帧参数 ──────────────────────────────────────────────
FRAME_FPS          = 1
WHISPER_MODEL_NAME = "tiny"
FFMPEG_PATH        = r"C:\ffmpeg\bin\ffmpeg.exe"

# ── Step 5: Ollama 多模态参数 ─────────────────────────────────────
OLLAMA_MODEL         = "qwen3.5:latest"
MAX_IMAGES_PER_BATCH = 5
FRAME_INTERVAL       = 1

# ── Phase 2: 评论生成配置 ─────────────────────────────────────────
COMMENT_CONFIG = {
    # 输入：Phase 1 生成的视频描述文件（自动使用 VIDEO_DESCRIPTION_JSON）
    "input_video_file":     VIDEO_DESCRIPTION_JSON,

    # 输入：学习样本
    "learning_sample_file": r"D:\Desktop\video_comment_generation\ALLinone\data_pre\json\douyin\sample\douyin_sample.json",

    # 输出：生成的评论结果
    "output_comment_file":  r"D:\Desktop\video_comment_generation\ALLinone\comment_generation\json\result\douyin_output_comments.json",

    # 缓存：预计算的 embedding
    "cache_file":           r"D:\Desktop\video_comment_generation\ALLinone\comment_generation\code\douyin\cached_samples.pkl",

    # 评论生成模型
    "ollama_model":         "qwen3.5:latest",

    # Embedding 模型
    "ollama_embed_model":   "qwen3-embedding:latest",

    # 语义检索 top-k
    "top_k":                3,

    # 每条相似样本最多取几条示例评论
    "examples_per_sample":  2,

    # ── Ollama 生成参数 ───────────────────────────────────────────
    "temperature":          0.75,
    "top_p":                0.9,
    "top_k_sampling":       40,
    "repeat_penalty":       1.1,
    "max_tokens":           512,
    "seed":                 -1,
    "mirostat":             0,
    "mirostat_tau":         5.0,
    "mirostat_eta":         0.1,
    "tfs_z":                1.0,
    "typical_p":            1.0,
    "presence_penalty":     0.0,
    "frequency_penalty":    0.0,
    "num_ctx":              None,
    "num_thread":           None,
}

# ── pydub / ffmpeg 初始化 ─────────────────────────────────────────
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", RuntimeWarning)
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg    = FFMPEG_PATH
    AudioSegment.ffprobe   = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")


# ════════════════════════════════════════════════════════════════
#  公共工具函数
# ════════════════════════════════════════════════════════════════

def load_existing_output(filepath: str) -> dict:
    """读取已有 JSON 输出文件，返回以 id 为 key 的字典；文件不存在返回空字典。"""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(item["id"]): item for item in data if "id" in item}
    except Exception as e:
        print(f"  [警告] 读取已有输出文件失败：{e}，将视为空文件处理。")
        return {}


def _detect_language(texts: list) -> str:
    """根据中文字符占比判断语言，返回 'zh' 或 'en'。"""
    combined = " ".join(t for t in texts if t)
    if not combined:
        return "zh"
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", combined)
    return "zh" if len(chinese_chars) / max(len(combined), 1) > 0.15 else "en"


def detect_language(text: str) -> str:
    """简单判断主体语言是中文还是英文（Phase 2 使用）。"""
    zh_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if zh_chars / max(len(text), 1) > 0.2 else "en"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def _extract_video_id_from_url(url: str) -> str:
    """从抖音视频 URL 中提取视频 ID，支持多种格式。"""
    m = re.search(r"douyin\.com/video/(\d+)", url)
    if m:
        return m.group(1)
    m = re.match(r"^\d+$", url.strip())
    if m:
        return url.strip()
    return ""


# ════════════════════════════════════════════════════════════════
#  STEP 1: 采集视频 URL
# ════════════════════════════════════════════════════════════════

def init_selenium_driver(headless: bool = False, chromedriver_path: str = "") -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--lang=zh-CN")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(executable_path=chromedriver_path) if chromedriver_path else Service()
    driver  = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def extract_videos(driver, max_duration_sec: int, max_count: int, seen_ids: set) -> list:
    """从当前页面提取符合时长条件的视频 ID，返回 video_id 字符串列表。"""
    results = []
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[id^='waterfall_item_']"))
        )
    except Exception:
        print("  [警告] 未找到视频卡片，页面可能需要登录或结构已变化。")
        return results

    cards = driver.find_elements(By.CSS_SELECTOR, "div[id^='waterfall_item_']")
    print(f"  当前页面共发现 {len(cards)} 个视频卡片")

    for card in cards:
        if len(results) >= max_count:
            break
        card_id = card.get_attribute("id")
        match   = re.search(r"waterfall_item_(\d+)", card_id)
        if not match:
            continue
        video_id = match.group(1)

        if video_id in seen_ids:
            continue

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
                       scroll_pause: float = 2.5) -> list:
    all_ids              = []
    seen_ids: set        = set()
    no_new_content_count = 0
    scroll_i             = 0

    while len(all_ids) < max_count:
        scroll_i += 1
        print(f"\n[第 {scroll_i} 次扫描] 已采集 {len(all_ids)}/{max_count} 条")
        batch = extract_videos(driver, max_duration_sec, max_count - len(all_ids), seen_ids)
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


def crawl_tag(driver, tag: str, max_duration_sec: int, max_count: int) -> list:
    search_url = build_search_url(tag)
    print(f"\n  搜索 URL：{search_url}")
    driver.get(search_url)
    time.sleep(2)
    return scroll_and_collect(driver, max_duration_sec, max_count)


def step1_crawl_urls():
    """
    Step 1: 采集视频 URL，保存至 VIDEO_URL_JSON。

    支持两种模式：
      A) Tag 搜索模式：用户输入搜索关键词，自动在抖音搜索页采集视频 URL
      B) 直接输入 URL 模式：用户粘贴视频 URL / 视频 ID，跳过浏览器采集

    label 字段在此阶段留空，由 Step 3 自动填充。
    """
    print("\n" + "═"*60)
    print("  STEP 1：采集抖音视频 URL")
    print("═"*60)
    print("  ℹ️  label 将在 Step 3 由 Ollama 自动分类，此处无需手动指定。")

    print("\n请选择输入方式：")
    print("  1. 输入搜索 Tag（自动搜索并采集视频）")
    print("  2. 直接输入视频 URL / 视频 ID（批量粘贴）")
    while True:
        mode = input("请输入 1 或 2：").strip()
        if mode in ("1", "2"):
            break
        print("  无效输入，请输入 1 或 2。")

    existing = load_existing_output(VIDEO_URL_JSON)
    if existing:
        print(f"  检测到已有输出文件，共 {len(existing)} 条记录，重复 id 将保留原内容。")
    default_start_id = max((int(k) for k in existing.keys()), default=0) + 1 if existing else 1

    # ── 模式 B：直接输入视频 URL ────────────────────────────────
    if mode == "2":
        print("\n请输入视频 URL 或视频 ID（每行一条，输入空行结束）：")
        raw_inputs = []
        while True:
            line = input().strip()
            if not line:
                break
            raw_inputs.append(line)

        if not raw_inputs:
            raise SystemExit("未输入任何 URL，终止。")

        try:
            start_id = int(
                input(f"id 从多少开始（直接回车默认 {default_start_id}）：").strip()
                or str(default_start_id)
            )
            if start_id <= 0:
                raise ValueError
        except ValueError:
            start_id = default_start_id

        parsed_urls = []
        for raw in raw_inputs:
            vid_id = _extract_video_id_from_url(raw)
            if vid_id:
                parsed_urls.append(f"https://www.douyin.com/video/{vid_id}")
            else:
                print(f"  [警告] 无法解析 '{raw}'，将原样保留。")
                parsed_urls.append(raw)

        merged = dict(existing)
        skipped, added = 0, 0
        for i, url in enumerate(parsed_urls):
            id_str = str(start_id + i)
            if id_str in merged:
                print(f"  id={id_str} 已存在，跳过。")
                skipped += 1
            else:
                merged[id_str] = {"id": id_str, "video_url": url, "label": ""}
                added += 1

        output_list = sorted(merged.values(), key=lambda x: int(x["id"]))
        os.makedirs(os.path.dirname(VIDEO_URL_JSON), exist_ok=True)
        with open(VIDEO_URL_JSON, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)

        print(f"\n{'═'*55}")
        print(f"  新增：{added} 条  |  跳过：{skipped} 条  |  总计：{len(output_list)} 条")
        print(f"  保存路径：{VIDEO_URL_JSON}")
        print(f"{'═'*55}")
        print(f"\n  ✅ Step 1 完成（URL 直接输入模式）→ {VIDEO_URL_JSON}")
        return

    # ── 模式 A：Tag 搜索模式 ─────────────────────────────────────
    print("\n请输入搜索 Tag（多个 tag 用逗号分隔，例如：搞笑短剧,搞笑动物）：")
    tag_input = input("Tag：").strip()
    if not tag_input:
        raise SystemExit("未输入任何 Tag，终止。")
    chosen_tags = [t.strip() for t in tag_input.split(",") if t.strip()]

    max_dur_str = input("\n最大视频时长（格式 MM:SS，直接回车默认 5:00）：").strip() or "5:00"
    max_duration_sec = parse_duration(max_dur_str)
    if max_duration_sec <= 0:
        print("  格式有误，使用默认 5:00")
        max_dur_str      = "5:00"
        max_duration_sec = 300

    try:
        count = int(input("需要采集多少条视频 URL（总数）：").strip())
        if count <= 0:
            raise ValueError
    except ValueError:
        count = 30
        print(f"  输入无效，默认 {count} 条")

    try:
        start_id = int(
            input(f"id 从多少开始（直接回车默认 {default_start_id}）：").strip()
            or str(default_start_id)
        )
        if start_id <= 0:
            raise ValueError
    except ValueError:
        start_id = default_start_id

    print(f"\n{'═'*55}")
    print(f"  Tags     : {', '.join(chosen_tags)}")
    print(f"  时长上限 : {max_dur_str}（{max_duration_sec} 秒）")
    print(f"  采集总数 : {count} 条  |  id 起始：{start_id}")
    print(f"  保存路径 : {VIDEO_URL_JSON}")
    print(f"{'═'*55}")

    driver        = init_selenium_driver(headless=False)
    new_video_ids = []
    try:
        combined_tag  = "#" + "#".join(chosen_tags)
        print(f"\n  搜索词：{combined_tag}")
        new_video_ids = crawl_tag(driver, combined_tag, max_duration_sec, count)
        print(f"  采集完成：{len(new_video_ids)} 条")
    finally:
        driver.quit()

    merged = dict(existing)
    skipped, added = 0, 0
    for i, vid in enumerate(new_video_ids):
        id_str = str(start_id + i)
        if id_str in merged:
            print(f"  id={id_str} 已存在，跳过。")
            skipped += 1
        else:
            merged[id_str] = {
                "id":        id_str,
                "video_url": f"https://www.douyin.com/video/{vid}",
                "label":     "",
            }
            added += 1

    output_list = sorted(merged.values(), key=lambda x: int(x["id"]))
    os.makedirs(os.path.dirname(VIDEO_URL_JSON), exist_ok=True)
    with open(VIDEO_URL_JSON, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*55}")
    print(f"  新增：{added} 条  |  跳过：{skipped} 条  |  总计：{len(output_list)} 条")
    print(f"  保存路径：{VIDEO_URL_JSON}")
    print(f"{'═'*55}")
    print(f"\n  ✅ Step 1 完成（Tag 搜索模式）→ {VIDEO_URL_JSON}")


# ════════════════════════════════════════════════════════════════
#  STEP 2: 下载视频 + 获取视频简介
# ════════════════════════════════════════════════════════════════

async def _get_aweme_detail(share_url: str):
    aweme_detail = None
    found_event  = asyncio.Event()

    async def intercept_aweme_response(response):
        nonlocal aweme_detail
        try:
            if "application/json" in response.headers.get("content-type", "").lower():
                json_body = await response.json()
                if isinstance(json_body, dict) and "aweme_detail" in json_body:
                    aweme_detail = json_body["aweme_detail"]
                    found_event.set()
        except Exception:
            pass

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(java_script_enabled=True)
        page    = await context.new_page()
        page.on("response", intercept_aweme_response)

        async def navigate_and_wait():
            try:
                await page.goto(share_url, wait_until="networkidle", timeout=60000)
            except Exception:
                pass

        task = asyncio.create_task(navigate_and_wait())
        try:
            await asyncio.wait_for(found_event.wait(), timeout=15)
        except asyncio.TimeoutError:
            print(f"[警告] 未获取到 aweme_detail: {share_url}")

        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        for obj in (page, context, browser):
            try:
                await obj.close()
            except Exception:
                pass

    return aweme_detail


def _download_video_file(video_url: str, filename: str) -> bool:
    headers = {"User-Agent": "Mozilla/5.0"}
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
    """Step 2: 下载视频 + 获取简介，保存至 VIDEO_INTRO_JSON。已存在的 id 直接保留。"""
    print("\n" + "═"*60)
    print("  STEP 2：下载视频 + 获取视频简介")
    print("═"*60)

    os.makedirs(VIDEO_DIR, exist_ok=True)
    with open(VIDEO_URL_JSON, "r", encoding="utf-8") as f:
        video_list = json.load(f)

    existing: dict = load_existing_output(VIDEO_INTRO_JSON)
    if existing:
        print(f"  检测到已有输出文件，共 {len(existing)} 条，已存在的 id 将直接保留。")

    skipped, added, failed = 0, 0, 0
    new_items: dict = {}
    print(f"\n📥 共需处理 {len(video_list)} 个视频...\n")

    for item in tqdm(video_list, desc="下载进度", unit="视频"):
        video_url = item.get("video_url", "")
        video_id  = str(item.get("id", "")).strip()

        if video_id in existing:
            tqdm.write(f"  [跳过] ID={video_id} 已存在，保留原内容。")
            skipped += 1
            continue

        tqdm.write(f"\n▶ ID={video_id}  URL={video_url}")
        aweme = await _get_aweme_detail(video_url)
        if not aweme:
            tqdm.write("  [失败] 无法获取 aweme_detail")
            failed += 1
            continue

        item["video_introduction"] = aweme.get("desc", "")
        play_urls = aweme.get("video", {}).get("play_addr", {}).get("url_list", [])
        if not play_urls:
            tqdm.write("  [失败] 未找到播放地址")
            failed += 1
            continue

        filepath = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if not _download_video_file(play_urls[0], filepath):
            tqdm.write("  [失败] 视频下载失败")
            failed += 1
            continue

        tqdm.write(f"  [完成] 已保存：{filepath}")
        new_items[video_id] = item
        added += 1

    merged      = {**existing, **new_items}
    result_list = sorted(
        merged.values(),
        key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else float("inf")
    )
    os.makedirs(os.path.dirname(VIDEO_INTRO_JSON), exist_ok=True)
    with open(VIDEO_INTRO_JSON, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"\n  新增：{added} | 跳过：{skipped} | 失败：{failed} | 总计：{len(result_list)}")
    print(f"  ✅ Step 2 完成 → {VIDEO_INTRO_JSON}")


# ════════════════════════════════════════════════════════════════
#  STEP 3: Ollama LLM 自动 label 分类
# ════════════════════════════════════════════════════════════════

def _load_fewshot_samples(sample_path: str, n_per_label: int) -> list:
    if not os.path.exists(sample_path):
        return []
    with open(sample_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    buckets: dict = {}
    for item in samples:
        label = item.get("label", "").strip()
        intro = item.get("video_introduction", "").strip()
        if not label or not intro:
            continue
        buckets.setdefault(label, []).append(item)

    result = []
    for label, items in buckets.items():
        picked = items[:n_per_label]
        result.extend(picked)

    return result


def _build_label_prompt(samples: list, target_intro: str, target_transcription: str,
                        target_description: str, lang: str) -> str:
    if lang == "zh":
        category_lines = "\n".join(f"{i+1}. {lbl}" for i, lbl in enumerate(LABELS_ZH))
        category_block = f"视频分类包括以下几类，请根据视频简介、转录文本和视频描述进行判断：\n{category_lines}\n"

        example_block = ""
        for item in samples:
            example_block += (
                f"视频简介：{item.get('video_introduction', '')}\n"
                f"转录内容：{item.get('all_transcription', item.get('video_description', ''))}\n"
                f"类别：{item.get('label', '')}\n\n"
            )

        valid_labels_str = "、".join(LABELS_ZH)
        query_block = (
            f"请根据上述分类定义和示例，为下面这个视频判断其类别，并用一句话说明原因。\n"
            f"注意：类别必须严格从以下列表中选择一个，不得自创：{valid_labels_str}\n"
            f"请严格按以下格式输出，不要输出任何其他内容：\n"
            f"类别：<类别名>\n"
            f"原因：<原因>\n\n"
            f"视频简介：{target_intro}\n"
            f"转录内容：{target_transcription}\n"
            f"视频描述：{target_description}\n"
        )
        return category_block + "\n以下是已标注的示例：\n" + example_block + query_block

    else:
        category_lines = "\n".join(f"{i+1}. {lbl}" for i, lbl in enumerate(LABELS_EN))
        category_block = (
            f"Classify the video into one of these categories based on its introduction, "
            f"transcription, and description:\n{category_lines}\n"
        )

        example_block = ""
        for item in samples:
            example_block += (
                f"Video Introduction: {item.get('video_introduction', '')}\n"
                f"Transcription: {item.get('all_transcription', item.get('video_description', ''))}\n"
                f"Category: {item.get('label', '')}\n\n"
            )

        valid_labels_str = ", ".join(LABELS_EN)
        query_block = (
            f"Based on the above definitions and examples, classify the following video "
            f"and give a one-sentence reason.\n"
            f"Note: The category MUST be chosen strictly from this list: {valid_labels_str}\n"
            f"Output ONLY in this exact format, nothing else:\n"
            f"Category: <category name>\n"
            f"Reason: <reason>\n\n"
            f"Video Introduction: {target_intro}\n"
            f"Transcription: {target_transcription}\n"
            f"Video Description: {target_description}\n"
        )
        return category_block + "\nHere are labeled examples:\n" + example_block + query_block


def _parse_label_response(response_text: str, lang: str) -> str:
    label  = ""
    labels = LABELS_ZH if lang == "zh" else LABELS_EN

    if lang == "zh":
        m = re.search(r"类别[：:]\s*(.+)", response_text)
        if m:
            label = m.group(1).strip().rstrip("。．.")
    else:
        m = re.search(r"Category[：:]\s*(.+)", response_text)
        if m:
            label = m.group(1).strip().rstrip(".")

    if label not in labels:
        for lbl in labels:
            if lbl in response_text:
                label = lbl
                break
        else:
            label = labels[-1]

    return label


def _classify_one_video(intro: str, transcription: str, description: str,
                        samples: list, lang: str, max_retries: int = 3) -> str:
    prompt = _build_label_prompt(samples, intro, transcription, description, lang)

    for attempt in range(1, max_retries + 1):
        try:
            response = ollama_chat(
                model=OLLAMA_TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.message.content.strip()
            return _parse_label_response(raw, lang)
        except Exception as e:
            print(f"    [警告] Ollama 调用失败（第 {attempt}/{max_retries} 次）：{e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)

    return LABELS_ZH[-1] if lang == "zh" else LABELS_EN[-1]


def step3_label_videos():
    """Step 3: 用 Ollama 本地 LLM 对每条视频进行 label 分类，回写所有相关 JSON。"""
    print("\n" + "═"*60)
    print(f"  STEP 3：Ollama（{OLLAMA_TEXT_MODEL}）自动 label 分类")
    print("═"*60)
    print("  输入来源：video_introduction + all_transcription + video_description")

    if not os.path.exists(VIDEO_DESCRIPTION_JSON):
        print(f"  [错误] 未找到 {VIDEO_DESCRIPTION_JSON}，请先执行 Step 4 和 Step 5。")
        return

    with open(VIDEO_DESCRIPTION_JSON, "r", encoding="utf-8") as f:
        desc_list = json.load(f)

    samples = _load_fewshot_samples(SAMPLE_JSON_PATH, LABEL_FEW_SHOT_N)
    if samples:
        label_counts: dict = {}
        for s in samples:
            lbl = s.get("label", "未知")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"  已从 {SAMPLE_JSON_PATH}")
        print(f"  加载 {len(samples)} 条 few-shot 示例（每个 label 最多 {LABEL_FEW_SHOT_N} 条，模型：{OLLAMA_TEXT_MODEL}）")
        for lbl, cnt in sorted(label_counts.items()):
            print(f"    · {lbl}: {cnt} 条")
    else:
        print(f"  [警告] 样本文件为空或不存在：{SAMPLE_JSON_PATH}")
        print(f"          将在无 few-shot 示例的情况下继续（准确率可能下降）。")

    print(f"\n  共 {len(desc_list)} 条视频待处理...\n")

    skipped, added, failed = 0, 0, 0

    for item in tqdm(desc_list, desc="  label 分类", unit="条"):
        video_id = str(item.get("id", "")).strip()

        if item.get("label", "").strip():
            tqdm.write(f"  [跳过] ID={video_id}  label='{item['label']}' 已存在")
            skipped += 1
            continue

        intro         = item.get("video_introduction", "").strip()
        transcription = item.get("all_transcription", "").strip()
        description   = item.get("video_description", "").strip()

        if not intro and not transcription and not description:
            tqdm.write(f"  [跳过] ID={video_id}  所有文本字段均为空，无法分类")
            skipped += 1
            continue

        lang = _detect_language([intro, transcription, description])
        lang_samples = [s for s in samples if _detect_language([s.get("video_introduction", "")]) == lang]
        use_samples  = lang_samples if lang_samples else samples

        tqdm.write(f"  ▶ ID={video_id}  lang={lang}  分类中...")
        label = _classify_one_video(intro, transcription, description, use_samples, lang)

        item["label"] = label
        tqdm.write(f"    → label={label}")
        added += 1

    os.makedirs(os.path.dirname(VIDEO_DESCRIPTION_JSON), exist_ok=True)
    with open(VIDEO_DESCRIPTION_JSON, "w", encoding="utf-8") as f:
        json.dump(desc_list, f, ensure_ascii=False, indent=2)

    desc_map = {str(item["id"]): item for item in desc_list}
    if os.path.exists(VIDEO_INTRO_JSON):
        with open(VIDEO_INTRO_JSON, "r", encoding="utf-8") as f:
            intro_list = json.load(f)
        for entry in intro_list:
            eid = str(entry.get("id", ""))
            if eid in desc_map and desc_map[eid].get("label"):
                entry["label"] = desc_map[eid]["label"]
        with open(VIDEO_INTRO_JSON, "w", encoding="utf-8") as f:
            json.dump(intro_list, f, ensure_ascii=False, indent=2)
        print(f"  label 已同步回写至 {VIDEO_INTRO_JSON}")

    if os.path.exists(VIDEO_URL_JSON):
        with open(VIDEO_URL_JSON, "r", encoding="utf-8") as f:
            url_list = json.load(f)
        for entry in url_list:
            eid = str(entry.get("id", ""))
            if eid in desc_map and desc_map[eid].get("label"):
                entry["label"] = desc_map[eid]["label"]
        with open(VIDEO_URL_JSON, "w", encoding="utf-8") as f:
            json.dump(url_list, f, ensure_ascii=False, indent=2)
        print(f"  label 已同步回写至 {VIDEO_URL_JSON}")

    all_labels = [item.get("label", "") for item in desc_list if item.get("label", "").strip()]
    print(f"\n  新增分类：{added} 条 | 跳过：{skipped} 条 | 失败：{failed} 条")
    if all_labels:
        total = len(all_labels)
        print("\n  ── 当前文件 label 分布 " + "─"*30)
        for lbl, cnt in Counter(all_labels).most_common():
            bar = "█" * round(cnt / total * 20)
            print(f"    {lbl:35s} {cnt:3d}  {bar}")
    print(f"\n  ✅ Step 3 完成 → {VIDEO_DESCRIPTION_JSON}")


# ════════════════════════════════════════════════════════════════
#  STEP 4: 抽帧 + 音频转录
# ════════════════════════════════════════════════════════════════

_ffmpeg_exe = None


def _find_ffmpeg() -> str:
    global _ffmpeg_exe
    if _ffmpeg_exe:
        return _ffmpeg_exe

    import shutil

    def _set(path: str) -> str:
        global _ffmpeg_exe
        _ffmpeg_exe            = path
        AudioSegment.converter = path
        AudioSegment.ffmpeg    = path
        AudioSegment.ffprobe   = path.replace("ffmpeg.exe", "ffprobe.exe")
        ffmpeg_dir = os.path.dirname(path)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return path

    if FFMPEG_PATH:
        if os.path.isfile(FFMPEG_PATH):
            return _set(FFMPEG_PATH)
        raise FileNotFoundError(f"配置区 FFMPEG_PATH 指定路径不存在：{FFMPEG_PATH}")

    found = shutil.which("ffmpeg")
    if found:
        return _set(found)

    for path in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(os.environ.get("USERPROFILE",  ""), "ffmpeg", "bin", "ffmpeg.exe"),
    ]:
        if os.path.isfile(path):
            return _set(path)

    raise FileNotFoundError(
        "\n[错误] 找不到 ffmpeg！\n"
        "  方式 A：下载后加入系统 PATH：https://www.gyan.dev/ffmpeg/builds/\n"
        "  方式 B：在配置区填写完整路径：FFMPEG_PATH = r'C:\\ffmpeg\\bin\\ffmpeg.exe'\n"
    )


def _extract_audio(video_path: str) -> str:
    ffmpeg     = _find_ffmpeg()
    tmp        = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = tmp.name
    tmp.close()
    subprocess.run(
        [ffmpeg, "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-y", audio_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return audio_path


def _transcribe(audio_path: str, model) -> str:
    return model.transcribe(audio_path, fp16=False)["text"]


def _save_frames(video_path: str, output_dir: str, fps: int) -> list:
    cap       = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval  = max(int(video_fps / fps), 1)
    frame_id, saved = 0, 0
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


def step4_chouzhen():
    """Step 4: 抽帧 + Whisper 转录，保存至 CHOUZHEN_JSON。已存在的 id 直接保留。"""
    print("\n" + "═"*60)
    print("  STEP 4：视频抽帧 + 音频转录")
    print("═"*60)

    ffmpeg_exe = _find_ffmpeg()
    print(f"  [ffmpeg] 使用路径：{ffmpeg_exe}")

    model = whisper.load_model(WHISPER_MODEL_NAME)

    with open(VIDEO_INTRO_JSON, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    label_map = {f"{item['id']}.mp4": item for item in original_data}
    for item in original_data:
        label_map[item.get("video_url", "")] = item

    existing: dict = load_existing_output(CHOUZHEN_JSON)
    if existing:
        print(f"  检测到已有输出文件，共 {len(existing)} 条，已存在的 id 将直接保留。")

    os.makedirs(IMAGE_DIR, exist_ok=True)
    videos = sorted(
        [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))],
        key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else float("inf"),
    )
    print(f"  共发现 {len(videos)} 个视频文件")

    new_items: dict = {}
    skipped, added  = 0, 0

    for video_file in tqdm(videos, desc="📦 处理视频", unit="个"):
        video_name = os.path.splitext(video_file)[0]

        if video_name in existing:
            tqdm.write(f"  [跳过] ID={video_name} 已存在，保留原内容。")
            skipped += 1
            continue

        video_path = os.path.join(VIDEO_DIR, video_file)
        frame_dir  = os.path.join(IMAGE_DIR, video_name, "frames")
        main_frames = _save_frames(video_path, frame_dir, FRAME_FPS)

        audio_path      = _extract_audio(video_path)
        full_transcript = _transcribe(audio_path, model)
        os.remove(audio_path)

        with open(os.path.join(IMAGE_DIR, video_name, "transcription.txt"), "w", encoding="utf-8") as tf:
            tf.write(full_transcript)

        meta = label_map.get(video_file, label_map.get(f"https://www.douyin.com/video/{video_name}", {}))
        new_items[video_name] = {
            "id":                video_name,
            "video_url":         video_file,
            "video_introduction": meta.get("video_introduction", ""),
            "label":              meta.get("label", ""),
            "image":              main_frames,
            "all_transcription":  full_transcript,
        }
        added += 1

    merged      = {**existing, **new_items}
    result_json = sorted(
        merged.values(),
        key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else float("inf"),
    )
    with open(CHOUZHEN_JSON, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)

    print(f"\n  新增：{added} | 跳过：{skipped} | 总计：{len(result_json)}")
    print(f"  ✅ Step 4 完成 → {CHOUZHEN_JSON}")


# ════════════════════════════════════════════════════════════════
#  STEP 5: Ollama 多模态模型生成视频描述
# ════════════════════════════════════════════════════════════════

def _call_ollama_with_images(transcription: str, video_intro: str,
                             frames: list, lang: str = "zh",
                             max_retries: int = 3) -> str:
    """调用本地 Ollama 多模态模型，逐批传入关键帧，拼接生成完整视频描述。"""
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
            "who hasn't seen the video can fully understand what it is about."
        )
        text_template = (
            "Below is the video's introduction, audio transcript, and some keyframe images (batch {batch_idx}):\n\n"
            "Video introduction: {video_intro}\n\n"
            "Audio transcript: {transcription}\n\n"
            "The smaller the image filename number, the earlier it appears in the video. "
            "Please write a natural, coherent, story-like description of the video content."
        )

    for batch_start in range(0, len(frames), MAX_IMAGES_PER_BATCH):
        image_batch  = frames[batch_start:batch_start + MAX_IMAGES_PER_BATCH]
        valid_images = [p for p in image_batch if os.path.isfile(p)]
        if not valid_images:
            continue

        batch_idx = batch_start // MAX_IMAGES_PER_BATCH + 1
        messages  = [
            {"role": "system", "content": system_prompt},
            {
                "role":    "user",
                "content": text_template.format(
                    batch_idx=batch_idx,
                    video_intro=video_intro,
                    transcription=transcription,
                ),
                "images": valid_images,
            },
        ]

        for attempt in range(1, max_retries + 1):
            try:
                response = ollama_chat(model=OLLAMA_MODEL, messages=messages)
                full_description += response.message.content.strip() + "\n"
                break
            except Exception as e:
                print(f"  ⚠️ Ollama 调用失败（批次 {batch_idx}，第 {attempt}/{max_retries} 次）：{e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    print("  ⏭️ 已达最大重试次数，跳过此批次。")

    return full_description.strip()


def step5_generate_descriptions():
    """Step 5: 调用本地 Ollama 多模态模型生成每个视频的描述，保存至 VIDEO_DESCRIPTION_JSON。"""
    print("\n" + "═"*60)
    print(f"  STEP 5：Ollama（{OLLAMA_MODEL}）生成视频描述")
    print("═"*60)

    with open(CHOUZHEN_JSON, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    existing: dict = load_existing_output(VIDEO_DESCRIPTION_JSON)
    if existing:
        print(f"  检测到已有输出文件，共 {len(existing)} 条，已存在的 id 将直接保留。")

    new_items: dict = {}
    skipped, added  = 0, 0

    for video in tqdm(input_data, desc="Processing videos"):
        video_id      = str(video.get("id", "")).strip()
        transcription = video.get("all_transcription", "")
        video_intro   = video.get("video_introduction", "")

        if video_id in existing:
            tqdm.write(f"  [跳过] ID={video_id} 已存在，保留原内容。")
            skipped += 1
            continue

        frames          = video.get("image", [])
        selected_frames = frames[::FRAME_INTERVAL] if frames else []
        lang            = _detect_language([transcription, video_intro])

        description = _call_ollama_with_images(
            transcription=transcription,
            video_intro=video_intro,
            frames=selected_frames,
            lang=lang,
        )

        record = dict(video)
        record["video_description"] = description
        record.pop("image", None)
        record.pop("all_transcription", None)

        new_items[video_id] = record
        added += 1

    merged      = {**existing, **new_items}
    result_list = sorted(
        merged.values(),
        key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else float("inf"),
    )
    os.makedirs(os.path.dirname(VIDEO_DESCRIPTION_JSON), exist_ok=True)
    with open(VIDEO_DESCRIPTION_JSON, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"\n  新增：{added} | 跳过：{skipped} | 总计：{len(result_list)}")
    print(f"  ✅ Step 5 完成 → {VIDEO_DESCRIPTION_JSON}")


# ════════════════════════════════════════════════════════════════
#  PHASE 2: 评论生成（原 test.py）
# ════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> np.ndarray:
    """调用 Ollama 本地 embedding 模型。"""
    resp = ollama.embed(model=COMMENT_CONFIG["ollama_embed_model"], input=text)
    vec = resp.embeddings[0] if hasattr(resp, "embeddings") else resp["embeddings"][0]
    return np.array(vec, dtype=np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def build_sample_text(sample: dict) -> str:
    """把一条样本拼成用于 embedding 的文本。"""
    parts = [
        sample.get("video_introduction", ""),
        sample.get("video_description", ""),
        sample.get("all_transcription", ""),
    ]
    return " ".join(p for p in parts if p)


def preprocess_samples(samples: list) -> list:
    """给每条学习样本计算 embedding，结果写回 sample dict。"""
    print("🔄 Computing embeddings for learning samples ...")
    for s in tqdm(samples, desc="Embedding samples", unit="sample"):
        text = build_sample_text(s)
        s["_embedding"] = get_embedding(text)
    return samples


def get_preprocessed_samples() -> list:
    """若缓存存在且完整则直接加载，否则重新计算并保存。"""
    cache_path = COMMENT_CONFIG["cache_file"]
    if os.path.exists(cache_path):
        try:
            print(f"✅ Loading cached samples from {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if not data:
                raise ValueError("Cache is empty")
            return data
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            print(f"⚠️  Cache corrupted ({e}), rebuilding ...")
            os.remove(cache_path)

    raw = load_json(COMMENT_CONFIG["learning_sample_file"])
    processed = preprocess_samples(raw)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(processed, f)
    print(f"✅ Cached {len(processed)} samples to {cache_path}")
    return processed


def build_label_index(samples: list) -> dict:
    """构建 label -> [sample, ...] 的索引。"""
    index = collections.defaultdict(list)
    for s in samples:
        label = s.get("label") or s.get("C1_label") or ""
        if label:
            index[label].append(s)
    return index


def top_k_similar(query_emb: np.ndarray, candidate_samples: list, k: int) -> list:
    """在候选样本中，返回与 query_emb 余弦相似度最高的前 k 条。"""
    scored = [
        (cosine_similarity(query_emb, s["_embedding"]), s)
        for s in candidate_samples
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def _get_c_labels(sample: dict) -> list:
    """从 sample 提取所有 c_label，兼容扁平/列表/单值格式。"""
    labels = []
    for i in range(1, 6):
        cl = sample.get(f"C{i}_label", "")
        if cl:
            labels.append(cl)
    if labels:
        return labels
    if "comments" in sample and isinstance(sample["comments"], list):
        return [c.get("c_label", "") for c in sample["comments"] if c.get("c_label")]
    cl = sample.get("c_label", "")
    return [cl] if cl else []


def _get_comments_by_c_label(sample: dict, c_label: str, top_n: int) -> list:
    """从 sample 中提取指定 c_label 对应的评论文本，最多取 top_n 条。"""
    results = []
    for i in range(1, 6):
        if sample.get(f"C{i}_label", "") == c_label:
            text = sample.get(f"comment_{i}", "")
            if text:
                results.append(text)
        if len(results) >= top_n:
            break
    if results:
        return results[:top_n]
    if "comments" in sample and isinstance(sample["comments"], list):
        matched = [c for c in sample["comments"] if c.get("c_label") == c_label]
        matched.sort(key=lambda c: c.get("rank", 9999))
        return [c.get("content", c.get("comment", "")) for c in matched[:top_n]]
    if sample.get("c_label") == c_label:
        text = sample.get("comment", sample.get("content", ""))
        return [text] if text else []
    return []


def pick_c_label_and_examples(top_samples: list) -> tuple:
    """输入 top-k (similarity, sample) 列表，返回 (最高频 c_label, 示例评论列表)。"""
    c_label_counter = collections.Counter()
    for _, s in top_samples:
        c_label_counter.update(_get_c_labels(s))

    best_c_label = c_label_counter.most_common(1)[0][0] if c_label_counter else "普通幽默"

    examples = []
    n = COMMENT_CONFIG["examples_per_sample"]
    for _, s in top_samples:
        examples.extend(_get_comments_by_c_label(s, best_c_label, n))

    return best_c_label, examples


def build_prompt(video: dict, c_label: str, examples: list) -> str:
    desc       = video.get("video_description", "")
    intro      = video.get("video_introduction", "")
    transcript = video.get("all_transcription", "")
    lang       = detect_language(desc)

    example_text = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(examples) if e)

    if lang == "zh":
        prompt = f"""你是一个抖音评论生成助手，擅长模仿真实用户的评论风格。

【视频信息】
- 视频介绍：{intro}
- 视频描述：{desc}
- 字幕信息：{transcript}

【目标评论风格】：{c_label}

【参考评论示例】（模仿这些句式和风格，但内容必须结合当前视频）：
{example_text}

要求：
- 评论要自然、真实，像真人在评论区留言
- 模仿示例的句式结构和语气，但不要照抄
- 内容必须结合本视频的 description
- 只输出一句评论，不要解释、不要加引号

评论："""
    else:
        prompt = f"""You are a YouTube comment generator. Mimic the style of real user comments.

[Video Info]
- Introduction: {intro}
- Description: {desc}
- Transcript: {transcript}

[Target comment style]: {c_label}

[Reference examples] (mimic the sentence structure and tone, but adapt to this video):
{example_text}

Requirements:
- Sound natural and authentic, like a real user
- Imitate the style of the examples, do NOT copy them directly
- Content must relate to this video's description
- Output ONLY the comment, no explanation, no quotes

Comment:"""

    return prompt.strip()


def _build_ollama_options() -> dict:
    """从 COMMENT_CONFIG 中收集 Ollama 生成参数。"""
    mapping = {
        "temperature":       "temperature",
        "top_p":             "top_p",
        "top_k_sampling":    "top_k",
        "repeat_penalty":    "repeat_penalty",
        "max_tokens":        "num_predict",
        "seed":              "seed",
        "mirostat":          "mirostat",
        "mirostat_tau":      "mirostat_tau",
        "mirostat_eta":      "mirostat_eta",
        "tfs_z":             "tfs_z",
        "typical_p":         "typical_p",
        "presence_penalty":  "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "num_ctx":           "num_ctx",
        "num_thread":        "num_thread",
    }
    options = {}
    for cfg_key, ollama_key in mapping.items():
        val = COMMENT_CONFIG.get(cfg_key)
        if val is not None:
            options[ollama_key] = val
    if options.get("seed") == -1:
        del options["seed"]
    return options


def generate_comment(prompt: str) -> str:
    resp = ollama.chat(
        model=COMMENT_CONFIG["ollama_model"],
        messages=[{"role": "user", "content": prompt}],
        options=_build_ollama_options(),
        think=False,
    )
    try:
        content = resp.message.content
    except AttributeError:
        content = resp["message"]["content"]
    return content.strip() if content else ""


def phase2_generate_comments():
    """Phase 2: 对 Phase 1 生成的视频描述文件逐条生成评论，保存至 output_comment_file。"""
    print("\n\n" + "★"*60)
    print("  PHASE 2：评论生成")
    print("★"*60)

    # 同步输入文件路径（以 Phase 1 实际输出为准）
    COMMENT_CONFIG["input_video_file"] = VIDEO_DESCRIPTION_JSON

    # 1. 加载预处理好的学习样本
    samples = get_preprocessed_samples()
    label_index  = build_label_index(samples)
    known_labels = set(label_index.keys())
    print(f"📚 Loaded {len(samples)} samples, {len(known_labels)} distinct labels.")

    # 2. 加载待处理视频
    video_data = load_json(COMMENT_CONFIG["input_video_file"])
    print(f"🎬 Processing {len(video_data)} videos ...")

    for video in tqdm(video_data, desc="Generating comments", unit="video"):
        video_label = video.get("label") or video.get("C1_label") or ""
        video_text  = build_sample_text(video)
        query_emb   = get_embedding(video_text)

        if video_label and video_label in known_labels:
            candidates = label_index[video_label]
            strategy   = f"same-label ({video_label})"
        else:
            candidates = samples
            strategy   = "global-search (new label)"

        top_samples            = top_k_similar(query_emb, candidates, COMMENT_CONFIG["top_k"])
        c_label, examples      = pick_c_label_and_examples(top_samples)
        prompt                 = build_prompt(video, c_label, examples)
        comment                = generate_comment(prompt)

        video["generated_comment"] = comment
        video["generated_c_label"] = c_label

        tqdm.write(f"  [{strategy}] c_label={c_label} | {comment[:60]}...")

    # 3. 保存
    os.makedirs(os.path.dirname(COMMENT_CONFIG["output_comment_file"]), exist_ok=True)
    save_json(video_data, COMMENT_CONFIG["output_comment_file"])
    print(f"\n✅ Done! Comments saved to {COMMENT_CONFIG['output_comment_file']}")


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="抖音视频全流程 + 评论生成一体化脚本",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=str, default="1,2,4,5,3",
        help=(
            "指定 Phase 1 要执行的步骤（逗号分隔），例如 --steps 2,4,5\n"
            "  1: 采集视频 URL\n"
            "  2: 下载视频 + 获取简介\n"
            "  4: 抽帧 + 音频转录\n"
            "  5: Ollama 多模态生成视频描述\n"
            "  3: Ollama LLM 自动 label 分类\n"
            "（默认全部执行，顺序固定为 1→2→4→5→3）"
        ),
    )
    parser.add_argument(
        "--skip-comments", action="store_true",
        help="跳过 Phase 2 评论生成，仅执行 Phase 1 Pipeline",
    )
    args = parser.parse_args()

    steps = set()
    for s in args.steps.split(","):
        s = s.strip()
        if s.isdigit():
            steps.add(int(s))

    # ── Phase 1 ──────────────────────────────────────────────────
    print("\n" + "★"*60)
    print("  PHASE 1：抖音视频处理 Pipeline")
    print(f"  将执行步骤：{sorted(steps)}  （实际执行顺序：1→2→4→5→3）")
    print("★"*60)

    if 1 in steps:
        step1_crawl_urls()

    if 2 in steps:
        asyncio.run(step2_download_videos())

    if 4 in steps:
        step4_chouzhen()

    if 5 in steps:
        step5_generate_descriptions()

    if 3 in steps:
        step3_label_videos()

    print("\n\n" + "★"*60)
    print("  ✅ Phase 1 所有指定步骤已完成！")
    print("★"*60)

    # ── Phase 2 ──────────────────────────────────────────────────
    if not args.skip_comments:
        phase2_generate_comments()
        print("\n" + "★"*60)
        print("  ✅ Phase 2 评论生成完成！")
        print("★"*60)
    else:
        print("\n  （已跳过 Phase 2 评论生成）")


if __name__ == "__main__":
    main()