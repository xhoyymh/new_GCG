"""
抖音视频全流程一键脚本
Pipeline:
  Step 1: 采集视频 URL            → douyin_video_url.json
  Step 2: 下载视频 + 获取简介     → douyin_video_introduction.json + video/*.mp4
  Step 3: 抓取评论                → douyin_top5_comments.json + douyin_all_comments.json
  Step 4: 抽帧 + 音频转录         → douyin_chouzhen.json + image/<id>/frames/
  Step 5: GPT 生成视频描述        → douyin_video_description.json

依赖安装：
  pip install selenium playwright DrissionPage tqdm opencv-python openai-whisper pydub requests
  playwright install chromium
"""

import os
import re
import json
import time
import asyncio
import base64
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
# pydub 延迟导入（在配置区之后），避免 ffmpeg 路径未设置时触发 RuntimeWarning

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from DrissionPage import ChromiumOptions, ChromiumPage


# ════════════════════════════════════════════════════════════════
#  ★ 全局配置区 — 所有路径和参数在此统一修改 ★
# ════════════════════════════════════════════════════════════════

BASE_DIR = r"D:\Desktop\video_comment_generation\ALLinone"

# 中间/输出文件路径
VIDEO_URL_JSON          = os.path.join(BASE_DIR, "data_pre","json", "douyin", "data_pre", "douyin_video_url.json")
VIDEO_INTRO_JSON        = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre","douyin_video_introduction.json")
ALL_COMMENTS_JSON       = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre", "douyin_all_comments.json")
TOP5_COMMENTS_JSON      = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre", "douyin_top5_comments.json")
CHOUZHEN_JSON           = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre", "douyin_chouzhen.json")
VIDEO_DESCRIPTION_JSON  = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "data_pre", "douyin_video_description.json")

# 参考样本 JSON（用于 label/tag 分析，Step 1）
SAMPLE_JSON_PATH        = os.path.join(BASE_DIR, "data_pre", "code", "douyin", "douyin_video_sample.json")

# 文件夹路径
VIDEO_DIR               = os.path.join(BASE_DIR, "data_pre", "video", "douyin")
IMAGE_DIR               = os.path.join(BASE_DIR, "data_pre", "douyin_image")
USERDATA_DIR            = os.path.join(BASE_DIR, "data_pre", "userdata_douyin")

# Step 3: 评论参数
SCROLL_ROUNDS           = 30       # 评论页滚动次数

# Step 4: 抽帧参数
FRAME_FPS               = 1        # 每秒抽帧数
WHISPER_MODEL_NAME      = "base"   # Whisper 模型大小
# ffmpeg 路径：留空则自动搜索系统 PATH 和常见安装位置
# 如果自动找不到，请手动填写完整路径，例如：
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# pydub 在此处导入，确保 FFMPEG_PATH 已定义，可立即告知 pydub 路径，避免 RuntimeWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
# 如果配置了路径，立即注入给 pydub 并加入 PATH（供 Whisper 等库使用）
if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg    = FFMPEG_PATH
    AudioSegment.ffprobe   = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    _ffmpeg_bin_dir = os.path.dirname(FFMPEG_PATH)
    os.environ["PATH"] = _ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")

# Step 5: GPT 参数
OPENAI_API_KEY          = "YOUR_OPENAI_API_KEY_HERE"
OPENAI_API_URL          = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL            = "gpt-4o"
FRAME_INTERVAL          = 3        # 传给 GPT 的帧间隔（每隔 N 帧取一张）


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
        print("  未选择任何 Tag，退出。")
    return result


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
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def extract_videos(driver, max_duration_sec: int, max_count: int, seen_ids: set) -> list:
    """从当前页面提取符合时长条件的视频 ID，返回 video_id 字符串列表"""
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
                       scroll_pause: float = 2.5) -> list:
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
    """打开搜索页并采集"""
    search_url = build_search_url(tag)
    print(f"\n  搜索 URL：{search_url}")
    driver.get(search_url)
    time.sleep(2)
    return scroll_and_collect(driver, max_duration_sec, max_count)


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
    chosen_tags = select_tags(top_tags, is_new_label)
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

    for i, vid in enumerate(new_video_ids):
        id_str = str(start_id + i)
        if id_str in merged:
            print(f"  id={id_str} 已存在，保留原内容，跳过。")
            skipped += 1
        else:
            merged[id_str] = {
                "id": id_str,
                "video_url": f"https://www.douyin.com/video/{vid}",
                "label": chosen_label
            }
            added += 1

    output_list = sorted(merged.values(), key=lambda x: int(x["id"]))
    os.makedirs(os.path.dirname(VIDEO_URL_JSON), exist_ok=True)
    with open(VIDEO_URL_JSON, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

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
    """Step 2: 下载视频并提取简介，保存至 VIDEO_INTRO_JSON"""
    print("\n" + "═"*60)
    print("  STEP 2：下载视频 + 获取视频简介")
    print("═"*60)

    os.makedirs(VIDEO_DIR, exist_ok=True)
    with open(VIDEO_URL_JSON, "r", encoding="utf-8") as f:
        video_list = json.load(f)

    result_list = []
    print(f"\n📥 共需处理 {len(video_list)} 个视频...\n")

    for item in tqdm(video_list, desc="下载进度", unit="视频"):
        video_url = item.get("video_url", "")
        video_id  = item.get("id", "")
        tqdm.write(f"\n▶ 正在处理: ID={video_id} URL={video_url}")

        aweme = await _get_aweme_detail(video_url)
        if not aweme:
            tqdm.write("[失败] 无法获取 aweme_detail")
            continue

        item["video_introduction"] = aweme.get("desc", "")

        play_urls = aweme.get("video", {}).get("play_addr", {}).get("url_list", [])
        if not play_urls:
            tqdm.write("[失败] 未找到播放地址")
            continue

        filepath = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        success = _download_video_file(play_urls[0], filepath)
        if success:
            tqdm.write(f"[完成] 视频已保存到: {filepath}")
        else:
            tqdm.write(f"[失败] 视频下载失败")
            continue

        result_list.append(item)

    with open(VIDEO_INTRO_JSON, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Step 2 完成，共处理 {len(result_list)} 条 → {VIDEO_INTRO_JSON}")


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


def step3_download_comments():
    """Step 3: 抓取每个视频的评论，保存至 TOP5_COMMENTS_JSON 和 ALL_COMMENTS_JSON"""
    print("\n" + "═"*60)
    print("  STEP 3：抓取视频评论")
    print("═"*60)

    with open(VIDEO_URL_JSON, 'r', encoding='utf-8') as f:
        video_list = json.load(f)
        if isinstance(video_list, dict):
            video_list = [video_list]

    co = ChromiumOptions()
    co.headless(True)
    co.set_user_data_path(USERDATA_DIR)
    driver = ChromiumPage(co)
    comment_api = 'aweme/v1/web/comment/list/'

    all_results = {}
    output_data = []

    for item in tqdm(video_list, desc="📦 Processing videos", unit="video"):
        vid = item['id']
        url = item['video_url']
        seen_ids = set()
        collected = []

        driver.listen.start(comment_api)
        driver.get(url)
        time.sleep(5)

        for _ in range(SCROLL_ROUNDS):
            _scroll_comment_container(driver)
            time.sleep(1.5)
            try:
                while True:
                    resp = driver.listen.wait(timeout=1.5)
                    if comment_api in resp.url:
                        data = resp.response.body
                        comments = data.get('comments', [])
                        if comments:
                            collected.extend(_collect_comments(comments, seen_ids))
                        break
            except Exception:
                continue

        driver.listen.stop()
        collected = sorted(collected, key=lambda x: x['digg_count'], reverse=True)
        all_results[vid] = collected
        top_comments = collected[:5]

        video_data = {"id": vid, "video_url": url, "video_introduction": ""}
        for i, comment in enumerate(top_comments, start=1):
            video_data[f"comment_{i}"] = comment['text']
        output_data.append(video_data)

    driver.quit()

    with open(ALL_COMMENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(TOP5_COMMENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Step 3 完成 → {ALL_COMMENTS_JSON}")
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


def _save_frames(video_path, output_dir, fps):
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
    """Step 4: 抽帧 + Whisper 转录，保存至 CHOUZHEN_JSON"""
    print("\n" + "═"*60)
    print("  STEP 4：视频抽帧 + 音频转录")
    print("═"*60)

    # 提前定位 ffmpeg，同步告知 pydub，消除 RuntimeWarning
    ffmpeg_exe = _find_ffmpeg()
    print(f"  [ffmpeg] 使用路径：{ffmpeg_exe}")

    model = whisper.load_model(WHISPER_MODEL_NAME)

    with open(VIDEO_INTRO_JSON, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    # 以文件名（id.mp4）为 key 建立映射
    label_map = {f"{item['id']}.mp4": item for item in original_data}
    # 同时支持以 video_url 为 key
    for item in original_data:
        label_map[item.get("video_url", "")] = item

    os.makedirs(IMAGE_DIR, exist_ok=True)
    videos = sorted(
        [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))],
        key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else float('inf')
    )
    print(f"  共发现 {len(videos)} 个视频文件")

    result_json = []
    for video_file in tqdm(videos, desc="📦 处理视频", unit="个"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        frame_dir = os.path.join(IMAGE_DIR, video_name, "frames")

        main_frames = _save_frames(video_path, frame_dir, FRAME_FPS)

        audio_path = _extract_audio(video_path)
        full_transcript = _transcribe(audio_path, model)
        os.remove(audio_path)

        txt_path = os.path.join(IMAGE_DIR, video_name, "transcription.txt")
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(full_transcript)

        meta = label_map.get(video_file, label_map.get(f"https://www.douyin.com/video/{video_name}", {}))
        result_json.append({
            "id": video_name,
            "video_url": video_file,
            "video_introduction": meta.get("video_introduction", ""),
            "label": meta.get("label", ""),
            "image": main_frames,
            "all_transcription": full_transcript
        })

    result_json.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else float('inf'))

    with open(CHOUZHEN_JSON, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)

    print(f"\n  ✅ Step 4 完成 → {CHOUZHEN_JSON}")


# ════════════════════════════════════════════════════════════════
#  STEP 5: GPT 生成视频描述
# ════════════════════════════════════════════════════════════════

def _encode_image_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Failed to encode image {path}: {e}")
        return None


def _detect_language(texts: list) -> str:
    combined = " ".join(texts)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', combined)
    return "zh" if len(chinese_chars) / max(len(combined), 1) > 0.2 else "en"


def _call_gpt_with_images(transcription, video_intro, frames, lang="zh",
                           max_images_per_batch=5, max_retries=3, temperature=0.3):
    full_description = ""

    if lang == "zh":
        system_prompt = (
            "你是一位视频内容叙述专家，你的任务是根据视频的关键帧图像和音频转录内容，"
            "用中文写出一段完整的故事性描述，帮助没有看过视频的读者完全理解视频讲了什么。"
            "你的描述应自然流畅、像讲故事一样，结合画面和声音的信息，真实、细腻地呈现"
            "视频中的人物、动作、场景、情节发展和情绪变化。"
        )
        text_template = (
            "以下是该视频的简介、音频转录文本和部分关键帧图像（第 {batch_idx} 批）：\n\n"
            "视频简介：{video_intro}\n\n"
            "音频转录文本：{transcription}\n\n"
            "每张图像的文件名数值越小表示越靠近视频开头。请结合图像和音频，写出自然连贯、像讲故事一样的视频内容叙述。"
        )
    else:
        system_prompt = (
            "You are a video content narration expert. Your task is to describe the video story based on "
            "the key frame images and the audio transcription. Write a complete story-like video description in English."
        )
        text_template = (
            "Below is the video's introduction, audio transcript, and some keyframe images (batch {batch_idx}):\n\n"
            "Video introduction: {video_intro}\n\n"
            "Audio transcript: {transcription}\n\n"
            "The smaller the image filename number, the earlier it appears in the video. "
            "Please write a natural, coherent, story-like description."
        )

    for batch_idx in range(0, len(frames), max_images_per_batch):
        image_batch = frames[batch_idx:batch_idx + max_images_per_batch]
        image_messages = []
        for frame_path in image_batch:
            b64 = _encode_image_to_base64(frame_path)
            if b64:
                image_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": text_template.format(
                    batch_idx=batch_idx // max_images_per_batch + 1,
                    video_intro=video_intro,
                    transcription=transcription
                )},
                *image_messages
            ]}
        ]

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        data = {"model": OPENAI_MODEL, "messages": messages, "temperature": temperature, "max_tokens": 2048}

        for attempt in range(1, max_retries + 1):
            try:
                res = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=180)
                res.raise_for_status()
                content = res.json()["choices"][0]["message"]["content"]
                full_description += content.strip() + "\n"
                break
            except Exception as e:
                print(f"⚠️ API 调用失败（批次 {batch_idx // max_images_per_batch + 1}，第 {attempt}/{max_retries} 次）: {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    print("⏭️ 已达最大重试次数，跳过此批次。")

    return full_description.strip()


def step5_generate_descriptions():
    """Step 5: 调用 GPT-4o 生成每个视频的描述，保存至 VIDEO_DESCRIPTION_JSON"""
    print("\n" + "═"*60)
    print("  STEP 5：GPT-4o 生成视频描述")
    print("═"*60)

    with open(CHOUZHEN_JSON, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    output_data = []
    for video in tqdm(input_data, desc="Processing videos"):
        video_id    = str(video.get("id", "")).strip()
        transcription = video.get("all_transcription", "")
        video_intro   = video.get("video_introduction", "No introduction provided.")
        lang = _detect_language([transcription, video_intro])

        base_path  = os.path.join(IMAGE_DIR, video_id)
        frames_dir = os.path.join(base_path, "frames") if os.path.isdir(os.path.join(base_path, "frames")) else base_path
        all_frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]) if os.path.isdir(frames_dir) else []
        frames = all_frames[::FRAME_INTERVAL]

        if not frames:
            print(f"⚠️ 视频 {video_id} 无有效帧，跳过。")
            continue

        description = _call_gpt_with_images(transcription, video_intro, frames, lang=lang)

        output_data.append({
            "id": video_id,
            "video_url": f"{video_id}.mp4",
            "video_introduction": video_intro,
            "label": video.get("label", ""),
            "all_transcription": transcription,
            "video_description": description
        })

    os.makedirs(os.path.dirname(VIDEO_DESCRIPTION_JSON), exist_ok=True)
    with open(VIDEO_DESCRIPTION_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Step 5 完成 → {VIDEO_DESCRIPTION_JSON}")


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="抖音视频全流程一键脚本",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--steps", type=str, default="1,2,3,4,5",
        help="指定要执行的步骤（逗号分隔），例如 --steps 2,3,4\n"
             "  1: 采集视频 URL\n"
             "  2: 下载视频 + 获取简介\n"
             "  3: 抓取评论\n"
             "  4: 抽帧 + 音频转录\n"
             "  5: GPT 生成视频描述\n"
             "（默认全部执行）"
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
        step1_crawl_urls()

    if 2 in steps:
        asyncio.run(step2_download_videos())

    if 3 in steps:
        step3_download_comments()

    if 4 in steps:
        step4_chouzhen()

    if 5 in steps:
        step5_generate_descriptions()

    print("\n\n" + "★"*60)
    print("  ✅ 所有指定步骤已完成！")
    print("★"*60)


if __name__ == "__main__":
    main()