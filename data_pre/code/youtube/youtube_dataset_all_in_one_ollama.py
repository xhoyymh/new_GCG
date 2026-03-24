"""
YouTube 视频数据采集与描述生成 — 一体化流水线
=================================================
流程：
  Step 1  搜索 YouTube Shorts URL 并保存
  Step 2  下载视频 + 抓取评论
  Step 3  抽帧 + Whisper 语音转录
  Step 4  调用 Ollama 多模态模型生成视频描述
  Step 5  C_label 自动标注

依赖：
  pip install google-api-python-client isodate yt-dlp opencv-python
              openai-whisper pydub tqdm ollama
  ffmpeg 需在系统 PATH 中或通过 FFMPEG_PATH 指定
"""

import json
import math as _math
import os
import re
import time
import tempfile
import subprocess
from collections import Counter, defaultdict as _defaultdict
from urllib.parse import urlparse, parse_qs

import cv2
import isodate
import whisper
from googleapiclient.discovery import build
from tqdm import tqdm
import yt_dlp
from ollama import chat

# ═══════════════════════════════════════════════════════════
#  全局配置（按需修改）
# ═══════════════════════════════════════════════════════════

API_KEY = "AIzaSyAp0cKrDn6M3--UQaSHlfJF1UcGfanWsug"

BASE_DIR     = r"D:\Desktop\video_comment_generation\ALLinone\data_pre"
JSON_DIR     = os.path.join(BASE_DIR, "json", "youtube", "data_pre")
VIDEO_DIR    = os.path.join(BASE_DIR, "video", "youtube")
IMAGE_DIR    = os.path.join(BASE_DIR, "youtube_image")

# 各步骤产出 JSON 路径
URL_JSON          = os.path.join(JSON_DIR, "youtube_video_url.json")
VIDEO_SAMPLE_JSON = os.path.join(BASE_DIR, "code", "youtube", "youtube_video_sample.json")
TOP5_COMMENT_JSON = os.path.join(JSON_DIR, "youtube_top5_comment.json")
ALL_COMMENT_JSON  = os.path.join(JSON_DIR, "youtube_all_comments.json")
CHOUZHEN_JSON     = os.path.join(JSON_DIR, "youtube_chouzhen.json")
DESCRIPTION_JSON  = os.path.join(JSON_DIR, "youtube_video_description.json")

# Step 5 输出 / Step 6 输入输出路径
SAMPLE_TRAIN_JSON = os.path.join(BASE_DIR, "json", "sample", "youtube_comments_sample.json")
YOUTUBE_SAMPLE_JSON = os.path.join(BASE_DIR, "json", "youtube", "sample", "youtube_sample.json")

# 抽帧 / Whisper / Ollama 参数
FRAMES_PER_SECOND    = 1
WHISPER_MODEL_NAME   = "base"
OLLAMA_MODEL         = "qwen3.5:latest"
MAX_IMAGES_PER_BATCH = 5
FRAME_INTERVAL       = 1   # 送入 Ollama 时每隔几帧取一张

# ffmpeg 路径（留空则自动从系统 PATH 中搜索；找不到时会报错提示）
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# ── pydub / ffmpeg 初始化（保持在配置区末尾）──────────────────────
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", RuntimeWarning)
    from pydub import AudioSegment
if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg    = FFMPEG_PATH
    AudioSegment.ffprobe   = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    _ffmpeg_bin_dir = os.path.dirname(FFMPEG_PATH)
    os.environ["PATH"] = _ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")


# ═══════════════════════════════════════════════════════════
#  共用工具
# ═══════════════════════════════════════════════════════════

def load_json_data(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════
#  Step 1  搜索 YouTube URL
# ═══════════════════════════════════════════════════════════

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
        chosen.extend([t.strip() for t in custom_input.split(",") if t.strip()])

    seen, result = set(), []
    for t in chosen:
        if t not in seen:
            seen.add(t)
            result.append(t)
    if not result:
        print("  未选择任何 Tag。")
    return result


def load_last_url_id() -> int:
    if not os.path.exists(URL_JSON):
        return 0
    with open(URL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[-1]["id"] if data else 0


def save_url_results(new_items: list):
    data = []
    if os.path.exists(URL_JSON):
        with open(URL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.extend(new_items)
    with open(URL_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def search_shorts(youtube_client, tag: str, max_minutes: int, limit: int) -> list:
    search_resp = youtube_client.search().list(
        q=tag, part="id", type="video",
        videoDuration="short", maxResults=50
    ).execute()
    video_ids = [item["id"]["videoId"] for item in search_resp["items"]]
    if not video_ids:
        return []
    videos_resp = youtube_client.videos().list(
        id=",".join(video_ids), part="contentDetails"
    ).execute()
    max_seconds = max_minutes * 60
    results = []
    for v in videos_resp["items"]:
        duration = isodate.parse_duration(v["contentDetails"]["duration"]).total_seconds()
        if duration <= max_seconds:
            results.append(v["id"])
        if len(results) >= limit:
            break
    return results


def step1_search_urls():
    print("\n" + "═" * 60)
    print("  Step 1 — 搜索 YouTube Shorts URL")
    print("═" * 60)
    os.makedirs(JSON_DIR, exist_ok=True)

    video_data = []
    if os.path.exists(VIDEO_SAMPLE_JSON):
        print(f"正在加载本地视频数据：{VIDEO_SAMPLE_JSON} …")
        video_data = load_json_data(VIDEO_SAMPLE_JSON)
        print(f"  共加载 {len(video_data)} 条视频记录。")
    else:
        print(f"  [提示] 未找到本地样本文件，将手动输入。")

    if video_data:
        labels = get_labels(video_data)
        selected_label, is_new_label = select_label(labels)
        print(f"\n已选择 Label：{selected_label}（{'全新' if is_new_label else '已有'}）")
        top_tags      = get_top_tags(video_data, selected_label) if not is_new_label else []
        selected_tags = select_tags(top_tags, is_new_label)
    else:
        selected_label = input("请输入 Label 名称：").strip()
        is_new_label   = True
        tag_input      = input("请输入搜索 Tag（多个用逗号分隔）：").strip()
        selected_tags  = [t.strip() for t in tag_input.split(",") if t.strip()]

    if not selected_tags:
        print("未选择任何 Tag，跳过 Step 1。")
        return

    max_minutes = int(input("\n请输入最大时长（分钟）："))
    limit       = int(input("请输入每个 Tag 最多下载多少条 URL："))

    youtube_client = build("youtube", "v3", developerKey=API_KEY)
    all_video_ids, seen_ids = [], set()
    for tag in selected_tags:
        # 已有 label 时用 "label tag" 组合搜索，全新 label 只用 tag
        query = f"{selected_label} {tag}" if not is_new_label else tag
        print(f"\n正在搜索 '{query}'，时长 ≤ {max_minutes} 分钟，数量限制 {limit} 条…")
        ids = search_shorts(youtube_client, query, max_minutes, limit)
        added = 0
        for vid in ids:
            if vid not in seen_ids:
                seen_ids.add(vid)
                all_video_ids.append(vid)
                added += 1
        print(f"  '{query}' 新增 {added} 条（去重后总计 {len(all_video_ids)} 条）")

    if not all_video_ids:
        print("\n未找到任何符合条件的视频。")
        return

    last_id   = load_last_url_id()
    new_items = []
    for vid in all_video_ids:
        last_id += 1
        new_items.append({
            "id":        last_id,
            "label":     selected_label,
            "video_url": f"https://www.youtube.com/watch?v={vid}",
        })
    save_url_results(new_items)
    print(f"\n✅ 已新增 {len(new_items)} 条记录，保存至 {URL_JSON}")


# ═══════════════════════════════════════════════════════════
#  Step 2  下载视频 + 抓取评论
# ═══════════════════════════════════════════════════════════

def extract_video_id(video_url: str) -> str:
    if "shorts" in video_url:
        return video_url.split("/")[-1]
    elif "watch?v=" in video_url:
        return video_url.split("v=")[-1].split("&")[0]
    else:
        parsed = urlparse(video_url)
        return parse_qs(parsed.query).get("v", [""])[0]


def download_video(url: str, save_id):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    ydl_opts = {
        "format":              "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl":             os.path.join(VIDEO_DIR, f"{save_id}.mp4"),
        "quiet":               True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def get_video_info(youtube_client, video_id: str) -> tuple:
    try:
        response = youtube_client.videos().list(part="snippet", id=video_id).execute()
        if response["items"]:
            snippet = response["items"][0]["snippet"]
            return snippet.get("description", ""), snippet.get("title", "")
    except Exception as e:
        print(f"❌ 获取视频信息失败：{video_id}", e)
    return "", ""


def get_top_comments(youtube_client, video_id: str, max_count: int = 5, max_total: int = 50000) -> tuple:
    comments, next_page_token, try_count = [], None, 0
    while True:
        try:
            response = youtube_client.commentThreads().list(
                part="snippet", videoId=video_id,
                maxResults=100, pageToken=next_page_token,
                textFormat="plainText"
            ).execute()
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text":      snippet.get("textDisplay", ""),
                    "likeCount": snippet.get("likeCount", 0),
                })
            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(comments) >= max_total:
                break
        except Exception as e:
            try_count += 1
            print(f"⚠️ 评论抓取失败尝试 {try_count} 次：{video_id}", e)
            if try_count >= 3:
                break
            time.sleep(1)
    sorted_comments = sorted(comments, key=lambda x: x["likeCount"], reverse=True)
    return sorted_comments[:max_count], sorted_comments


def step2_download_comments():
    print("\n" + "═" * 60)
    print("  Step 2 — 下载视频 & 抓取评论")
    print("═" * 60)

    if not os.path.exists(URL_JSON):
        print(f"  [错误] 未找到 URL 文件：{URL_JSON}，请先运行 Step 1。")
        return

    youtube_client        = build("youtube", "v3", developerKey=API_KEY)
    video_list            = load_json_data(URL_JSON)
    all_results           = []
    all_comments_by_video = {}

    for idx, video_data in enumerate(video_list, start=1):
        video_url = video_data["video_url"]
        video_id  = extract_video_id(video_url)
        save_id   = video_data["id"]

        print(f"\n🔄 处理视频 {idx}/{len(video_list)}：{video_id}")
        download_video(video_url, save_id)
        description, title = get_video_info(youtube_client, video_id)
        top_5, _           = get_top_comments(youtube_client, video_id)

        result = {
            "id":                 save_id,
            "video_url":          video_url,
            "video_introduction": title,
            "video_description":  description,
            "label":              video_data.get("label", ""),
        }
        for i, comment in enumerate(top_5):
            result[f"comment_{i+1}"] = comment.get("text", "")
            result[f"C{i+1}_label"]  = "待标注" if comment.get("text", "") else ""
        all_results.append(result)

        all_comments_by_video[str(idx)] = [
            {"text": c.get("text", ""), "digg_count": c.get("likeCount", 0)}
            for c in top_5
        ]

    os.makedirs(JSON_DIR, exist_ok=True)
    with open(TOP5_COMMENT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    with open(ALL_COMMENT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_comments_by_video, f, indent=4, ensure_ascii=False)
    print(f"\n✅ 主数据 → {TOP5_COMMENT_JSON}")
    print(f"✅ 评论详情 → {ALL_COMMENT_JSON}")


# ═══════════════════════════════════════════════════════════
#  Step 3  抽帧 + Whisper 转录
# ═══════════════════════════════════════════════════════════

def extract_audio(video_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = tmp.name
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-y", audio_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return audio_path


def transcribe(audio_path: str, model) -> str:
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]


def save_frames(video_path: str, output_dir: str, fps: int) -> list:
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


def step3_extract_frames_transcribe():
    print("\n" + "═" * 60)
    print("  Step 3 — 抽帧 & Whisper 语音转录")
    print("═" * 60)

    if not os.path.exists(TOP5_COMMENT_JSON):
        print(f"  [错误] 未找到评论文件：{TOP5_COMMENT_JSON}，请先运行 Step 2。")
        return

    original_data = load_json_data(TOP5_COMMENT_JSON)
    label_map     = {item["video_url"]: item for item in original_data}

    os.makedirs(IMAGE_DIR, exist_ok=True)
    model       = whisper.load_model(WHISPER_MODEL_NAME)
    result_json = []

    videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))]
    videos.sort(key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else float("inf"))

    for video_file in tqdm(videos, desc="📦 处理视频", unit="个"):
        video_path       = os.path.join(VIDEO_DIR, video_file)
        video_name       = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(IMAGE_DIR, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        frame_dir   = os.path.join(video_output_dir, "frames")
        main_frames = save_frames(video_path, frame_dir, FRAMES_PER_SECOND)

        audio_path      = extract_audio(video_path)
        full_transcript = transcribe(audio_path, model)
        os.remove(audio_path)

        txt_path = os.path.join(video_output_dir, "transcription.txt")
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(full_transcript)

        meta         = label_map.get(video_file, {})
        label        = meta.get("label", "")
        introduction = meta.get("video_introduction", "")

        result_json.append({
            "id":                 video_name,
            "video_url":          video_file,
            "video_introduction": introduction,
            "label":              label,
            "image":              main_frames,
            "all_transcription":  full_transcript,
        })

    result_json.sort(key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else float("inf"))
    os.makedirs(JSON_DIR, exist_ok=True)
    with open(CHOUZHEN_JSON, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 抽帧 & 转录完成，结果 → {CHOUZHEN_JSON}")


# ═══════════════════════════════════════════════════════════
#  Step 4  Ollama 多模态视频描述生成
# ═══════════════════════════════════════════════════════════

def detect_language(texts: list) -> str:
    combined      = " ".join(texts)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", combined)
    return "zh" if len(chinese_chars) / max(len(combined), 1) > 0.2 else "en"


def get_images_from_folder(folder_path: str) -> list:
    if not os.path.isdir(folder_path):
        return []
    return sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def call_ollama_with_images(transcription: str, video_intro: str, frames: list,
                             lang: str = "zh", max_retries: int = 3) -> str:
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
        image_batch  = frames[batch_idx : batch_idx + MAX_IMAGES_PER_BATCH]
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
                response          = chat(model=OLLAMA_MODEL, messages=messages)
                full_description += response.message.content.strip() + "\n"
                break
            except Exception as e:
                print(f"⚠️ Ollama 调用失败（batch {batch_idx // MAX_IMAGES_PER_BATCH + 1}，"
                      f"第 {attempt}/{max_retries} 次）：{e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    print("⏭️ 已达最大重试次数，跳过本批次。")
    return full_description.strip()


def step4_generate_descriptions():
    print("\n" + "═" * 60)
    print("  Step 4 — Ollama 多模态视频描述生成")
    print("═" * 60)

    if not os.path.exists(CHOUZHEN_JSON):
        print(f"  [错误] 未找到抽帧文件：{CHOUZHEN_JSON}，请先运行 Step 3。")
        return

    input_data  = load_json_data(CHOUZHEN_JSON)
    output_data = []

    for video in tqdm(input_data, desc="生成视频描述"):
        video_id      = str(video.get("id", "")).strip()
        transcription = video.get("all_transcription", "")
        video_intro   = video.get("video_introduction", "No introduction provided.")
        lang          = detect_language([transcription, video_intro])

        base_path  = os.path.join(IMAGE_DIR, video_id)
        frames_dir = (
            os.path.join(base_path, "frames")
            if os.path.isdir(os.path.join(base_path, "frames"))
            else base_path
        )
        all_frames = get_images_from_folder(frames_dir)
        frames     = all_frames[::FRAME_INTERVAL]

        if not frames:
            print(f"⚠️ 视频 {video_id} 无有效帧，跳过。（路径：{frames_dir}）")
            continue

        description = call_ollama_with_images(transcription, video_intro, frames, lang=lang)
        if not description:
            print(f"⚠️ 视频 {video_id} 描述为空，跳过。")
            continue

        output_data.append({
            "id":                 video_id,
            "video_url":          f"{video_id}.mp4",
            "video_introduction": video_intro,
            "label":              video.get("label", ""),
            "all_transcription":  transcription,
            "video_description":  description,
        })

    os.makedirs(JSON_DIR, exist_ok=True)
    with open(DESCRIPTION_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 视频描述已生成 → {DESCRIPTION_JSON}")




# ═══════════════════════════════════════════════════════════
#  Step 5  C_label 自动标注
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

# ── 5b. 情绪检测 ──────────────────────────────────────────────────
_S6_EMOTION_RULES = [
    (r"\[泪奔\]|\[哭\]|\[流泪\]|\[泣不成声\]|\[大哭\]",                              "deep_empathy"),
    (r"\[发怒\]|\[愤怒\]|\[鄙视\]",                                                        "anger"),
    (r"\[捂脸\]|\[尬笑\]|\[黑脸\]|\[白眼\]",                                            "speechless"),
    (r"哈哈|笑死|笑发财|笑喷|颠|太逗|太好笑|搞笑|好笑|绝了|\[大笑\]|\[呲牙\]|\[憨笑\]", "humor"),
    (r"\[赞\]|\[鼓掌\]|好看|牛啊|厉害|超棒|真棒|666|绝绝子",                                 "admiration"),
    (r"\[玫瑰\]|\[心\]|\[比心\]|\[爱心\]|可爱|萌|爱了",                                  "affection"),
    (r"原来|宁可|结果|所以说|说白了|分明|这哪|不过是|罢了|才发现",                                 "irony"),
    (r"有没有|请问|为什么|怎么|吗[？?]|啊[？?]",                                                   "curiosity"),
    (r"\[耶\]|\[微笑\]|不错|还行|挺好",                                                        "mild_positive"),
]

def _s6_detect_emotion(text):
    for pattern, label in _S6_EMOTION_RULES:
        if re.search(pattern, text):
            return label
    return "empty" if not text.strip() else "neutral"

# ── 5c. 关键词规则 ────────────────────────────────────────────────
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
def step5_label_comments():
    """Step 5: 对 top5 评论进行 C_label 标注，合并视频描述，写出 youtube_sample.json。
    已存在于输出文件中的 id 直接保留，不重复处理。同时检测并报告重复 id。"""
    print("\n" + "═"*60)
    print("  Step 5 — 评论 C_label 自动标注")
    print("═"*60)

    # ── 检查必要文件 ──
    for path, desc in [
        (SAMPLE_TRAIN_JSON,  "训练集 JSON（SAMPLE_TRAIN_JSON）"),
        (DESCRIPTION_JSON,   "视频描述 JSON（DESCRIPTION_JSON）"),
        (TOP5_COMMENT_JSON,  "Top5评论 JSON（TOP5_COMMENT_JSON）"),
    ]:
        if not os.path.exists(path):
            print(f"  [错误] 未找到 {desc}：{path}")
            return

    # ── 载入三个输入文件 ──
    with open(SAMPLE_TRAIN_JSON, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(DESCRIPTION_JSON, "r", encoding="utf-8") as f:
        desc_records = json.load(f)
    with open(TOP5_COMMENT_JSON, "r", encoding="utf-8") as f:
        top5_raw = json.load(f)
    top5_records = list(top5_raw.values()) if isinstance(top5_raw, dict) else top5_raw

    print(f"  训练集 {len(train_records)} 条 | "
          f"视频描述 {len(desc_records)} 条 | "
          f"Top5评论 {len(top5_records)} 条")

    # ── 重复 id 检测 ──
    desc_ids,  _ = _s6_check_dup(desc_records,  "youtube_video_description")
    top5_ids,  _ = _s6_check_dup(top5_records,  "youtube_top5_comment")
    _,         _ = _s6_check_dup(train_records, "youtube_comments_sample（训练集）")

    only_in_desc = desc_ids - top5_ids
    only_in_top5 = top5_ids - desc_ids
    if only_in_desc:
        print(f"  ⚠️  仅在视频描述中存在（无对应评论）的 id：{sorted(only_in_desc)}")
    if only_in_top5:
        print(f"  ⚠️  仅在评论文件中存在（无对应描述）的 id：{sorted(only_in_top5)}")
    if not only_in_desc and not only_in_top5:
        print("  ✅  视频描述与评论文件 id 完全对齐")

    # ── 加载已有输出，已存在的 id 直接保留 ──
    existing = _s6_load_existing(YOUTUBE_SAMPLE_JSON)
    if existing:
        print(f"\n  检测到已有输出文件，共 {len(existing)} 条，已存在 id 将直接保留。")

    # ── 构建 TF-IDF + KNN 索引 ──
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

    # ── 逐条预测 ──
    id_to_top5  = {str(r.get("id", "")): r for r in top5_records}
    new_items   = {}
    all_clabels = []
    skipped, added = 0, 0

    for vd in tqdm(desc_records, desc="标注评论"):
        vid_id      = str(vd.get("id", "")).strip()
        video_url   = vd.get("video_url", "")
        video_intro = vd.get("video_introduction", "")
        video_label = vd.get("label", "").strip()
        video_desc  = vd.get("video_description", "")

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

    # ── 合并旧数据 + 新数据，按 id 排序写出 ──
    merged = {**existing, **new_items}
    output_list = sorted(
        merged.values(),
        key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else float("inf")
    )

    os.makedirs(os.path.dirname(YOUTUBE_SAMPLE_JSON), exist_ok=True)
    with open(YOUTUBE_SAMPLE_JSON, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

    print(f"\n  新增：{added} 条 | 跳过（已存在）：{skipped} 条 | 文件总计：{len(output_list)} 条")
    print(f"  ✅ Step 5 完成 → {YOUTUBE_SAMPLE_JSON}")

    if all_clabels:
        total = len(all_clabels)
        print("\n  ── 本次新增标签分布 " + "─"*30)
        for label, cnt in Counter(all_clabels).most_common():
            bar = "█" * round(cnt/total*20)
            print(f"    {label:25s} {cnt:3d}  {bar}")

# ═══════════════════════════════════════════════════════════
#  主入口 — 交互式菜单
# ═══════════════════════════════════════════════════════════

STEP_FUNCS = {
    "1": ("搜索 YouTube Shorts URL",  step1_search_urls),
    "2": ("下载视频 & 抓取评论",       step2_download_comments),
    "3": ("抽帧 & Whisper 语音转录",   step3_extract_frames_transcribe),
    "4": ("Ollama 多模态视频描述生成", step4_generate_descriptions),
    "5": ("C_label 自动标注",          step5_label_comments),
}


def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║     YouTube 视频数据采集与描述生成 — 一体化流水线         ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for key, (desc, _) in STEP_FUNCS.items():
        print(f"║  {key}. {desc:<52}║")
    print("║  A. 顺序执行全部步骤（1 → 2 → 3 → 4 → 5）              ║")
    print("║  Q. 退出                                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    choice = input("\n请选择要执行的步骤：").strip().upper()

    if choice == "Q":
        print("已退出。")
    elif choice == "A":
        for key in ["1", "2", "3", "4", "5"]:
            STEP_FUNCS[key][1]()
    elif choice in STEP_FUNCS:
        STEP_FUNCS[choice][1]()
    else:
        print(f"无效选项：{choice}")


if __name__ == "__main__":
    main()