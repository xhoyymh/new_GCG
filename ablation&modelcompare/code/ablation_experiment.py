"""
抖音视频评论生成 — 消融实验脚本
=====================================
四类消融实验：

EXP-1：无视频简介，无RAG引导
  Phase1: Step1 → Step2(完整简介) → Step4 → Step5  （无Step3，无label）
  Phase2: 直接调用模型，根据视频内容自由生成评论（无c_label，无示例）

EXP-2：简介置空，无RAG引导
  Phase1: Step1 → Step4 → Step5 → Step3  （Step2仅下视频，简介强制为空）
  Phase2: 直接调用模型，根据视频内容自由生成评论（无c_label，无示例）

EXP-3：完整Pipeline + 同label随机模仿
  Phase1: Step1 → Step2 → Step4 → Step5 → Step3（完整）
  Phase2: 确定label后，从同label样本中【随机】抽取评论模仿（不用语义检索）

EXP-4：完整Pipeline + 跨label随机模仿
  Phase1: Step1 → Step2 → Step4 → Step5 → Step3（完整）
  Phase2: 从【非当前label】的样本中随机抽取评论模仿

多模型对比：每类实验中，Phase2分别用4个模型生成评论，
结果字段：qwen3.5_generated_comment / glm_generated_comment /
          deepseek-r1_generated_comment / minimax_generated_comment

所有四类实验结果合并到同一个文件，通过 ablation_exp_type 字段区分。

用法：
  # 运行全部四类实验（默认）
  python ablation_experiment.py

  # 只运行指定实验
  python ablation_experiment.py --exps 1 2

  # 指定用于Phase2的模型（逗号分隔）
  python ablation_experiment.py --phase2-models qwen3.5,glm

  # Phase1已完成，跳过（直接用已有的description json）
  python ablation_experiment.py --skip-phase1
"""

import os, re, json, time, random, asyncio, pickle, requests, subprocess
import tempfile, argparse, collections
from urllib.parse import quote

import cv2, whisper, numpy as np, jieba
from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from ollama import chat as ollama_chat
import ollama

# ══════════════════════════════════════════════════════════════════
#  ★ 全局路径配置
# ══════════════════════════════════════════════════════════════════

BASE_DIR = r"D:\Desktop\video_comment_generation\ALLinone"

# Phase 1 原始中间文件（正式流水线产出）
VIDEO_URL_JSON         = os.path.join(BASE_DIR, "comment_generation", "json", "douyin", "douyin_video_url.json")
VIDEO_INTRO_JSON       = os.path.join(BASE_DIR, "comment_generation", "json", "douyin", "douyin_video_introduction.json")
CHOUZHEN_JSON          = os.path.join(BASE_DIR, "comment_generation", "json", "douyin", "douyin_chouzhen.json")
VIDEO_DESCRIPTION_JSON = os.path.join(BASE_DIR, "comment_generation", "json", "douyin", "douyin_video_description.json")

VIDEO_DIR    = os.path.join(BASE_DIR, "comment_generation", "video", "douyin")
IMAGE_DIR    = os.path.join(BASE_DIR, "comment_generation", "image", "douyin")
USERDATA_DIR = os.path.join(BASE_DIR, "comment_generation", "userdata_douyin")

SAMPLE_JSON_PATH  = os.path.join(BASE_DIR, "data_pre", "code", "douyin", "douyin_video_sample.json")
LEARNING_SAMPLE   = os.path.join(BASE_DIR, "data_pre", "json", "douyin", "sample", "douyin_sample.json")
CACHE_FILE        = os.path.join(BASE_DIR, "comment_generation", "code", "douyin", "cached_samples.pkl")
HOT_MEME_FOLDER   = os.path.join(BASE_DIR, "comment_generation", "hotmeme")
LABEL_FEW_SHOT_N  = 20

# 消融实验输出目录
ABLATION_OUTPUT_DIR = os.path.join(BASE_DIR, "ablation_results")
ABLATION_ALL_FILE   = os.path.join(ABLATION_OUTPUT_DIR, "ablation_all_results.json")

# 标签
LABELS_ZH = ["搞笑短剧类", "日常生活段子类", "动物搞笑类", "幽默解说类", "脱口秀表演相声表演类", "其他"]
LABELS_EN = ["Comedy Skit", "Funny Everyday Moments", "Animal Comedy", "Humorous Commentary", "Talk Show / Crosstalk Performance", "Other"]

# 帧 & 音频
FRAME_FPS          = 1
WHISPER_MODEL_NAME = "tiny"
FFMPEG_PATH        = r"C:\ffmpeg\bin\ffmpeg.exe"
MAX_IMAGES_PER_BATCH = 5
FRAME_INTERVAL       = 1

# ══════════════════════════════════════════════════════════════════
#  ★ 模型配置
# ══════════════════════════════════════════════════════════════════

MODEL_ALIASES = {
    "qwen3.5":    "qwen3.5:latest",
    "glm":        "glm4:latest",
    "deepseek-r1": "deepseek-r1:latest",
    "minimax":    "minimax-m2.5:latest",
}
# Step3/Step5 固定用 qwen3.5（不作为消融变量）
FIXED_MODEL = "qwen3.5"

# Phase2 多模型对比（默认全部四个）
DEFAULT_PHASE2_MODELS = ["qwen3.5", "glm", "deepseek-r1", "minimax"]

def resolve_model(alias: str) -> str:
    alias = alias.strip().lower()
    if alias in MODEL_ALIASES:
        return MODEL_ALIASES[alias]
    for key in MODEL_ALIASES:
        if key in alias or alias in key:
            return MODEL_ALIASES[key]
    print(f"  [警告] 未识别模型别名 '{alias}'，回退到 qwen3.5")
    return MODEL_ALIASES["qwen3.5"]

# ══════════════════════════════════════════════════════════════════
#  pydub 初始化
# ══════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def detect_language(text: str) -> str:
    zh_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if zh_chars / max(len(text), 1) > 0.2 else "en"

def _detect_language(texts: list) -> str:
    combined = " ".join(t for t in texts if t)
    if not combined:
        return "zh"
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", combined)
    return "zh" if len(chinese_chars) / max(len(combined), 1) > 0.15 else "en"

def extract_keywords(text: str) -> set:
    return set(jieba.cut_for_search(text))

def load_existing_output(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(item["id"]): item for item in data if "id" in item}
    except Exception as e:
        print(f"  [警告] 读取已有输出文件失败：{e}")
        return {}

# ══════════════════════════════════════════════════════════════════
#  STEP 4：抽帧 + 音频转录
# ══════════════════════════════════════════════════════════════════

_ffmpeg_exe = None

def _find_ffmpeg() -> str:
    global _ffmpeg_exe
    if _ffmpeg_exe:
        return _ffmpeg_exe
    import shutil
    def _set(path):
        global _ffmpeg_exe
        _ffmpeg_exe            = path
        AudioSegment.converter = path
        AudioSegment.ffmpeg    = path
        AudioSegment.ffprobe   = path.replace("ffmpeg.exe", "ffprobe.exe")
        ffmpeg_dir = os.path.dirname(path)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return path
    if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
        return _set(FFMPEG_PATH)
    found = shutil.which("ffmpeg")
    if found:
        return _set(found)
    raise FileNotFoundError("找不到 ffmpeg，请配置 FFMPEG_PATH")

def _extract_audio(video_path):
    ffmpeg = _find_ffmpeg()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = tmp.name
    tmp.close()
    subprocess.run(
        [ffmpeg, "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-y", audio_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return audio_path

def _save_frames(video_path, output_dir, fps):
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

def step4_chouzhen(intro_json: str = None):
    """
    Step4: 抽帧 + 音频转录。
    intro_json: 视频简介来源文件（EXP-2时传入空简介版本）
    """
    in_path = intro_json or VIDEO_INTRO_JSON
    print("\n" + "═"*60)
    print("  STEP 4：视频抽帧 + 音频转录")
    print("═"*60)
    _find_ffmpeg()
    model = whisper.load_model(WHISPER_MODEL_NAME)

    with open(in_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    label_map = {f"{item['id']}.mp4": item for item in original_data}
    for item in original_data:
        label_map[item.get("video_url", "")] = item

    existing = load_existing_output(CHOUZHEN_JSON)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    videos = sorted(
        [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))],
        key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else float("inf"),
    )
    new_items: dict = {}
    skipped, added  = 0, 0
    for video_file in tqdm(videos, desc="📦 处理视频", unit="个"):
        video_name = os.path.splitext(video_file)[0]
        if video_name in existing:
            skipped += 1
            continue
        video_path = os.path.join(VIDEO_DIR, video_file)
        frame_dir  = os.path.join(IMAGE_DIR, video_name, "frames")
        main_frames = _save_frames(video_path, frame_dir, FRAME_FPS)
        audio_path      = _extract_audio(video_path)
        full_transcript = whisper.load_model(WHISPER_MODEL_NAME).transcribe(audio_path, fp16=False)["text"]
        os.remove(audio_path)
        meta = label_map.get(video_file, label_map.get(f"https://www.douyin.com/video/{video_name}", {}))
        new_items[video_name] = {
            "id": video_name, "video_url": video_file,
            "video_introduction": meta.get("video_introduction", ""),
            "label": meta.get("label", ""),
            "image": main_frames, "all_transcription": full_transcript,
        }
        added += 1
    merged      = {**existing, **new_items}
    result_json = sorted(
        merged.values(),
        key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else float("inf"),
    )
    save_json(result_json, CHOUZHEN_JSON)
    print(f"\n  新增：{added} | 跳过：{skipped}")
    print(f"  ✅ Step 4 完成 → {CHOUZHEN_JSON}")

# ══════════════════════════════════════════════════════════════════
#  STEP 5：多模态模型生成视频描述（固定使用 qwen3.5）
# ══════════════════════════════════════════════════════════════════

def _call_ollama_with_images(transcription: str, video_intro: str,
                              frames: list, lang: str = "zh",
                              max_retries: int = 3) -> str:
    ollama_model = resolve_model(FIXED_MODEL)
    full_description = ""

    if lang == "zh":
        system_prompt = (
            "你是一位视频内容叙述专家，你的任务是根据视频的关键帧图像和音频转录内容，"
            "用中文写出一段完整的故事性描述，帮助没有看过视频的读者完全理解视频讲了什么。"
            "你的描述应自然流畅、像讲故事一样，真实、细腻地呈现视频中的人物、动作、场景、情节发展和情绪变化。"
        )
        text_template = (
            "以下是该视频的简介、音频转录文本和部分关键帧图像（第 {batch_idx} 批）：\n\n"
            "视频简介：{video_intro}\n\n音频转录文本：{transcription}\n\n"
            "请结合图像和音频，写出自然连贯、像讲故事一样的视频内容叙述。"
        )
    else:
        system_prompt = (
            "You are a video content narration expert. Write a complete story-like "
            "video description in English based on the key frames and audio transcript."
        )
        text_template = (
            "Below is the video's introduction, audio transcript, and keyframe images (batch {batch_idx}):\n\n"
            "Video introduction: {video_intro}\n\nAudio transcript: {transcription}\n\n"
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
                    batch_idx=batch_idx, video_intro=video_intro, transcription=transcription,
                ),
                "images": valid_images,
            },
        ]
        for attempt in range(1, max_retries + 1):
            try:
                response = ollama_chat(model=ollama_model, messages=messages)
                full_description += response.message.content.strip() + "\n"
                break
            except Exception as e:
                print(f"  ⚠️ Ollama({ollama_model}) 调用失败（批次 {batch_idx}，第 {attempt}/{max_retries} 次）：{e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
    return full_description.strip()

def step5_generate_descriptions(input_json: str = None, output_json: str = None):
    """Step5: 生成视频描述（固定 qwen3.5，输出到指定路径）"""
    in_path  = input_json  or CHOUZHEN_JSON
    out_path = output_json or VIDEO_DESCRIPTION_JSON

    print("\n" + "═"*60)
    print(f"  STEP 5：视频描述生成  模型={FIXED_MODEL}")
    print("═"*60)

    with open(in_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    existing = load_existing_output(out_path)
    new_items: dict = {}
    skipped, added = 0, 0

    for video in tqdm(input_data, desc="Generating descriptions"):
        video_id      = str(video.get("id", "")).strip()
        transcription = video.get("all_transcription", "")
        video_intro   = video.get("video_introduction", "")

        if video_id in existing:
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
        new_items[video_id] = record
        added += 1

    merged      = {**existing, **new_items}
    result_list = sorted(
        merged.values(),
        key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else float("inf"),
    )
    save_json(result_list, out_path)
    print(f"\n  新增：{added} | 跳过：{skipped}")
    print(f"  ✅ Step 5 完成 → {out_path}")
    return out_path

# ══════════════════════════════════════════════════════════════════
#  STEP 3：LLM 自动 label 分类（固定使用 qwen3.5）
# ══════════════════════════════════════════════════════════════════

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
        result.extend(items[:n_per_label])
    return result

def _build_label_prompt(samples, target_intro, target_transcription, target_description, lang):
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
            f"类别：<类别名>\n原因：<原因>\n\n"
            f"视频简介：{target_intro}\n转录内容：{target_transcription}\n视频描述：{target_description}\n"
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
            f"Based on the above, classify the following video and give a one-sentence reason.\n"
            f"Note: Category MUST be from this list: {valid_labels_str}\n"
            f"Output ONLY in this exact format:\n"
            f"Category: <category name>\nReason: <reason>\n\n"
            f"Video Introduction: {target_intro}\nTranscription: {target_transcription}\n"
            f"Video Description: {target_description}\n"
        )
        return category_block + "\nHere are labeled examples:\n" + example_block + query_block

def _parse_label_response(response_text: str, lang: str) -> str:
    labels = LABELS_ZH if lang == "zh" else LABELS_EN
    label  = ""
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

def _classify_one_video(intro, transcription, description, samples, lang, max_retries=3):
    prompt       = _build_label_prompt(samples, intro, transcription, description, lang)
    ollama_model = resolve_model(FIXED_MODEL)
    for attempt in range(1, max_retries + 1):
        try:
            response = ollama_chat(model=ollama_model, messages=[{"role": "user", "content": prompt}])
            return _parse_label_response(response.message.content.strip(), lang)
        except Exception as e:
            print(f"    [警告] Ollama({ollama_model}) 调用失败（{attempt}/{max_retries}）：{e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    return LABELS_ZH[-1] if lang == "zh" else LABELS_EN[-1]

def step3_label_videos(input_json: str, output_json: str):
    """Step3: label 分类，读 input_json 写 output_json（固定 qwen3.5）"""
    print("\n" + "═"*60)
    print(f"  STEP 3：label 分类  模型={FIXED_MODEL}")
    print("═"*60)

    with open(input_json, "r", encoding="utf-8") as f:
        desc_list = json.load(f)

    samples = _load_fewshot_samples(SAMPLE_JSON_PATH, LABEL_FEW_SHOT_N)
    print(f"  加载 {len(samples)} 条 few-shot 示例")

    skipped, added = 0, 0
    for item in tqdm(desc_list, desc="  label 分类", unit="条"):
        if item.get("label", "").strip():
            skipped += 1
            continue
        intro         = item.get("video_introduction", "").strip()
        transcription = item.get("all_transcription", "").strip()
        description   = item.get("video_description", "").strip()
        if not intro and not transcription and not description:
            skipped += 1
            continue
        lang = _detect_language([intro, transcription, description])
        lang_samples = [s for s in samples if _detect_language([s.get("video_introduction", "")]) == lang]
        use_samples  = lang_samples if lang_samples else samples
        item["label"] = _classify_one_video(intro, transcription, description, use_samples, lang)
        added += 1

    save_json(desc_list, output_json)
    print(f"\n  新增分类：{added} | 跳过：{skipped}")
    print(f"  ✅ Step 3 完成 → {output_json}")
    return output_json

# ══════════════════════════════════════════════════════════════════
#  PHASE 2 辅助：Embedding + 样本加载
# ══════════════════════════════════════════════════════════════════

EMBED_MODEL = "qwen3-embedding:latest"

def get_embedding(text: str) -> np.ndarray:
    resp = ollama.embed(model=EMBED_MODEL, input=text)
    vec = resp.embeddings[0] if hasattr(resp, "embeddings") else resp["embeddings"][0]
    return np.array(vec, dtype=np.float32)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def build_sample_text(sample: dict) -> str:
    parts = [sample.get("video_introduction", ""), sample.get("video_description", ""),
             sample.get("all_transcription", "")]
    return " ".join(p for p in parts if p)

def get_preprocessed_samples() -> list:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = __import__("pickle").load(f)
            if data:
                print(f"✅ 加载缓存样本 from {CACHE_FILE}")
                return data
        except Exception as e:
            print(f"⚠️  缓存损坏 ({e})，重建中...")
            os.remove(CACHE_FILE)
    raw = load_json(LEARNING_SAMPLE)
    print("🔄 Computing embeddings for learning samples ...")
    for s in tqdm(raw, desc="Embedding samples", unit="sample"):
        s["_embedding"] = get_embedding(build_sample_text(s))
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        __import__("pickle").dump(raw, f)
    return raw

def build_label_index(samples: list) -> dict:
    index = collections.defaultdict(list)
    for s in samples:
        label = s.get("label") or s.get("C1_label") or ""
        if label:
            index[label].append(s)
    return index

def _get_all_comments(sample: dict) -> list:
    """从样本中提取所有评论文本"""
    results = []
    for i in range(1, 6):
        text = sample.get(f"comment_{i}", "")
        if text:
            results.append(text)
    if not results and "comments" in sample and isinstance(sample["comments"], list):
        for c in sample["comments"]:
            text = c.get("content", c.get("comment", ""))
            if text:
                results.append(text)
    return results

def _random_examples(sample_pool: list, n: int = 3) -> list:
    """从样本池中随机抽取n条评论"""
    comments = []
    pool_shuffled = random.sample(sample_pool, min(len(sample_pool), 20))
    for s in pool_shuffled:
        comments.extend(_get_all_comments(s))
        if len(comments) >= n * 3:
            break
    if not comments:
        return []
    return random.sample(comments, min(n, len(comments)))

# ══════════════════════════════════════════════════════════════════
#  PHASE 2 Prompt 构建
# ══════════════════════════════════════════════════════════════════

def build_prompt_direct(video: dict) -> str:
    """EXP-1/2 用：直接根据视频内容生成评论，无c_label，无示例"""
    desc       = video.get("video_description", "")
    intro      = video.get("video_introduction", "")
    transcript = video.get("all_transcription", "")
    lang = detect_language(desc or intro)

    if lang == "zh":
        return f"""你是一个抖音评论生成助手，擅长模仿真实用户的评论风格。

【视频信息】
- 视频介绍：{intro}
- 视频描述：{desc}
- 字幕信息：{transcript}

要求：
- 评论要自然、真实，像真人在评论区留言
- 内容必须结合本视频的 description
- 只输出一句评论，不要解释、不要加引号

评论：""".strip()
    else:
        return f"""You are a TikTok comment generator. Mimic the style of real user comments.

[Video Info]
- Introduction: {intro}
- Description: {desc}
- Transcript: {transcript}

Requirements:
- Sound natural and authentic, like a real user
- Content must relate to this video's description
- Output ONLY the comment, no explanation, no quotes

Comment:""".strip()

def build_prompt_with_examples(video: dict, c_label: str, examples: list) -> str:
    """EXP-3/4 用：有示例，但示例是随机抽取的，c_label 仅在 EXP-3 中有效"""
    desc       = video.get("video_description", "")
    intro      = video.get("video_introduction", "")
    transcript = video.get("all_transcription", "")
    lang       = detect_language(desc or intro)
    example_text = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(examples) if e)

    if lang == "zh":
        label_line = f"\n【目标评论风格】：{c_label}" if c_label else ""
        return f"""你是一个抖音评论生成助手，擅长模仿真实用户的评论风格。

【视频信息】
- 视频介绍：{intro}
- 视频描述：{desc}
- 字幕信息：{transcript}{label_line}

【参考评论示例】（模仿这些句式和风格，但内容必须结合当前视频）：
{example_text}

要求：
- 评论要自然、真实，像真人在评论区留言
- 模仿示例的句式结构和语气，但不要照抄
- 内容必须结合本视频的 description
- 只输出一句评论，不要解释、不要加引号

评论：""".strip()
    else:
        label_line = f"\n[Target comment style]: {c_label}" if c_label else ""
        return f"""You are a TikTok comment generator. Mimic the style of real user comments.

[Video Info]
- Introduction: {intro}
- Description: {desc}
- Transcript: {transcript}{label_line}

[Reference examples] (mimic the sentence structure and tone, but adapt to this video):
{example_text}

Requirements:
- Sound natural and authentic, like a real user
- Imitate the style of the examples, do NOT copy them directly
- Content must relate to this video's description
- Output ONLY the comment, no explanation, no quotes

Comment:""".strip()

# ══════════════════════════════════════════════════════════════════
#  PHASE 2 评论生成（单个模型）
# ══════════════════════════════════════════════════════════════════

def generate_comment_with_model(prompt: str, model_alias: str, max_retries: int = 3) -> str:
    ollama_model = resolve_model(model_alias)
    for attempt in range(1, max_retries + 1):
        try:
            resp = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.75, "top_p": 0.9, "repeat_penalty": 1.1, "num_predict": 512},
                think=False,
            )
            content = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
            return content.strip() if content else ""
        except Exception as e:
            print(f"    [警告] {model_alias}({ollama_model}) 失败（{attempt}/{max_retries}）：{e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    return ""

# ══════════════════════════════════════════════════════════════════
#  ★ 四类消融实验核心逻辑
# ══════════════════════════════════════════════════════════════════

# EXP metadata
EXP_META = {
    1: {
        "ablation_exp_type": "EXP-1",
        "ablation_exp_name": "无视频简介 + 直接生成",
        "ablation_exp_desc": "Phase1: Step1→Step2(完整简介)→Step4→Step5（无Step3）；Phase2: 直接调用模型生成评论，无c_label，无示例",
        "ablation_variable": "无label引导，无RAG示例",
    },
    2: {
        "ablation_exp_type": "EXP-2",
        "ablation_exp_name": "简介置空 + 直接生成",
        "ablation_exp_desc": "Phase1: Step1→Step4→Step5→Step3（Step2仅下视频，简介强制为空）；Phase2: 直接调用模型生成评论，无c_label，无示例",
        "ablation_variable": "video_introduction 置空",
    },
    3: {
        "ablation_exp_type": "EXP-3",
        "ablation_exp_name": "完整Pipeline + 同label随机模仿",
        "ablation_exp_desc": "完整Phase1；Phase2: 确定label后，从相同label样本中随机抽取评论模仿（非语义检索）",
        "ablation_variable": "语义检索→随机抽样（同label）",
    },
    4: {
        "ablation_exp_type": "EXP-4",
        "ablation_exp_name": "完整Pipeline + 跨label随机模仿",
        "ablation_exp_desc": "完整Phase1；Phase2: 从非当前label的样本中随机抽取评论模仿",
        "ablation_variable": "label分类精度验证（跨label随机）",
    },
}

def _process_video_exp1(video: dict, phase2_models: list) -> dict:
    """EXP-1: 无label，直接生成"""
    prompt = build_prompt_direct(video)
    result = dict(video)
    result.update(EXP_META[1])
    result["exp_c_label"]  = ""
    result["exp_examples"] = []
    for model in phase2_models:
        result[f"{model}_generated_comment"] = generate_comment_with_model(prompt, model)
    return result

def _process_video_exp2(video: dict, phase2_models: list) -> dict:
    """EXP-2: 简介置空，直接生成（调用时 video 的 video_introduction 已为空）"""
    prompt = build_prompt_direct(video)
    result = dict(video)
    result.update(EXP_META[2])
    result["exp_c_label"]  = ""
    result["exp_examples"] = []
    for model in phase2_models:
        result[f"{model}_generated_comment"] = generate_comment_with_model(prompt, model)
    return result

def _process_video_exp3(video: dict, label_index: dict, samples: list, phase2_models: list) -> dict:
    """EXP-3: 同label随机抽取示例"""
    video_label = video.get("label", "").strip()
    if video_label and video_label in label_index:
        pool = label_index[video_label]
    else:
        pool = samples
    examples = _random_examples(pool, n=3)
    prompt = build_prompt_with_examples(video, c_label=video_label, examples=examples)
    result = dict(video)
    result.update(EXP_META[3])
    result["exp_c_label"]  = video_label
    result["exp_examples"] = examples
    for model in phase2_models:
        result[f"{model}_generated_comment"] = generate_comment_with_model(prompt, model)
    return result

def _process_video_exp4(video: dict, label_index: dict, samples: list, phase2_models: list) -> dict:
    """EXP-4: 跨label随机抽取示例"""
    video_label = video.get("label", "").strip()
    # 排除同label的样本
    if video_label and video_label in label_index:
        cross_pool = [s for s in samples if (s.get("label") or s.get("C1_label") or "") != video_label]
    else:
        cross_pool = samples
    if not cross_pool:
        cross_pool = samples  # fallback
    examples = _random_examples(cross_pool, n=3)
    prompt = build_prompt_with_examples(video, c_label="", examples=examples)
    result = dict(video)
    result.update(EXP_META[4])
    result["exp_c_label"]  = video_label  # 记录视频本身的label，便于分析
    result["exp_examples"] = examples
    for model in phase2_models:
        result[f"{model}_generated_comment"] = generate_comment_with_model(prompt, model)
    return result

# ══════════════════════════════════════════════════════════════════
#  ★ Phase 1 流水线（各实验版本）
# ══════════════════════════════════════════════════════════════════

# EXP-1/3/4 共用的正式 description 文件路径
EXP_FULL_DESC_JSON  = os.path.join(ABLATION_OUTPUT_DIR, "exp_full_description.json")
# EXP-2 专用（简介置空版本）
EXP2_CHOUZHEN_JSON  = os.path.join(ABLATION_OUTPUT_DIR, "exp2_chouzhen_no_intro.json")
EXP2_DESC_JSON      = os.path.join(ABLATION_OUTPUT_DIR, "exp2_description_no_intro.json")

def run_phase1_for_exps(exps_to_run: set, skip_phase1: bool = False):
    """
    根据需要运行的实验集合，准备对应的 description JSON 文件。
    返回：{exp_id: description_json_path}
    """
    os.makedirs(ABLATION_OUTPUT_DIR, exist_ok=True)
    desc_paths = {}

    if skip_phase1:
        print("\n⏩ 跳过 Phase1，直接使用已有 description 文件")
        for exp_id in exps_to_run:
            if exp_id == 2:
                if os.path.exists(EXP2_DESC_JSON):
                    desc_paths[2] = EXP2_DESC_JSON
                else:
                    print(f"  [警告] EXP-2 需要 {EXP2_DESC_JSON}，未找到，将跳过 EXP-2")
            else:
                if os.path.exists(EXP_FULL_DESC_JSON):
                    desc_paths[exp_id] = EXP_FULL_DESC_JSON
                elif os.path.exists(VIDEO_DESCRIPTION_JSON):
                    desc_paths[exp_id] = VIDEO_DESCRIPTION_JSON
                else:
                    print(f"  [警告] 未找到 description 文件，将跳过 EXP-{exp_id}")
        return desc_paths

    # ── Step 4（抽帧转录）──────────────────────────────────
    # EXP-2 需要 Step2 只下视频、简介置空；其余 Step2 正常
    # Step4 和 Step2 耦合：Step4 读的是 VIDEO_INTRO_JSON
    # 为了 EXP-2，需要构造一个简介为空的版本

    need_full_step4  = bool({1, 3, 4} & exps_to_run)
    need_exp2_step4  = 2 in exps_to_run

    # ── 正式流水线（EXP-1/3/4）──
    if need_full_step4:
        print("\n[Phase1] 正式流水线 Step4（EXP-1/3/4 共用）")
        # Step4 写入全局 CHOUZHEN_JSON（如未完成）
        if not os.path.exists(CHOUZHEN_JSON):
            step4_chouzhen()
        else:
            print(f"  Step4 已存在，跳过：{CHOUZHEN_JSON}")

        print("\n[Phase1] 正式流水线 Step5（EXP-1/3/4 共用）")
        if not os.path.exists(EXP_FULL_DESC_JSON):
            step5_generate_descriptions(input_json=CHOUZHEN_JSON, output_json=EXP_FULL_DESC_JSON)
        else:
            print(f"  Step5 已存在，跳过：{EXP_FULL_DESC_JSON}")

        # EXP-3/4 需要 Step3（label分类）
        if {3, 4} & exps_to_run:
            print("\n[Phase1] Step3 label分类（EXP-3/4 需要）")
            # 检查是否已经有 label
            with open(EXP_FULL_DESC_JSON, "r", encoding="utf-8") as f:
                check = json.load(f)
            if any(not item.get("label", "").strip() for item in check):
                step3_label_videos(input_json=EXP_FULL_DESC_JSON, output_json=EXP_FULL_DESC_JSON)
            else:
                print("  Step3 已存在（label 已全部填充），跳过")

        for exp_id in {1, 3, 4} & exps_to_run:
            desc_paths[exp_id] = EXP_FULL_DESC_JSON

    # ── EXP-2 专用流水线（简介置空）──
    if need_exp2_step4:
        print("\n[Phase1 EXP-2] Step2仅下视频，简介强制置空")
        # 构造一个简介为空的 intro json（供 Step4 读取）
        if os.path.exists(VIDEO_INTRO_JSON):
            with open(VIDEO_INTRO_JSON, "r", encoding="utf-8") as f:
                orig_intro = json.load(f)
            no_intro = [{**item, "video_introduction": ""} for item in orig_intro]
        else:
            no_intro = []  # Step1/2 未完成，Step4 会从 VIDEO_DIR 扫描

        exp2_intro_json = os.path.join(ABLATION_OUTPUT_DIR, "exp2_video_introduction_no_intro.json")
        save_json(no_intro, exp2_intro_json)

        print("\n[Phase1 EXP-2] Step4（简介置空版）")
        # EXP-2 的 chouzhen 独立
        if not os.path.exists(EXP2_CHOUZHEN_JSON):
            # 临时替换全局变量
            orig_intro_json_backup = VIDEO_INTRO_JSON
            # 通过自定义参数调用 step4
            _step4_exp2(exp2_intro_json, EXP2_CHOUZHEN_JSON)
        else:
            print(f"  EXP-2 Step4 已存在，跳过：{EXP2_CHOUZHEN_JSON}")

        print("\n[Phase1 EXP-2] Step5（简介置空版）")
        if not os.path.exists(EXP2_DESC_JSON):
            step5_generate_descriptions(input_json=EXP2_CHOUZHEN_JSON, output_json=EXP2_DESC_JSON)
        else:
            print(f"  EXP-2 Step5 已存在，跳过：{EXP2_DESC_JSON}")

        print("\n[Phase1 EXP-2] Step3 label分类（简介置空版）")
        with open(EXP2_DESC_JSON, "r", encoding="utf-8") as f:
            check = json.load(f)
        if any(not item.get("label", "").strip() for item in check):
            step3_label_videos(input_json=EXP2_DESC_JSON, output_json=EXP2_DESC_JSON)
        else:
            print("  EXP-2 Step3 已存在，跳过")

        desc_paths[2] = EXP2_DESC_JSON

    return desc_paths

def _step4_exp2(intro_json: str, output_chouzhen: str):
    """EXP-2 专用的 Step4：简介来自 intro_json（已置空），写到 output_chouzhen"""
    print(f"\n" + "═"*60)
    print("  STEP 4 (EXP-2)：视频抽帧 + 音频转录（简介置空）")
    print("═"*60)
    _find_ffmpeg()
    model = whisper.load_model(WHISPER_MODEL_NAME)

    with open(intro_json, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    label_map = {f"{item['id']}.mp4": item for item in original_data}
    for item in original_data:
        label_map[item.get("video_url", "")] = item

    existing = load_existing_output(output_chouzhen) if os.path.exists(output_chouzhen) else {}
    os.makedirs(IMAGE_DIR, exist_ok=True)
    videos = sorted(
        [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))],
        key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else float("inf"),
    )
    new_items: dict = {}
    skipped, added  = 0, 0
    for video_file in tqdm(videos, desc="📦 EXP-2 处理视频", unit="个"):
        video_name = os.path.splitext(video_file)[0]
        if video_name in existing:
            skipped += 1
            continue
        video_path = os.path.join(VIDEO_DIR, video_file)
        frame_dir  = os.path.join(IMAGE_DIR, video_name, "frames")
        main_frames = _save_frames(video_path, frame_dir, FRAME_FPS)
        audio_path  = _extract_audio(video_path)
        full_transcript = model.transcribe(audio_path, fp16=False)["text"]
        os.remove(audio_path)
        meta = label_map.get(video_file, label_map.get(f"https://www.douyin.com/video/{video_name}", {}))
        new_items[video_name] = {
            "id": video_name, "video_url": video_file,
            "video_introduction": "",  # 强制置空
            "label": meta.get("label", ""),
            "image": main_frames, "all_transcription": full_transcript,
        }
        added += 1
    merged      = {**existing, **new_items}
    result_json = sorted(
        merged.values(),
        key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else float("inf"),
    )
    save_json(result_json, output_chouzhen)
    print(f"\n  新增：{added} | 跳过：{skipped}")
    print(f"  ✅ EXP-2 Step4 完成 → {output_chouzhen}")

# ══════════════════════════════════════════════════════════════════
#  ★ 消融实验主流程
# ══════════════════════════════════════════════════════════════════

def run_ablation_experiments(exps_to_run: list, phase2_models: list, skip_phase1: bool = False):
    """
    运行指定的消融实验，所有结果合并到一个 JSON 文件。

    Args:
        exps_to_run:   要运行的实验编号列表，如 [1, 2, 3, 4]
        phase2_models: Phase2 对比的模型别名列表
        skip_phase1:   是否跳过 Phase1
    """
    os.makedirs(ABLATION_OUTPUT_DIR, exist_ok=True)
    print(f"\n{'★'*60}")
    print(f"  消融实验启动")
    print(f"  实验类型：{exps_to_run}")
    print(f"  Phase2 模型：{phase2_models}")
    print(f"{'★'*60}")

    # ── Phase 1 ──────────────────────────────────────────────
    desc_paths = run_phase1_for_exps(set(exps_to_run), skip_phase1=skip_phase1)

    # ── 加载样本（EXP-3/4 需要）──────────────────────────────
    need_samples = bool({3, 4} & set(exps_to_run))
    samples     = []
    label_index = {}
    if need_samples:
        print("\n🔄 加载学习样本（EXP-3/4 需要）...")
        samples     = get_preprocessed_samples()
        label_index = build_label_index(samples)
        print(f"  共 {len(samples)} 条样本，{len(label_index)} 个 label")

    # ── Phase 2：逐实验处理 ──────────────────────────────────
    all_results = []  # 所有实验结果合并到此

    for exp_id in exps_to_run:
        if exp_id not in desc_paths:
            print(f"\n  ⚠️ EXP-{exp_id} 缺少 description 文件，跳过")
            continue

        desc_json  = desc_paths[exp_id]
        meta_info  = EXP_META[exp_id]
        exp_type   = meta_info["ablation_exp_type"]

        print(f"\n\n{'▶'*60}")
        print(f"  {exp_type}：{meta_info['ablation_exp_name']}")
        print(f"  描述：{meta_info['ablation_exp_desc']}")
        print(f"{'▶'*60}")

        video_data = load_json(desc_json)
        print(f"  共 {len(video_data)} 个视频")

        for video in tqdm(video_data, desc=f"  {exp_type} Phase2", unit="video"):
            if exp_id == 1:
                result = _process_video_exp1(video, phase2_models)
            elif exp_id == 2:
                result = _process_video_exp2(video, phase2_models)
            elif exp_id == 3:
                result = _process_video_exp3(video, label_index, samples, phase2_models)
            elif exp_id == 4:
                result = _process_video_exp4(video, label_index, samples, phase2_models)
            else:
                continue

            # 记录哪些模型参与了对比
            result["phase2_models_used"] = phase2_models
            all_results.append(result)

            # 打印进度
            first_model = phase2_models[0]
            comment_preview = result.get(f"{first_model}_generated_comment", "")[:50]
            tqdm.write(f"    [{exp_type}] label={result.get('exp_c_label', '-')} | {first_model}: {comment_preview}...")

    # ── 保存合并结果 ──────────────────────────────────────────
    save_json(all_results, ABLATION_ALL_FILE)
    print(f"\n\n{'★'*60}")
    print(f"  ✅ 消融实验完成！共 {len(all_results)} 条记录")
    print(f"  合并结果文件：{ABLATION_ALL_FILE}")
    print(f"{'★'*60}")

    # 打印各实验统计
    from collections import Counter
    exp_counts = Counter(r["ablation_exp_type"] for r in all_results)
    print("\n  各实验记录数：")
    for exp_type, cnt in sorted(exp_counts.items()):
        print(f"    {exp_type}: {cnt} 条")

    # 输出字段说明
    print("\n  输出字段说明：")
    print("    ablation_exp_type  — 实验类型（EXP-1/2/3/4）")
    print("    ablation_exp_name  — 实验名称")
    print("    ablation_exp_desc  — 实验描述")
    print("    ablation_variable  — 消融的变量")
    print("    exp_c_label        — 评论类型（EXP-1/2 为空）")
    print("    exp_examples       — 传入 prompt 的示例评论列表")
    print("    phase2_models_used — 参与对比的模型列表")
    for m in phase2_models:
        print(f"    {m}_generated_comment  — {m} 生成的评论")

    return all_results

# ══════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="抖音视频评论生成 — 消融实验脚本",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exps", type=int, nargs="+", default=[1, 2, 3, 4],
        metavar="N",
        help="要运行的实验编号（空格分隔）。默认：1 2 3 4\n"
             "  1: 无视频简介+直接生成\n"
             "  2: 简介置空+直接生成\n"
             "  3: 完整Pipeline+同label随机模仿\n"
             "  4: 完整Pipeline+跨label随机模仿"
    )
    parser.add_argument(
        "--phase2-models", type=str,
        default=",".join(DEFAULT_PHASE2_MODELS),
        metavar="M1,M2,...",
        help=f"Phase2 评论生成对比的模型（逗号分隔）。\n"
             f"可选：qwen3.5,glm,deepseek-r1,minimax\n"
             f"默认：{','.join(DEFAULT_PHASE2_MODELS)}"
    )
    parser.add_argument(
        "--skip-phase1", action="store_true",
        help="跳过 Phase1（假设 description json 已存在）"
    )
    args = parser.parse_args()

    # 解析模型列表
    phase2_models = [m.strip() for m in args.phase2_models.split(",") if m.strip() in MODEL_ALIASES]
    if not phase2_models:
        print(f"  [警告] 未识别模型，使用默认：{DEFAULT_PHASE2_MODELS}")
        phase2_models = DEFAULT_PHASE2_MODELS

    # 验证实验编号
    valid_exps = [e for e in args.exps if e in EXP_META]
    if not valid_exps:
        print("  [错误] 无有效实验编号（1-4）")
        return

    run_ablation_experiments(
        exps_to_run=valid_exps,
        phase2_models=phase2_models,
        skip_phase1=args.skip_phase1,
    )


if __name__ == "__main__":
    main()