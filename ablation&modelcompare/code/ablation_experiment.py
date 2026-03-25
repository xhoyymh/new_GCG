"""
抖音 / YouTube 视频评论生成 — 消融实验脚本
============================================

运行时首先选择平台（douyin / youtube），再执行四种消融实验。

  EXP-1  有简介 + 直接生成（无任何示例/RAG）
         Pipeline: Step1 → Step2（含完整简介）→ Step4 → Step5
                 → Phase2：直接调用模型，不传 c_label，不传示例
         目的：验证 RAG 示例检索对评论质量的整体贡献

  EXP-2  无简介 + 直接生成（无任何示例/RAG）
         Pipeline: Step1 → Step2（简介强制置空）→ Step4 → Step5 → Step3
                 → Phase2：直接调用模型，不传示例
         目的：验证 video_introduction 的独立贡献（与EXP-1对比）

  EXP-3  有完整 Pipeline + 同 label 随机模仿
         Pipeline: Step1 → Step2 → Step4 → Step5 → Step3
                 → Phase2：在相同 label 样本中随机抽评论作示例，不确定 c_label
         目的：验证语义检索 top-k 相似性（vs 同 label 随机示例）

  EXP-4  有完整 Pipeline + 跨 label 随机模仿
         Pipeline: Step1 → Step2 → Step4 → Step5 → Step3
                 → Phase2：随机抽取不同 label 评论作示例
         目的：验证 label 分类精度（Step3）的影响

多模型对比：
  每种消融实验，用 qwen3.5 / glm / deepseek-r1 / llama 分别生成评论，
  字段命名为 {model_alias}_generated_comment，全部写入同一条记录。

平台差异：
  douyin  — Playwright 拦截 aweme_detail，直链下载 mp4，简介来自 aweme.desc
  youtube — yt-dlp 下载，YouTube Data API v3 获取标题(作为简介)和描述

输出：
  ablation_results/{platform}/
    ablation_exp_1.json / _2 / _3 / _4   ← 各类单独结果
    ablation_all_results.json             ← 四类合并（ablation_exp_type 字段区分）
    ablation_summary.json                 ← 实验配置 & 元数据 & 字段说明

用法：
  # 交互式选择平台（推荐）
  python ablation_experiment.py

  # 直接指定平台
  python ablation_experiment.py --platform douyin
  python ablation_experiment.py --platform youtube

  # 只运行部分实验 / 部分模型
  python ablation_experiment.py --platform youtube --exp 1 3 --models qwen3.5 glm

  # Phase1 已完成，跳过
  python ablation_experiment.py --platform douyin --skip-phase1
"""

import os, re, json, time, random, asyncio, pickle, copy
import subprocess, tempfile, argparse, collections, requests
from urllib.parse import urlparse, parse_qs

import cv2, whisper, numpy as np, jieba
from tqdm import tqdm
from bs4 import BeautifulSoup
from ollama import chat as ollama_chat
import ollama


# ════════════════════════════════════════════════════════════════
#  ★ 全局基础配置（两个平台共用）
# ════════════════════════════════════════════════════════════════

BASE_DIR    = r"D:\Desktop\video_comment_generation\ALLinone"
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# YouTube Data API Key（仅 YouTube 平台使用）
YOUTUBE_API_KEY = "AIzaSyAp0cKrDn6M3--UQaSHlfJF1UcGfanWsug"

# ── 模型配置 ──────────────────────────────────────────────────────
MODEL_ALIASES = {
    "qwen3.5":     "qwen3.5:latest",
    "glm":         "glm-4.7-flash:latest",
    "deepseek-r1": "deepseek-r1:8b",
    "llama":     "llama4:latest",
}
ALL_MODELS  = list(MODEL_ALIASES.keys())
STEP3_MODEL = "qwen3.5"   # Step3/Step5 固定，不作为消融变量
STEP5_MODEL = "qwen3.5"
EMBED_MODEL = "qwen3-embedding:latest"

# ── Phase 1 参数 ───────────────────────────────────────────────
LABELS_ZH = ["搞笑短剧类", "日常生活段子类", "动物搞笑类", "幽默解说类", "脱口秀表演相声表演类", "其他"]
LABELS_EN = ["Comedy Skit", "Funny Everyday Moments", "Animal Comedy",
             "Humorous Commentary", "Talk Show / Crosstalk Performance", "Other"]
LABEL_FEW_SHOT_N     = 20
FRAME_FPS            = 1
WHISPER_MODEL_NAME   = "tiny"
MAX_IMAGES_PER_BATCH = 5
FRAME_INTERVAL       = 1

# ── Phase 2 生成参数 ──────────────────────────────────────────
GEN_OPTIONS = {
    "temperature":       0.75,
    "top_p":             0.9,
    "top_k":             40,
    "repeat_penalty":    1.1,
    "num_predict":       512,
    "mirostat":          0,
    "tfs_z":             1.0,
    "typical_p":         1.0,
    "presence_penalty":  0.0,
    "frequency_penalty": 0.0,
}

# ── 全局平台配置（运行时由 build_platform_cfg() 填充）──────────
CFG: dict = {}


# ════════════════════════════════════════════════════════════════
#  ★ 平台配置工厂
# ════════════════════════════════════════════════════════════════

def build_platform_cfg(platform: str) -> dict:
    """根据平台名生成全部路径配置。"""
    p = platform.lower()
    assert p in ("douyin", "youtube"), f"平台必须为 douyin 或 youtube，得到 '{platform}'"
    jd = os.path.join(BASE_DIR, "comment_generation", "json", p)
    return {
        "platform":         p,
        # Phase 1 JSON
        "video_url_json":   os.path.join(jd, f"{p}_video_url.json"),
        "video_intro_json": os.path.join(jd, f"{p}_video_introduction.json"),
        "chouzhen_json":    os.path.join(jd, f"{p}_chouzhen.json"),
        "video_desc_json":  os.path.join(jd, f"{p}_video_description.json"),
        # 媒体文件目录
        "video_dir":        os.path.join(BASE_DIR, "comment_generation", "video", p),
        "image_dir":        os.path.join(BASE_DIR, "comment_generation", "image", p),
        # 学习样本（优先使用 sample_raw_json，其次 sample_json）
        "sample_json":      os.path.join(BASE_DIR, "data_pre", "code", p, f"{p}_video_sample.json"),
        "sample_raw_json":  os.path.join(BASE_DIR, "data_pre", "json", p, "sample", f"{p}_sample.json"),
        "sample_cache":     os.path.join(BASE_DIR, "comment_generation", "code", p, "cached_samples.pkl"),
        # 热梗缓存（两平台共享同一目录）
        "hot_meme_folder":  os.path.join(BASE_DIR, "comment_generation", "hotmeme"),
        # 消融结果输出
        "ablation_dir":     os.path.join(BASE_DIR, "ablation&modelcompare", "json", "ablation_results", p),
    }


# ════════════════════════════════════════════════════════════════
#  ★ 四类消融实验元数据
# ════════════════════════════════════════════════════════════════

EXP_META = {
    "1": {
        "id": "EXP-1", "name": "有简介 + 直接生成",
        "desc": "Step2 正常获取简介；Phase2 直接调用模型，不传 c_label，不传任何示例",
        "pipeline": "Step1→Step2(含简介)→Step4→Step5→Phase2(直接生成,无示例)",
        "variable": "RAG 示例检索对评论质量的整体贡献",
        "phase2_mode": "direct",
    },
    "2": {
        "id": "EXP-2", "name": "无简介 + 直接生成",
        "desc": "Step2 简介强制置空；Phase2 直接调用模型，不传示例",
        "pipeline": "Step1→Step2(简介置空)→Step4→Step5→Step3→Phase2(直接生成,无示例)",
        "variable": "video_introduction 的独立贡献（与EXP-1对比）",
        "phase2_mode": "direct",
    },
    "3": {
        "id": "EXP-3", "name": "同 label 随机模仿",
        "desc": "完整 Phase1；Phase2 在相同 label 样本中随机抽评论作示例，不确定 c_label",
        "pipeline": "Step1→Step2(含简介)→Step4→Step5→Step3→Phase2(同label随机示例)",
        "variable": "语义检索 top-k 相似性 vs 同 label 随机抽取",
        "phase2_mode": "same_label_random",
    },
    "4": {
        "id": "EXP-4", "name": "跨 label 随机模仿",
        "desc": "完整 Phase1；Phase2 随机抽取不同 label 的评论作示例",
        "pipeline": "Step1→Step2(含简介)→Step4→Step5→Step3→Phase2(跨label随机示例)",
        "variable": "label 分类精度（Step3）的影响",
        "phase2_mode": "diff_label_random",
    },
}


# ════════════════════════════════════════════════════════════════
#  pydub / ffmpeg 初始化
# ════════════════════════════════════════════════════════════════

import warnings as _w
with _w.catch_warnings():
    _w.simplefilter("ignore")
    from pydub import AudioSegment

if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg    = FFMPEG_PATH
    AudioSegment.ffprobe   = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")


# ════════════════════════════════════════════════════════════════
#  公共工具函数
# ════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_existing(path) -> dict:
    if not os.path.exists(path): return {}
    try:
        return {str(x["id"]): x for x in load_json(path) if "id" in x}
    except Exception:
        return {}

def detect_lang(texts) -> str:
    text = " ".join(t for t in (texts if isinstance(texts, list) else [texts]) if t)
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if zh / max(len(text), 1) > 0.15 else "en"

def detect_language(text: str) -> str:
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return "zh" if zh / max(len(text), 1) > 0.2 else "en"

def resolve_model(alias: str) -> str:
    a = alias.strip().lower()
    if a in MODEL_ALIASES: return MODEL_ALIASES[a]
    for k in MODEL_ALIASES:
        if k in a or a in k: return MODEL_ALIASES[k]
    print(f"  [警告] 未识别模型 '{alias}'，回退 qwen3.5")
    return MODEL_ALIASES["qwen3.5"]

def extract_keywords(text: str) -> set:
    return set(jieba.cut_for_search(text))

def build_sample_text(s: dict) -> str:
    return " ".join(filter(None, [
        s.get("video_introduction",""),
        s.get("video_description",""),
        s.get("all_transcription",""),
    ]))

def _sort_by_id(lst):
    return sorted(lst, key=lambda x: int(x["id"]) if str(x.get("id","")).isdigit() else float("inf"))


# ════════════════════════════════════════════════════════════════
#  ffmpeg 工具（两平台共用）
# ════════════════════════════════════════════════════════════════

_ffmpeg_resolved = None

def _ffmpeg() -> str:
    global _ffmpeg_resolved
    if _ffmpeg_resolved: return _ffmpeg_resolved
    import shutil
    def _set(p):
        global _ffmpeg_resolved
        _ffmpeg_resolved       = p
        AudioSegment.converter = p
        AudioSegment.ffmpeg    = p
        AudioSegment.ffprobe   = p.replace("ffmpeg.exe","ffprobe.exe")
        d = os.path.dirname(p)
        if d not in os.environ.get("PATH",""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH","")
        return p
    if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH): return _set(FFMPEG_PATH)
    f = shutil.which("ffmpeg")
    if f: return _set(f)
    for candidate in [r"C:\ffmpeg\bin\ffmpeg.exe",
                      r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
        if os.path.isfile(candidate): return _set(candidate)
    raise FileNotFoundError("找不到 ffmpeg，请配置 FFMPEG_PATH")

def _extract_audio(video_path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
    subprocess.run([_ffmpeg(), "-i", video_path, "-vn", "-acodec","pcm_s16le",
                    "-ar","16000", "-y", tmp.name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name

def _save_frames(video_path, out_dir, fps):
    cap = cv2.VideoCapture(video_path)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(vfps / fps), 1)
    fid = saved = 0; paths = []
    os.makedirs(out_dir, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if fid % interval == 0:
            p = os.path.join(out_dir, f"{saved+1}.jpg")
            cv2.imwrite(p, frame)
            paths.append(os.path.abspath(p))
            saved += 1
        fid += 1
    cap.release()
    return paths


# ════════════════════════════════════════════════════════════════
#  平台专属：抖音（Playwright）
# ════════════════════════════════════════════════════════════════

async def _get_aweme_detail(share_url: str):
    """抖音：Playwright 拦截 aweme_detail API 响应。"""
    from playwright.async_api import async_playwright
    aweme = None
    found = asyncio.Event()

    async def intercept(resp):
        nonlocal aweme
        try:
            if "application/json" in resp.headers.get("content-type","").lower():
                body = await resp.json()
                if isinstance(body, dict) and "aweme_detail" in body:
                    aweme = body["aweme_detail"]; found.set()
        except Exception: pass

    async with async_playwright() as p:
        br  = await p.chromium.launch(headless=True)
        ctx = await br.new_context()
        pg  = await ctx.new_page()
        pg.on("response", intercept)
        task = asyncio.create_task(
            pg.goto(share_url, wait_until="networkidle", timeout=60000)
        )
        try: await asyncio.wait_for(found.wait(), timeout=15)
        except asyncio.TimeoutError: pass
        if not task.done(): task.cancel()
        for x in (pg, ctx, br):
            try: await x.close()
            except: pass
    return aweme

def _dl_direct(url, path) -> bool:
    """抖音：直链下载 mp4。"""
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, stream=True, timeout=30)
        with open(path,"wb") as f:
            for chunk in r.iter_content(8192):
                if chunk: f.write(chunk)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception as e:
        print(f"  [下载错误] {e}"); return False


# ════════════════════════════════════════════════════════════════
#  平台专属：YouTube（yt-dlp + Data API）
# ════════════════════════════════════════════════════════════════

def _extract_yt_video_id(url: str) -> str:
    if "shorts/" in url: return url.split("shorts/")[-1].split("?")[0]
    if "watch?v=" in url: return url.split("v=")[-1].split("&")[0]
    return parse_qs(urlparse(url).query).get("v",[""])[0]

def _get_yt_video_info(video_id: str) -> tuple:
    """YouTube Data API 获取 (title, description)。"""
    try:
        from googleapiclient.discovery import build as gapi_build
        yt   = gapi_build("youtube","v3",developerKey=YOUTUBE_API_KEY)
        resp = yt.videos().list(part="snippet", id=video_id).execute()
        if resp.get("items"):
            sn = resp["items"][0]["snippet"]
            return sn.get("title",""), sn.get("description","")
    except Exception as e:
        print(f"  [警告] YouTube API 获取失败：{e}")
    return "", ""

def _dl_ytdlp(video_url: str, save_path: str) -> bool:
    """YouTube：yt-dlp 下载视频。"""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({
            "outtmpl": save_path, "format": "bv*+ba/b",
            "merge_output_format": "mp4", "quiet": True, "noprogress": True,
        }) as ydl:
            ydl.download([video_url])
        return True
    except Exception as e:
        print(f"  [下载失败] {e}"); return False


# ════════════════════════════════════════════════════════════════
#  PHASE 1 — Step 2（两平台统一入口）
# ════════════════════════════════════════════════════════════════

async def step2_download(mode: str = "full", out_json: str = None):
    """
    Step 2: 下载视频 + 获取简介。

    mode:
      "full"     → 正常获取 video_introduction（标准流程）
      "no_intro" → 仅下载视频，video_introduction 强制置空（EXP-2 专用）

    平台差异：
      douyin  → Playwright 拦截 aweme_detail；直链下载 mp4；简介来自 aweme.desc
      youtube → yt-dlp 下载；YouTube Data API 获取 title（作为简介）
    """
    dst      = out_json or CFG["video_intro_json"]
    platform = CFG["platform"]
    tag = "简介置空 [EXP-2]" if mode == "no_intro" else "含完整简介 [标准]"
    print(f"\n{'═'*60}\n  STEP 2 — {platform.upper()} — {tag}\n{'═'*60}")

    os.makedirs(CFG["video_dir"], exist_ok=True)
    video_list = load_json(CFG["video_url_json"])
    existing   = load_existing(dst)
    added = skipped = failed = 0
    new_items = {}

    for item in tqdm(video_list, desc=f"Step2 ({mode})"):
        vid  = str(item.get("id","")).strip()
        vurl = item.get("video_url","")
        if vid in existing:
            skipped += 1; continue

        fp = os.path.join(CFG["video_dir"], f"{vid}.mp4")

        # ── 抖音 ────────────────────────────────────────────────
        if platform == "douyin":
            aweme = await _get_aweme_detail(vurl)
            if not aweme: failed += 1; continue
            play_urls = aweme.get("video",{}).get("play_addr",{}).get("url_list",[])
            if not play_urls: failed += 1; continue
            if not os.path.exists(fp):
                if not _dl_direct(play_urls[0], fp): failed += 1; continue
            intro = "" if mode == "no_intro" else aweme.get("desc","")

        # ── YouTube ─────────────────────────────────────────────
        else:
            if not os.path.exists(fp):
                if not _dl_ytdlp(vurl, fp): failed += 1; continue
            if mode == "no_intro":
                intro = ""
            else:
                yt_id = _extract_yt_video_id(vurl)
                title, _ = _get_yt_video_info(yt_id) if yt_id else ("","")
                intro = title   # YouTube 用 title 作为 video_introduction

        r = dict(item)
        r["video_introduction"] = intro
        new_items[vid] = r
        added += 1

    merged = {**existing, **new_items}
    save_json(_sort_by_id(list(merged.values())), dst)
    print(f"  新增:{added} | 跳过:{skipped} | 失败:{failed} → {dst}")
    return dst


# ════════════════════════════════════════════════════════════════
#  PHASE 1 — Step 4: 抽帧 + 转录（两平台完全相同）
# ════════════════════════════════════════════════════════════════

def step4_chouzhen(intro_json: str = None):
    in_path = intro_json or CFG["video_intro_json"]
    print(f"\n{'═'*60}\n  STEP 4：抽帧 + 转录（{CFG['platform'].upper()}）\n{'═'*60}")
    _ffmpeg()
    wmodel = whisper.load_model(WHISPER_MODEL_NAME)

    data     = load_json(in_path)
    meta_map = {f"{x['id']}.mp4": x for x in data}

    existing = load_existing(CFG["chouzhen_json"])
    os.makedirs(CFG["image_dir"], exist_ok=True)
    videos = sorted(
        [f for f in os.listdir(CFG["video_dir"]) if f.lower().endswith((".mp4",".mov",".avi"))],
        key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else float("inf")
    )

    new_items = {}; added = skipped = 0
    for vf in tqdm(videos, desc="Step4"):
        vid = os.path.splitext(vf)[0]
        if vid in existing: skipped += 1; continue

        vpath  = os.path.join(CFG["video_dir"], vf)
        frames = _save_frames(vpath, os.path.join(CFG["image_dir"], vid, "frames"), FRAME_FPS)
        audio  = _extract_audio(vpath)
        transcript = wmodel.transcribe(audio, fp16=False)["text"]
        os.remove(audio)

        # 同步保存转录文本（YouTube 脚本原版行为）
        tr_dir = os.path.join(CFG["image_dir"], vid)
        os.makedirs(tr_dir, exist_ok=True)
        with open(os.path.join(tr_dir,"transcription.txt"),"w",encoding="utf-8") as tf:
            tf.write(transcript)

        m = meta_map.get(vf, {})
        new_items[vid] = {
            "id": vid, "video_url": vf,
            "video_introduction": m.get("video_introduction",""),
            "label": m.get("label",""),
            "image": frames, "all_transcription": transcript,
        }
        added += 1

    merged = {**existing, **new_items}
    save_json(_sort_by_id(list(merged.values())), CFG["chouzhen_json"])
    print(f"  新增:{added} | 跳过:{skipped} → {CFG['chouzhen_json']}")


# ════════════════════════════════════════════════════════════════
#  PHASE 1 — Step 5: 视频描述生成（两平台相同，prompt 标注平台）
# ════════════════════════════════════════════════════════════════

def _call_step5(transcription, video_intro, frames, lang, retries=3):
    model = resolve_model(STEP5_MODEL)
    pname = {"douyin":"抖音","youtube":"YouTube"}.get(CFG["platform"], CFG["platform"])
    desc  = ""

    if lang == "zh":
        sys_p = (f"你是一位视频内容叙述专家，根据{pname}视频的关键帧和音频转录，"
                 "用中文写出完整、故事性的视频描述，帮助没看过视频的读者理解内容。")
        tmpl  = ("视频简介：{intro}\n音频转录：{tr}\n（第{b}批关键帧，文件名越小越靠近视频开头）\n"
                 "请写出自然连贯的视频内容叙述。")
    else:
        sys_p = (f"You are a video narration expert. Describe the {pname} video story "
                 "based on keyframes and transcript, so readers can fully understand it.")
        tmpl  = ("Introduction: {intro}\nTranscript: {tr}\n(Batch {b}; smaller filename = earlier)\n"
                 "Write a coherent, story-like description.")

    for start in range(0, len(frames), MAX_IMAGES_PER_BATCH):
        imgs = [p for p in frames[start:start+MAX_IMAGES_PER_BATCH] if os.path.isfile(p)]
        if not imgs: continue
        msg  = tmpl.format(intro=video_intro, tr=transcription, b=start//MAX_IMAGES_PER_BATCH+1)
        for attempt in range(1, retries+1):
            try:
                r = ollama_chat(model=model, messages=[
                    {"role":"system","content":sys_p},
                    {"role":"user","content":msg,"images":imgs},
                ])
                desc += r.message.content.strip() + "\n"; break
            except Exception:
                if attempt < retries: time.sleep(2*attempt)
    return desc.strip()

def step5_descriptions(chouzhen_json: str = None, out_json: str = None):
    in_path  = chouzhen_json or CFG["chouzhen_json"]
    out_path = out_json or CFG["video_desc_json"]
    print(f"\n{'═'*60}\n  STEP 5：视频描述生成（{CFG['platform'].upper()}，模型：{STEP5_MODEL}）\n{'═'*60}")

    data = load_json(in_path)
    existing = load_existing(out_path)
    new_items = {}; added = skipped = 0

    for v in tqdm(data, desc="Step5"):
        vid = str(v.get("id","")).strip()
        if vid in existing: skipped += 1; continue
        frames = v.get("image",[])
        lang   = detect_lang([v.get("all_transcription",""), v.get("video_introduction","")])
        desc   = _call_step5(v.get("all_transcription",""), v.get("video_introduction",""),
                             frames[::FRAME_INTERVAL], lang)
        r = dict(v)
        r["video_description"] = desc
        r.pop("image",None); r.pop("all_transcription",None)
        new_items[vid] = r; added += 1

    merged = {**existing, **new_items}
    save_json(_sort_by_id(list(merged.values())), out_path)
    print(f"  新增:{added} | 跳过:{skipped} → {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════════
#  PHASE 1 — Step 3: label 分类（两平台完全相同）
# ════════════════════════════════════════════════════════════════

def _load_fewshot(n) -> list:
    # 优先使用 sample_json（few-shot 专用），fallback 到 sample_raw_json
    for key in ("sample_json","sample_raw_json"):
        path = CFG.get(key,"")
        if os.path.exists(path):
            data = load_json(path)
            buckets = {}
            for x in data:
                lb = x.get("label","").strip()
                if lb and x.get("video_introduction","").strip():
                    buckets.setdefault(lb,[]).append(x)
            out = []
            for items in buckets.values(): out.extend(items[:n])
            return out
    return []

def _label_prompt(samples, intro, transcript, description, lang) -> str:
    if lang == "zh":
        cats = "\n".join(f"{i+1}. {l}" for i,l in enumerate(LABELS_ZH))
        exs  = "".join(
            f"视频简介：{s.get('video_introduction','')}\n"
            f"转录内容：{s.get('all_transcription',s.get('video_description',''))}\n"
            f"类别：{s.get('label','')}\n\n" for s in samples
        )
        valid = "、".join(LABELS_ZH)
        return (f"视频分类：\n{cats}\n\n已标注示例：\n{exs}"
                f"请为下方视频判断类别（必须从此列表选一个）：{valid}\n"
                f"格式：类别：<类别名>\n原因：<原因>\n\n"
                f"视频简介：{intro}\n转录内容：{transcript}\n视频描述：{description}")
    else:
        cats = "\n".join(f"{i+1}. {l}" for i,l in enumerate(LABELS_EN))
        exs  = "".join(
            f"Introduction: {s.get('video_introduction','')}\n"
            f"Transcription: {s.get('all_transcription',s.get('video_description',''))}\n"
            f"Category: {s.get('label','')}\n\n" for s in samples
        )
        valid = ", ".join(LABELS_EN)
        return (f"Categories:\n{cats}\n\nLabeled examples:\n{exs}"
                f"Classify (choose from: {valid}):\n"
                f"Output: Category: <n>\nReason: <reason>\n\n"
                f"Introduction: {intro}\nTranscription: {transcript}\nDescription: {description}")

def _parse_label(text, lang) -> str:
    labels = LABELS_ZH if lang == "zh" else LABELS_EN
    prefix = "类别" if lang == "zh" else "Category"
    m = re.search(rf"{prefix}[：:]\s*(.+)", text)
    label = m.group(1).strip().rstrip("。．.") if m else ""
    if label not in labels:
        for l in labels:
            if l in text: return l
        return labels[-1]
    return label

def step3_label(in_json: str = None, out_json: str = None):
    in_path  = in_json  or CFG["video_desc_json"]
    out_path = out_json or CFG["video_desc_json"]
    model    = resolve_model(STEP3_MODEL)
    print(f"\n{'═'*60}\n  STEP 3：label 分类（{CFG['platform'].upper()}，模型：{STEP3_MODEL}）\n{'═'*60}")

    if not os.path.exists(in_path):
        print(f"  [错误] 文件不存在：{in_path}"); return

    data    = load_json(in_path)
    samples = _load_fewshot(LABEL_FEW_SHOT_N)
    added = skipped = 0

    for item in tqdm(data, desc="Step3"):
        if item.get("label","").strip(): skipped += 1; continue
        intro = item.get("video_introduction","")
        tr    = item.get("all_transcription","")
        desc  = item.get("video_description","")
        if not (intro or tr or desc): skipped += 1; continue

        lang   = detect_lang([intro, tr, desc])
        prompt = _label_prompt(
            [s for s in samples if detect_lang([s.get("video_introduction","")])==lang] or samples,
            intro, tr, desc, lang
        )
        for attempt in range(1, 4):
            try:
                r = ollama_chat(model=model, messages=[{"role":"user","content":prompt}])
                item["label"] = _parse_label(r.message.content.strip(), lang); break
            except Exception:
                if attempt < 3: time.sleep(2*attempt)
                else: item["label"] = LABELS_ZH[-1] if lang=="zh" else LABELS_EN[-1]
        added += 1

    save_json(data, out_path)

    # 同步回写 intro_json 和 url_json（与原版 YouTube 脚本保持一致）
    desc_map = {str(x["id"]): x for x in data}
    for jpath in (CFG["video_intro_json"], CFG["video_url_json"]):
        if os.path.exists(jpath):
            lst = load_json(jpath)
            for entry in lst:
                eid = str(entry.get("id",""))
                if eid in desc_map and desc_map[eid].get("label"):
                    entry["label"] = desc_map[eid]["label"]
            save_json(lst, jpath)
            tqdm.write(f"  label 已同步回写 → {jpath}")

    print(f"  新增:{added} | 跳过:{skipped} → {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════════
#  ★ BASELINE 实验元数据（完整 RAG，与 allinone.py Phase2 相同）
# ════════════════════════════════════════════════════════════════

BASELINE_META = {
    "id":       "BASELINE",
    "name":     "完整 RAG Pipeline",
    "desc":     "完整 Step1→2→4→5→3 + Phase2 标准 RAG 流程（语义检索→c_label→示例→生成）",
    "pipeline": "Step1→Step2(含简介)→Step4→Step5→Step3→Phase2(完整RAG，语义检索+c_label+示例)",
    "variable": "基线（完整系统，无任何消融）",
}


# ════════════════════════════════════════════════════════════════
#  PHASE 2 — 公共基础设施
# ════════════════════════════════════════════════════════════════

# ── Embedding ──────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    resp = ollama.embed(model=EMBED_MODEL, input=text)
    vec  = resp.embeddings[0] if hasattr(resp,"embeddings") else resp["embeddings"][0]
    return np.array(vec, dtype=np.float32)

def cosine_sim(v1, v2) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 0.0 if n1==0 or n2==0 else float(np.dot(v1,v2)/(n1*n2))

# ── 学习样本预处理 & 缓存 ───────────────────────────────────────

def get_preprocessed_samples() -> list:
    cache    = CFG["sample_cache"]
    raw_path = CFG.get("sample_raw_json","")
    src      = raw_path if os.path.exists(raw_path) else CFG.get("sample_json","")

    if os.path.exists(cache):
        try:
            with open(cache,"rb") as f: data = pickle.load(f)
            if data:
                print(f"  ✅ 加载缓存样本 {len(data)} 条 ({CFG['platform']})"); return data
        except Exception as e:
            print(f"  ⚠️ 缓存损坏({e})，重新计算"); os.remove(cache)

    if not os.path.exists(src):
        print(f"  [警告] 样本文件不存在：{src}"); return []

    raw = load_json(src)
    print(f"  🔄 计算 {len(raw)} 条样本 embedding ({CFG['platform']}) ...")
    for s in tqdm(raw, desc="Embedding"):
        s["_embedding"] = get_embedding(build_sample_text(s))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache,"wb") as f: pickle.dump(raw, f)
    return raw

def build_label_index(samples: list) -> dict:
    idx = collections.defaultdict(list)
    for s in samples:
        lb = s.get("label") or s.get("C1_label") or ""
        if lb: idx[lb].append(s)
    return idx

def _get_any_comments(sample: dict, top_n: int = 2) -> list:
    """不区分 c_label，随机取评论（EXP-3/4 专用）。"""
    pool = [sample.get(f"comment_{i}","") for i in range(1,6) if sample.get(f"comment_{i}","")]
    if pool: return random.sample(pool, min(top_n, len(pool)))
    if isinstance(sample.get("comments"), list):
        cs = [c.get("content",c.get("comment","")) for c in sample["comments"]
              if c.get("content") or c.get("comment")]
        if cs: return random.sample(cs, min(top_n, len(cs)))
    return []

def _get_c_labels(sample: dict) -> list:
    """提取 sample 中所有 c_label（兼容扁平/列表两种格式）。"""
    labels = [sample.get(f"C{i}_label","") for i in range(1,6) if sample.get(f"C{i}_label","")]
    if labels: return labels
    if isinstance(sample.get("comments"), list):
        return [c.get("c_label","") for c in sample["comments"] if c.get("c_label","")]
    cl = sample.get("c_label","")
    return [cl] if cl else []

def _get_comments_by_c_label(sample: dict, c_label: str, top_n: int) -> list:
    """从 sample 提取指定 c_label 对应的评论，最多 top_n 条。"""
    results = []
    for i in range(1, 6):
        if sample.get(f"C{i}_label","") == c_label:
            t = sample.get(f"comment_{i}","")
            if t: results.append(t)
        if len(results) >= top_n: break
    if results: return results[:top_n]
    if isinstance(sample.get("comments"), list):
        matched = sorted(
            [c for c in sample["comments"] if c.get("c_label","") == c_label],
            key=lambda c: c.get("rank", 9999)
        )
        return [c.get("content", c.get("comment","")) for c in matched[:top_n]]
    return []

# ── 梗搜索（完整保留 YouTube 脚本的双源搜索逻辑）──────────────

def _search_meme_zh(keyword) -> tuple:
    try:
        r = requests.get(f"https://regengbaike.com/search?q={keyword}", timeout=5)
        soup = BeautifulSoup(r.text,"html.parser")
        res  = soup.find("div",class_="search-result")
        if res: return res.find("h2").text.strip(), res.find("p").text.strip()
    except Exception: pass
    return "", ""

def _search_meme_en(keyword) -> tuple:
    headers = {"User-Agent":"Mozilla/5.0"}
    # Urban Dictionary
    try:
        r    = requests.get(f"https://www.urbandictionary.com/define.php?term={keyword}",
                            headers=headers, timeout=5)
        soup = BeautifulSoup(r.text,"html.parser")
        word = soup.find("a",class_="word")
        meaning = soup.find("div",class_="meaning")
        if word and meaning: return word.text.strip(), meaning.text.strip()
    except Exception: pass
    # Know Your Meme
    try:
        r    = requests.get(f"https://knowyourmeme.com/search?q={keyword}",
                            headers=headers, timeout=5)
        soup = BeautifulSoup(r.text,"html.parser")
        entry = soup.find("td",class_="entry-info")
        if entry:
            link   = "https://knowyourmeme.com" + entry.find("a")["href"]
            detail = requests.get(link, headers=headers, timeout=5)
            dsoup  = BeautifulSoup(detail.text,"html.parser")
            about  = dsoup.find("section",id="about")
            if about:
                return entry.find("a").text.strip(), about.text.strip().split("\n")[0]
    except Exception: pass
    return "", ""

def _load_cached_meme(keyword) -> dict | None:
    folder = CFG.get("hot_meme_folder","")
    if not folder or not os.path.isdir(folder): return None
    for fn in os.listdir(folder):
        if not fn.endswith(".json"): continue
        try:
            with open(os.path.join(folder,fn),encoding="utf-8") as fh:
                meme = json.load(fh)
            if keyword in meme.get("梗名","") or keyword in meme.get("定义",""):
                return meme
        except Exception: continue
    return None

def _save_meme(meme, context="", example=""):
    folder = CFG.get("hot_meme_folder","")
    if not folder: return
    os.makedirs(folder, exist_ok=True)
    path     = os.path.join(folder, f"{meme['梗名']}.json")
    existing = meme
    if os.path.exists(path):
        try:
            with open(path,encoding="utf-8") as fh: existing = json.load(fh)
        except Exception: pass
    if context and context not in existing.get("适用场景",[]):
        existing.setdefault("适用场景",[]).append(context)
    if example and example not in existing.get("表达方式",[]):
        existing.setdefault("表达方式",[]).append(example)
    with open(path,"w",encoding="utf-8") as fh:
        json.dump(existing, fh, ensure_ascii=False, indent=2)

def find_related_meme(video: dict) -> dict | None:
    text = " ".join(filter(None,[
        video.get("video_introduction",""),
        video.get("video_description",""),
        video.get("all_transcription",""),
    ]))
    lang = detect_language(text)
    for word in extract_keywords(text):
        meme = _load_cached_meme(word)
        if meme: return meme
        title, desc = (_search_meme_zh if lang=="zh" else _search_meme_en)(word)
        if title and desc:
            meme_data = {
                "梗名": title, "定义": desc,
                "适用场景": [video.get("video_description","")],
                "表达方式": [],
            }
            _save_meme(meme_data, video.get("video_description",""))
            return meme_data
    return None

# ── Prompt 构建 ────────────────────────────────────────────────

def _pname():
    return {"douyin":"抖音","youtube":"YouTube"}.get(CFG.get("platform",""),"")

def prompt_full_rag(video: dict, c_label: str, examples: list,
                    meme_data: dict = None) -> str:
    """
    BASELINE 专用：完整 RAG prompt（含 c_label + 示例 + 梗，与 allinone.py 相同）。
    """
    intro = video.get("video_introduction","")
    desc  = video.get("video_description","")
    tr    = video.get("all_transcription","")
    lang  = detect_lang([desc, intro, tr])
    p     = _pname()
    ex    = "\n".join(f"  {i+1}. {e}" for i,e in enumerate(examples) if e)

    if lang == "zh":
        meme_line = (
            f"\n使用梗：【{meme_data['梗名']}】，定义：{meme_data['定义']}"
            if (c_label in ("梗应用","Meme Application") and meme_data) else ""
        )
        return (
            f"你是一个{p}评论生成助手，擅长模仿真实用户的评论风格。\n\n"
            f"【视频简介】{intro}\n【视频描述】{desc}\n【字幕转录】{tr}{meme_line}\n\n"
            f"【目标评论风格】：{c_label}\n\n"
            f"【参考评论示例】（模仿句式和语气，内容需结合当前视频）：\n{ex}\n\n"
            "要求：只输出一句评论，不要解释，不要加引号。\n\n评论："
        ).strip()
    else:
        meme_line = (
            f"\nUse the meme '{meme_data['梗名']}': '{meme_data['定义']}'"
            if (c_label in ("梗应用","Meme Application") and meme_data) else ""
        )
        return (
            f"You are a {p} comment generator. Mimic real user comment style.\n\n"
            f"[Introduction] {intro}\n[Description] {desc}\n[Transcript] {tr}{meme_line}\n\n"
            f"[Target style]: {c_label}\n\n"
            f"[Reference examples] (mimic style, adapt to this video):\n{ex}\n\n"
            "Output ONLY the comment, no explanation.\n\nComment:"
        ).strip()


def prompt_direct(video: dict) -> str:
    """EXP-1/2：不传示例，让模型直接根据视频内容生成评论。"""
    intro = video.get("video_introduction","")
    desc  = video.get("video_description","")
    tr    = video.get("all_transcription","")
    lang  = detect_lang([desc, intro, tr])
    p     = _pname()
    if lang == "zh":
        return (f"你是一个{p}评论生成助手，请根据视频内容生成一条幽默、自然的评论。\n\n"
                f"【视频简介】{intro}\n【视频描述】{desc}\n【字幕转录】{tr}\n\n"
                "要求：只输出一句评论，不要解释，不要加引号。\n\n评论：").strip()
    else:
        return (f"You are a {p} comment generator. Write a humorous, natural comment.\n\n"
                f"[Introduction] {intro}\n[Description] {desc}\n[Transcript] {tr}\n\n"
                "Output ONLY the comment, no explanation.\n\nComment:").strip()

def prompt_with_examples(video: dict, label_hint: str, examples: list) -> str:
    """EXP-3/4：传入随机抽取的示例评论。"""
    intro = video.get("video_introduction","")
    desc  = video.get("video_description","")
    tr    = video.get("all_transcription","")
    lang  = detect_lang([desc, intro, tr])
    p     = _pname()
    ex    = "\n".join(f"  {i+1}. {e}" for i,e in enumerate(examples) if e)
    if lang == "zh":
        return (f"你是一个{p}评论生成助手，请模仿下方参考评论的风格，为当前视频生成一条评论。\n\n"
                f"【视频简介】{intro}\n【视频描述】{desc}\n【字幕转录】{tr}\n\n"
                f"【参考评论示例】（模仿句式和语气，内容需结合当前视频）：\n{ex}\n\n"
                "要求：只输出一句评论，不要解释，不要加引号。\n\n评论：").strip()
    else:
        return (f"You are a {p} comment generator. Mimic the style of the examples below.\n\n"
                f"[Introduction] {intro}\n[Description] {desc}\n[Transcript] {tr}\n\n"
                f"[Reference examples] (mimic style, adapt to current video):\n{ex}\n\n"
                "Output ONLY the comment, no explanation.\n\nComment:").strip()

# ── 生成评论 ──────────────────────────────────────────────────

def generate_comment(prompt: str, model_alias: str) -> str:
    try:
        resp = ollama.chat(
            model=resolve_model(model_alias),
            messages=[{"role":"user","content":prompt}],
            options=GEN_OPTIONS, think=False,
        )
        content = resp.message.content if hasattr(resp,"message") else resp["message"]["content"]
        return content.strip() if content else ""
    except Exception as e:
        print(f"  [生成错误] {model_alias}: {e}"); return ""


# ════════════════════════════════════════════════════════════════
#  PHASE 2 — 单视频消融处理（四种模式）
# ════════════════════════════════════════════════════════════════

def phase2_single(video: dict, samples: list, label_index: dict,
                  exp_type: str, model_aliases: list,
                  top_k: int = 3, examples_per_sample: int = 2) -> dict:
    mode        = EXP_META[exp_type]["phase2_mode"]
    video_label = video.get("label","")
    vid         = str(video.get("id",""))
    result      = {}

    # ── EXP-1 / EXP-2: 直接生成，不传示例 ───────────────────────
    if mode == "direct":
        prompt     = prompt_direct(video)
        examples   = []
        label_used = ""

    # ── EXP-3: 相同 label 中随机抽取示例 ─────────────────────────
    elif mode == "same_label_random":
        pool = label_index.get(video_label,[]) or (list(label_index.values())[0] if label_index else samples)
        pool = [s for s in pool if str(s.get("id","")) != vid] or pool
        picked   = random.sample(pool, min(top_k, len(pool)))
        examples = []
        for s in picked: examples.extend(_get_any_comments(s, top_n=examples_per_sample))
        random.shuffle(examples)
        label_used = f"同label({video_label})随机"
        prompt     = prompt_with_examples(video, label_used, examples)

    # ── EXP-4: 不同 label 中随机抽取示例 ─────────────────────────
    else:
        cross = [s for s in samples if (s.get("label") or s.get("C1_label","")) != video_label]
        if not cross: cross = samples
        picked   = random.sample(cross, min(top_k, len(cross)))
        examples = []
        for s in picked: examples.extend(_get_any_comments(s, top_n=examples_per_sample))
        random.shuffle(examples)
        label_used = f"跨label随机(非{video_label})"
        prompt     = prompt_with_examples(video, label_used, examples)

    result["exp_phase2_mode"]    = mode
    result["exp_label_used"]     = label_used if mode != "direct" else ""
    result["exp_examples_count"] = len(examples)

    # ── 多模型分别生成 ─────────────────────────────────────────────
    for alias in model_aliases:
        comment = generate_comment(prompt, alias)
        result[f"{alias}_generated_comment"] = comment
        tqdm.write(f"    [{alias}] {comment[:60]}...")

    return result


# ════════════════════════════════════════════════════════════════
#  BASELINE — 单视频处理（完整 RAG，与 allinone.py Phase2 相同）
# ════════════════════════════════════════════════════════════════

def baseline_single(video: dict, samples: list, label_index: dict,
                    model_aliases: list, top_k: int = 3,
                    examples_per_sample: int = 2) -> dict:
    """
    BASELINE：完整 RAG 流程。
      1. 语义检索 top-k 相似样本
      2. 投票确定 c_label
      3. 提取对应示例评论
      4. 若 c_label 为梗应用，搜索梗信息
      5. 各模型生成评论
    """
    video_label  = video.get("label","")
    video_text   = build_sample_text(video)
    result       = {}

    # 语义检索
    query_emb  = get_embedding(video_text)
    candidates = label_index.get(video_label, []) if video_label else []
    if not candidates:
        candidates = samples
    scored = sorted(
        [(cosine_sim(query_emb, s["_embedding"]), s) for s in candidates],
        key=lambda x: x[0], reverse=True
    )[:top_k]

    # 投票确定 c_label
    c_counter = collections.Counter()
    for _, s in scored:
        c_counter.update(_get_c_labels(s))
    c_label = c_counter.most_common(1)[0][0] if c_counter else "普通幽默"

    # 提取示例评论
    examples = []
    for _, s in scored:
        examples.extend(_get_comments_by_c_label(s, c_label, examples_per_sample))

    # 梗搜索（仅当 c_label 为梗应用时）
    meme_data = None
    if c_label in ("梗应用","Meme Application"):
        meme_data = find_related_meme(video)
        if meme_data:
            tqdm.write(f"    🔍 找到梗：【{meme_data['梗名']}】")

    # 构建 prompt
    prompt = prompt_full_rag(video, c_label, examples, meme_data)

    result["exp_phase2_mode"]    = "full_rag"
    result["exp_c_label"]        = c_label
    result["exp_label_used"]     = c_label
    result["exp_examples_count"] = len(examples)

    # 各模型分别生成
    for alias in model_aliases:
        comment = generate_comment(prompt, alias)
        result[f"{alias}_generated_comment"] = comment
        tqdm.write(f"    [{alias}] {comment[:60]}...")

    return result


def run_baseline(desc_json: str, samples: list, model_aliases: list,
                 top_k: int = 3, examples_per_sample: int = 2) -> list:
    """
    BASELINE 实验：对 desc_json 中的所有视频运行完整 RAG 流程。
    输出结构与消融实验完全相同，ablation_exp_type = "BASELINE"。
    """
    video_data  = load_json(desc_json)
    label_index = build_label_index(samples)
    platform    = CFG["platform"].upper()
    meta        = BASELINE_META

    print(f"\n{'━'*60}")
    print(f"  ▶ [{platform}] {meta['id']} — {meta['name']}")
    print(f"    {meta['desc']}")
    print(f"    模型对比：{', '.join(model_aliases)}")
    print(f"{'━'*60}")

    results = []
    for video in tqdm(video_data, desc=f"  {meta['id']}", unit="视频"):
        rec = copy.deepcopy(video)

        rec["ablation_platform"]     = CFG["platform"]
        rec["ablation_exp_type"]     = meta["id"]
        rec["ablation_exp_name"]     = meta["name"]
        rec["ablation_exp_desc"]     = meta["desc"]
        rec["ablation_exp_pipeline"] = meta["pipeline"]
        rec["ablation_variable"]     = meta["variable"]

        tqdm.write(
            f"\n  [{platform}] ID={rec.get('id')} "
            f"| label={rec.get('label','未分类')} "
            f"| 简介={'有' if rec.get('video_introduction') else '空'}"
        )

        gen = baseline_single(rec, samples, label_index, model_aliases,
                              top_k, examples_per_sample)
        rec.update(gen)
        results.append(rec)

    return results


# ════════════════════════════════════════════════════════════════
#  单类实验入口
# ════════════════════════════════════════════════════════════════

def run_one_exp(desc_json: str, samples: list, exp_type: str,
                model_aliases: list, top_k: int, examples_per_sample: int) -> list:
    video_data  = load_json(desc_json)
    label_index = build_label_index(samples)
    meta        = EXP_META[exp_type]
    platform    = CFG["platform"].upper()

    print(f"\n{'━'*60}")
    print(f"  ▶ [{platform}] {meta['id']} — {meta['name']}")
    print(f"    {meta['desc']}")
    print(f"    控制变量：{meta['variable']}")
    print(f"    模型对比：{', '.join(model_aliases)}")
    print(f"{'━'*60}")

    results = []
    for video in tqdm(video_data, desc=f"  {meta['id']}", unit="视频"):
        rec = copy.deepcopy(video)

        # 写入实验标注字段
        rec["ablation_platform"]     = CFG["platform"]
        rec["ablation_exp_type"]     = meta["id"]
        rec["ablation_exp_name"]     = meta["name"]
        rec["ablation_exp_desc"]     = meta["desc"]
        rec["ablation_exp_pipeline"] = meta["pipeline"]
        rec["ablation_variable"]     = meta["variable"]

        tqdm.write(
            f"\n  [{platform}] ID={rec.get('id')} "
            f"| label={rec.get('label','未分类')} "
            f"| 简介={'有' if rec.get('video_introduction') else '空'}"
        )

        gen = phase2_single(rec, samples, label_index, exp_type,
                            model_aliases, top_k, examples_per_sample)
        rec.update(gen)
        results.append(rec)

    return results


# ════════════════════════════════════════════════════════════════
#  ★ 消融实验总控制器
# ════════════════════════════════════════════════════════════════

def run_ablation(exp_types: list, model_aliases: list,
                 skip_phase1: bool = False,
                 top_k: int = 3, examples_per_sample: int = 2,
                 run_baseline_exp: bool = True):
    """
    消融实验总入口。

    run_baseline_exp=True 时，在四类消融实验之外额外运行 BASELINE 实验：
      完整 RAG Pipeline（Step1→2→4→5→3 + 语义检索→c_label→示例→生成）

    Phase1 策略：
      EXP-1：Step2(full)→Step4→Step5              （不跑Step3，Phase2不依赖label）
      EXP-2：Step2(no_intro)→Step4→Step5→Step3
      EXP-3/4 / BASELINE：Step2(full)→Step4→Step5→Step3
    """
    platform = CFG["platform"]
    abl_dir  = CFG["ablation_dir"]
    os.makedirs(abl_dir, exist_ok=True)
    t0 = time.time()

    summary = {
        "experiment_time":     time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform":            platform,
        "exp_types_run":       exp_types,
        "models_tested":       model_aliases,
        "step3_model_fixed":   STEP3_MODEL,
        "step5_model_fixed":   STEP5_MODEL,
        "embed_model":         EMBED_MODEL,
        "top_k":               top_k,
        "examples_per_sample": examples_per_sample,
        "experiments":         {},
        "field_schema": {
            "ablation_platform":     f"平台（{platform}）",
            "ablation_exp_type":     "实验类型ID（EXP-1/2/3/4）",
            "ablation_exp_name":     "实验名称",
            "ablation_exp_desc":     "实验说明",
            "ablation_exp_pipeline": "完整 Pipeline",
            "ablation_variable":     "消融的变量",
            "exp_phase2_mode":       "Phase2生成模式（direct/same_label_random/diff_label_random）",
            "exp_label_used":        "示例来源（EXP-1/2 为空字符串）",
            "exp_examples_count":    "传入示例数量（EXP-1/2 为 0）",
            "{model}_generated_comment": "各模型生成的评论，如 qwen3.5_generated_comment",
        },
    }

    # ── 路径规划 ─────────────────────────────────────────────────
    standard_desc = CFG["video_desc_json"]
    exp2_intro    = os.path.join(abl_dir, "exp2_video_introduction_no_intro.json")
    exp2_chouzhen = os.path.join(abl_dir, "exp2_chouzhen_no_intro.json")
    exp2_desc     = os.path.join(abl_dir, "exp2_video_description_no_intro.json")

    # ════════════════════════════════════════════════════════
    #  Phase 1
    # ════════════════════════════════════════════════════════
    if not skip_phase1:
        print(f"\n{'★'*60}\n  PHASE 1 — {platform.upper()} 视频处理 Pipeline\n{'★'*60}")

        # Step 2 标准（EXP-1/3/4 或 BASELINE 使用）
        if set(exp_types) & {"1","3","4"} or run_baseline_exp:
            print("\n[Step 2 — 标准，含简介] → 供 EXP-1/3/4 + BASELINE 使用")
            asyncio.run(step2_download(mode="full", out_json=CFG["video_intro_json"]))

        # Step 2 无简介（EXP-2）
        if "2" in exp_types:
            print("\n[Step 2 — 简介置空] → 供 EXP-2 使用")
            asyncio.run(step2_download(mode="no_intro", out_json=exp2_intro))

        # Step 4（共享，与简介无关，只需运行一次）
        step4_chouzhen(intro_json=CFG["video_intro_json"])

        # Step 5 标准（EXP-1/3/4 或 BASELINE 使用）
        if set(exp_types) & {"1","3","4"} or run_baseline_exp:
            print("\n[Step 5 — 标准，含简介] → 供 EXP-1/3/4 + BASELINE 使用")
            step5_descriptions(chouzhen_json=CFG["chouzhen_json"], out_json=standard_desc)

        # Step 5 无简介（EXP-2）：把 chouzhen 的简介字段清空再送入 step5
        if "2" in exp_types:
            print("\n[Step 5 — 无简介输入] → 供 EXP-2 使用")
            chz = load_json(CFG["chouzhen_json"])
            save_json([dict(r, video_introduction="") for r in chz], exp2_chouzhen)
            step5_descriptions(chouzhen_json=exp2_chouzhen, out_json=exp2_desc)

        # Step 3（EXP-2/3/4 和 BASELINE 需要 label；EXP-1 不需要）
        needs_step3 = set(exp_types) & {"2","3","4"}
        if run_baseline_exp:
            needs_step3.add("baseline")  # 触发条件

        if needs_step3:
            print("\n[Step 3 — 标准] → 供 EXP-2/3/4 + BASELINE 使用")
            step3_label(in_json=standard_desc, out_json=standard_desc)
        if "2" in exp_types:
            print("\n[Step 3 — EXP-2 无简介版]")
            step3_label(in_json=exp2_desc, out_json=exp2_desc)

    else:
        print("\n  [--skip-phase1] 跳过 Phase1，使用已有文件")
        missing = []
        if (set(exp_types) & {"1","3","4"} or run_baseline_exp) and not os.path.exists(standard_desc):
            missing.append(standard_desc)
        if "2" in exp_types and not os.path.exists(exp2_desc):
            missing.append(exp2_desc)
        if missing:
            print("  [错误] 文件缺失，请先运行 Phase1：")
            for m in missing: print(f"    · {m}")
            return

    # ════════════════════════════════════════════════════════
    #  预加载学习样本 embedding
    # ════════════════════════════════════════════════════════
    print(f"\n📚 加载学习样本 embedding ({platform}) ...")
    samples = get_preprocessed_samples()

    # ════════════════════════════════════════════════════════
    #  Phase 2 — 逐类消融实验
    # ════════════════════════════════════════════════════════
    all_results = []

    for exp_type in exp_types:
        t_exp     = time.time()
        desc_json = exp2_desc if exp_type == "2" else standard_desc

        results = run_one_exp(
            desc_json=desc_json, samples=samples, exp_type=exp_type,
            model_aliases=model_aliases, top_k=top_k,
            examples_per_sample=examples_per_sample,
        )

        elapsed    = round(time.time() - t_exp, 1)
        single_out = os.path.join(abl_dir, f"ablation_exp_{exp_type}.json")
        save_json(results, single_out)
        print(f"\n  ✅ {EXP_META[exp_type]['id']} 完成，{len(results)} 条，{elapsed}s → {single_out}")

        summary["experiments"][exp_type] = {
            "exp_id":   EXP_META[exp_type]["id"],
            "exp_name": EXP_META[exp_type]["name"],
            "variable": EXP_META[exp_type]["variable"],
            "n_videos": len(results), "elapsed_s": elapsed, "file": single_out,
        }
        all_results.extend(results)

    # ════════════════════════════════════════════════════════
    #  BASELINE 实验（完整 RAG）
    # ════════════════════════════════════════════════════════
    baseline_results = []
    if run_baseline_exp:
        t_base = time.time()
        baseline_results = run_baseline(
            desc_json          = standard_desc,
            samples            = samples,
            model_aliases      = model_aliases,
            top_k              = top_k,
            examples_per_sample= examples_per_sample,
        )
        elapsed_base = round(time.time() - t_base, 1)
        base_out = os.path.join(abl_dir, "baseline_results.json")
        save_json(baseline_results, base_out)
        print(f"\n  ✅ BASELINE 完成，{len(baseline_results)} 条，{elapsed_base}s → {base_out}")
        summary["experiments"]["BASELINE"] = {
            "exp_id":   "BASELINE",
            "exp_name": BASELINE_META["name"],
            "variable": BASELINE_META["variable"],
            "n_videos": len(baseline_results),
            "elapsed_s": elapsed_base,
            "file":     base_out,
        }
        all_results.extend(baseline_results)

    # ── 合并 & 汇总 ──────────────────────────────────────────
    all_out = os.path.join(abl_dir, "ablation_all_results.json")
    save_json(all_results, all_out)
    summary.update({
        "total_elapsed_s":   round(time.time() - t0, 1),
        "all_results_file":  all_out,
        "total_records":     len(all_results),
    })
    save_json(summary, os.path.join(abl_dir, "ablation_summary.json"))

    # ── 汇总打印 ──────────────────────────────────────────────
    print(f"\n\n{'★'*60}")
    print(f"  ★ [{platform.upper()}] 实验全部完成！总耗时 {summary['total_elapsed_s']}s")
    print(f"{'★'*60}\n  输出目录：{abl_dir}\n")
    W = 42
    print(f"  {'文件':<{W}} 说明")
    print(f"  {'─'*W} {'─'*28}")
    for et in exp_types:
        info = summary["experiments"][et]
        print(f"  {f'ablation_exp_{et}.json':<{W}} {info['exp_name']}（{info['n_videos']}条，{info['elapsed_s']}s）")
    if run_baseline_exp and "BASELINE" in summary["experiments"]:
        info = summary["experiments"]["BASELINE"]
        print(f"  {'baseline_results.json':<{W}} {info['exp_name']}（{info['n_videos']}条，{info['elapsed_s']}s）")
    print(f"  {'ablation_all_results.json':<{W}} 全部合并（{len(all_results)}条，含BASELINE）")
    print(f"  {'ablation_summary.json':<{W}} 实验配置 & 字段说明")
    print(f"\n  评论字段：", " | ".join(f"{a}_generated_comment" for a in model_aliases))
    print(f"  区分实验：ablation_exp_type 字段（EXP-1/2/3/4 / BASELINE）")
    return summary


# ════════════════════════════════════════════════════════════════
#  ★ 主入口（首先交互选择平台）
# ════════════════════════════════════════════════════════════════

def select_platform_interactive() -> str:
    print("\n" + "╔" + "═"*56 + "╗")
    print("║" + "  抖音 / YouTube 视频评论生成 — 消融实验脚本".center(58) + "║")
    print("╚" + "═"*56 + "╝")
    print()
    print("  请选择目标平台：")
    print()
    print("    1. 🎵  抖音  (Douyin)")
    print("         下载方式：Playwright 拦截 aweme_detail")
    print("         简介来源：视频描述文字（aweme.desc）")
    print()
    print("    2. 📺  YouTube")
    print("         下载方式：yt-dlp")
    print("         简介来源：视频标题（YouTube Data API v3）")
    print()
    while True:
        choice = input("  输入 1 或 2（或 douyin / youtube）：").strip().lower()
        if choice in ("1","douyin"):  return "douyin"
        if choice in ("2","youtube"): return "youtube"
        print("  ⚠️  无效输入，请输入 1 或 2。")


def main():
    parser = argparse.ArgumentParser(
        description="抖音 / YouTube 视频评论生成 — 消融实验脚本",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
平台选择：
  不指定 --platform 时进入交互式选择。
  --platform douyin   使用抖音配置（Playwright + aweme_detail）
  --platform youtube  使用 YouTube 配置（yt-dlp + Data API）

四种消融实验：
  1  有简介 + 直接生成  → Phase2 不传任何示例（验证 RAG 整体贡献）
  2  无简介 + 直接生成  → Step2 简介置空，Phase2 不传示例（验证简介贡献）
  3  同label随机模仿    → 相同label随机抽评论作示例（验证语义检索贡献）
  4  跨label随机模仿    → 不同label随机抽评论作示例（验证label分类贡献）

BASELINE 多模型对比实验（默认同时运行，用 --no-baseline 跳过）：
  完整 RAG Pipeline：语义检索→c_label确定→示例提取→梗搜索→生成
  四个模型各自生成，输出到 baseline_results.json

输出（以 douyin 为例）：
  ablation_results/douyin/ablation_exp_{1-4}.json   ← 各类单独
  ablation_results/douyin/ablation_all_results.json ← 全部合并
  ablation_results/douyin/ablation_summary.json     ← 配置&字段说明

评论字段：
  qwen3.5_generated_comment / glm_generated_comment
  deepseek-r1_generated_comment / llama_generated_comment
"""
    )
    parser.add_argument("--platform", choices=["douyin","youtube"], default=None,
                        help="目标平台（不指定则交互选择）")
    parser.add_argument("--exp", nargs="+", choices=["1","2","3","4"],
                        default=["1","2","3","4"], metavar="N",
                        help="消融实验编号（默认全部 1 2 3 4）")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                        metavar="MODEL",
                        help=f"Phase2 评论生成模型（默认全部）\n可选：{ALL_MODELS}")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="跳过 Phase1，直接用已有 JSON 运行 Phase2")
    parser.add_argument("--no-baseline", action="store_true",
                        help="跳过 BASELINE 实验，只运行四类消融实验")
    parser.add_argument("--top-k", type=int, default=3,
                        help="EXP-3/4 随机抽取样本数（默认 3）")
    parser.add_argument("--examples-per-sample", type=int, default=2,
                        help="每条样本最多取几条示例评论（默认 2）")

    args = parser.parse_args()

    # ── 选择平台 ──────────────────────────────────────────────
    platform = args.platform or select_platform_interactive()

    # ── 初始化全局平台配置 ─────────────────────────────────────
    global CFG
    CFG = build_platform_cfg(platform)

    print(f"\n{'═'*60}")
    print(f"  平台       ：{platform.upper()}")
    print(f"  实验编号   ：{args.exp}")
    print(f"  对比模型   ：{args.models}")
    print(f"  skip_phase1：{args.skip_phase1}")
    print(f"  top_k      ：{args.top_k}")
    print(f"  examples_ps：{args.examples_per_sample}")
    print(f"  输出目录   ：{CFG['ablation_dir']}")
    print(f"{'═'*60}\n")

    run_ablation(
        exp_types          = args.exp,
        model_aliases      = args.models,
        skip_phase1        = args.skip_phase1,
        top_k              = args.top_k,
        examples_per_sample= args.examples_per_sample,
        run_baseline_exp   = not args.no_baseline,
    )


if __name__ == "__main__":
    main()