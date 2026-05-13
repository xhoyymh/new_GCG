from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from model_api_adapter import generate_comment_with_backend, register_local_generator
from original_comment_loader import (
    extract_original_comments,
    get_original_comment_file,
    load_original_comment_index,
)


BASE_MODULE_PATH = Path(__file__).with_name("comment_generate.py")
BASE_SPEC = importlib.util.spec_from_file_location("douyin_comment_generate_base", BASE_MODULE_PATH)
_base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(_base)


PLATFORM = "douyin"
VIDEO_DESCRIPTION_JSON = _base.COMMENT_CONFIG["input_video_file"]
COMMENT_CONFIG = copy.deepcopy(_base.COMMENT_CONFIG)
COMMENT_CONFIG["output_comment_file"] = str(
    Path(_base.BASE_DIR) / "comment_generation" / "json" / "result" / "douyin_output_comments_api_ready.json"
)
COMMENT_CONFIG["original_comment_file"] = str(get_original_comment_file(PLATFORM))

ORIGINAL_COMMENT_INDEX = load_original_comment_index(PLATFORM)

register_local_generator(PLATFORM, _base.generate_comment)


def build_prompt(video: dict, c_label: str, examples: list[str], original_comments: list[str]) -> str:
    if not original_comments:
        return _base.build_prompt(video, c_label, examples)

    desc = video.get("video_description", "")
    intro = video.get("video_introduction", "")
    transcript = video.get("all_transcription", "")
    lang = _base.detect_language(desc)

    example_text = "\n".join(f"  {i + 1}. {e}" for i, e in enumerate(examples) if e)
    original_comment_text = "\n".join(
        f"  {i + 1}. {text}" for i, text in enumerate(original_comments) if text
    )

    if lang == "zh":
        prompt = f"""你是一个短视频评论生成助手，擅长模仿真实用户的评论风格。

【视频信息】
- 视频介绍：{intro}
- 视频描述：{desc}
- 字幕信息：{transcript}

【目标评论风格】：{c_label}

【检索示例评论】（模仿句式和语气，但不要照抄）
{example_text}

【平台原评论参考】（仅参考语气和表达，不要直接复写）
{original_comment_text}

要求：
- 评论要自然、真实，像真人在评论区留言
- 内容必须结合当前视频，不要复述原评论
- 可以借鉴平台原评论的语气和表达方式，但不能直接复制
- 只输出一句评论，不要解释，不要加引号

评论："""
    else:
        prompt = f"""You are a short-video comment generator. Mimic the style of real user comments.

[Video Info]
- Introduction: {intro}
- Description: {desc}
- Transcript: {transcript}

[Target comment style]: {c_label}

[Retrieved examples] (imitate the tone and sentence structure, but do not copy them)
{example_text}

[Platform original comments] (use them only as tone reference, do not copy them)
{original_comment_text}

Requirements:
- Sound natural and authentic, like a real user
- Content must stay grounded in this video's content
- You may borrow tone from the platform comments, but do not repeat them directly
- Output ONLY the comment, with no explanation and no quotes

Comment:"""

    return prompt.strip()


def prepare_comment_jobs(video_data: list[dict], samples: list, label_index: dict, *,
                         include_original_comments: bool = True,
                         backend: str = "ollama",
                         model_alias: str | None = None) -> list[dict]:
    known_labels = set(label_index.keys())
    jobs = []

    for video in _base.tqdm([dict(video) for video in video_data], desc="Preparing comment jobs", unit="video"):
        video_id = str(video.get("id") or video.get("video_id") or "").strip()
        original_comments = (
            extract_original_comments(ORIGINAL_COMMENT_INDEX.get(video_id))
            if include_original_comments else []
        )

        video_label = video.get("label") or video.get("C1_label") or ""
        video_text = _base.build_sample_text(video)
        query_emb = _base.get_embedding(video_text)

        if video_label and video_label in known_labels:
            candidates = label_index[video_label]
            strategy = f"same-label ({video_label})"
        else:
            candidates = samples
            strategy = "global-search (new label)"

        top_samples = _base.top_k_similar(query_emb, candidates, COMMENT_CONFIG["top_k"])
        c_label, examples = _base.pick_c_label_and_examples(top_samples)
        prompt = build_prompt(video, c_label, examples, original_comments)

        jobs.append(
            {
                "video": video,
                "prompt": prompt,
                "generated_c_label": c_label,
                "strategy": strategy,
                "backend": backend,
                "model_alias": model_alias,
            }
        )

    return jobs


def generate_comments_from_jobs(jobs: list[dict]) -> list[dict]:
    results = []

    for job in _base.tqdm(jobs, desc="Generating comments", unit="video"):
        comment = generate_comment_with_backend(
            job["prompt"],
            backend=job["backend"],
            platform=PLATFORM,
            model_alias=job.get("model_alias"),
            video_record=job["video"],
        )
        video = job["video"]
        video["generated_comment"] = comment
        video["generated_c_label"] = job["generated_c_label"]
        results.append(video)
        _base.tqdm.write(f"  [{job['strategy']}] c_label={job['generated_c_label']} | {comment[:60]}...")

    return results


def run_phase2_with_records(video_data: list[dict], *,
                            learning_sample_file: str | None = None,
                            cache_file: str | None = None,
                            output_file: str | None = None,
                            include_original_comments: bool = True,
                            backend: str = "ollama",
                            model_alias: str | None = None) -> list[dict]:
    samples = _base.get_preprocessed_samples(
        learning_sample_file=learning_sample_file,
        cache_file=cache_file,
    )
    label_index = _base.build_label_index(samples)
    known_labels = set(label_index.keys())
    print(f"📚 Loaded {len(samples)} samples, {len(known_labels)} distinct labels.")
    print(f"🎬 Processing {len(video_data)} videos ...")

    jobs = []
    try:
        print("🧭 Stage 1/2: preparing comment jobs ...")
        jobs = prepare_comment_jobs(
            video_data,
            samples,
            label_index,
            include_original_comments=include_original_comments,
            backend=backend,
            model_alias=model_alias,
        )
    finally:
        _base.stop_ollama_model(
            COMMENT_CONFIG.get("ollama_embed_model", ""),
            "embedding stage",
            wait_timeout=20.0,
        )

    try:
        print("💬 Stage 2/2: generating comments ...")
        results = generate_comments_from_jobs(jobs)
    finally:
        if backend == "ollama":
            _base.stop_ollama_model(
                COMMENT_CONFIG.get("ollama_model", ""),
                "comment generation stage",
                wait_timeout=20.0,
            )

    if output_file:
        _base.save_json(results, output_file)
        print(f"\n✅ Done! Comments saved to {output_file}")

    return results


def phase2_generate_comments() -> None:
    print("\n\n" + "★" * 60)
    print("  PHASE 2 (API READY)：评论生成")
    print("★" * 60)

    COMMENT_CONFIG["input_video_file"] = str(VIDEO_DESCRIPTION_JSON)
    video_data = _base.load_json(COMMENT_CONFIG["input_video_file"])
    run_phase2_with_records(
        video_data,
        learning_sample_file=COMMENT_CONFIG["learning_sample_file"],
        cache_file=COMMENT_CONFIG["cache_file"],
        output_file=COMMENT_CONFIG["output_comment_file"],
        backend="ollama",
    )
