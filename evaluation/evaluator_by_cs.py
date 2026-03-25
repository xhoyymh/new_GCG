"""
消融实验 + BASELINE 多模型对比 评论质量评估脚本
=================================================

在 evaluate_for_douyin.py / evaluate_for_youtube.py 基础上修改而来。
评分对象：
  ① 消融实验（EXP-1/2/3/4）ablation_all_results.json  ← 四种控制变量实验
  ② BASELINE 多模型对比     baseline_results.json      ← 完整 RAG，四模型正常输出
  ③ 额外评论通道            --extra 指定的 JSON 文件   ← 其他系统/人工评论

三组结果在同一套评分维度下打分，报告中分栏对比，方便直接写入论文。

评分维度（各满分 10，总分 = 三维度均值，满分 10）：
  原创性      与学习样本语义相似度（越低越好），含总结关键词/批次重复各 -3
  具体性      当前评论相关度 vs 参考人类基线差距
  风格符合性  情感一致性（10/5）- 长度惩罚（0~5），重复时情感分压至 1

模型：
  SentenceTransformer  all-MiniLM-L6-v2
  情感分析             distilbert-base-uncased-finetuned-sst-2-english

输出文件（evaluation/result/{platform}/）：
  ablation_eval_detail.json    每条 × 每模型 详细得分
  ablation_eval_summary.csv    消融实验：按实验类型 × 生成模型汇总
  baseline_eval_summary.csv    BASELINE：按生成模型汇总
  ablation_eval_extra.csv      额外评论汇总（有 --extra 时生成）
  ablation_eval_combined.csv   三组统一对比表（实验组 × 模型 → 各维度均值）

用法：
  python ablation_evaluator.py                          # 交互选平台，全量评估
  python ablation_evaluator.py --platform douyin        # 指定平台
  python ablation_evaluator.py --platform douyin --extra my_sys.json other.json
  python ablation_evaluator.py --platform douyin --no-ablation --no-baseline --extra my.json
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# ════════════════════════════════════════════════════════════════
#  ★ 全局配置
# ════════════════════════════════════════════════════════════════

BASE_DIR             = r"D:\Desktop\video_comment_generation\ALLinone"
SBERT_MODEL_NAME     = "all-MiniLM-L6-v2"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# 消融实验 & BASELINE 中四个模型生成的评论字段
GENERATED_COMMENT_FIELDS = [
    "qwen3.5_generated_comment",
    "glm_generated_comment",
    "deepseek-r1_generated_comment",
    "llama_generated_comment",
]

# ── 平台专属参数 & 默认路径 ──────────────────────────────────────
PLATFORM_CFG = {
    "douyin": {
        "ideal_length":  30,
        "summary_kws":   ["视频", "画面", "场景", "描述", "故事"],
        "sample_files": [
            os.path.join(BASE_DIR, "data_pre", "json", "douyin", "sample", "douyin_sample.json"),
            os.path.join(BASE_DIR, "evaluation", "original_comments_for_douyin.json"),
        ],#平台的原来的评论数据，包含了评论和视频描述，用于计算具体性基线和原创性评分
        "reference_file": os.path.join(BASE_DIR, "evaluation", "original_comments_for_douyin.json"),
        # 消融实验结果（EXP-1/2/3/4 合并）
        "ablation_all":   os.path.join(BASE_DIR, "ablation&modelcompare", "json", "ablation_results", "douyin", "ablation_all_results.json"),
        # BASELINE 多模型对比结果
        "baseline_file":  os.path.join(BASE_DIR, "ablation_results", "douyin", "baseline_results.json"),
        "eval_dir":       os.path.join(BASE_DIR, "evaluation", "result", "douyin"),
    },
    "youtube": {
        "ideal_length":  72,
        "summary_kws":   ["video", "scene", "description", "story", "footage"],
        "sample_files": [
            os.path.join(BASE_DIR, "data_pre", "json", "youtube", "sample", "youtube_sample.json"),
            os.path.join(BASE_DIR, "evaluation", "original_comments_for_youtube.json"),
        ],#平台的原来的评论数据，包含了评论和视频描述，用于计算具体性基线和原创性评分
        "reference_file": os.path.join(BASE_DIR, "evaluation", "original_comments_for_youtube.json"),
        "ablation_all":   os.path.join(BASE_DIR, "ablation_results", "youtube", "ablation_all_results.json"),
        "baseline_file":  os.path.join(BASE_DIR, "ablation_results", "youtube", "baseline_results.json"),
        "eval_dir":       os.path.join(BASE_DIR, "evaluation", "result", "youtube"),
    },
}


# ════════════════════════════════════════════════════════════════
#  模型懒加载
# ════════════════════════════════════════════════════════════════

_sbert_model     = None
_sentiment_model = None


def get_sbert() -> SentenceTransformer:
    global _sbert_model
    if _sbert_model is None:
        print(f"  🔄 加载语义模型 {SBERT_MODEL_NAME} ...")
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        print(f"  ✅ 语义模型已加载")
    return _sbert_model


def get_sentiment_pipeline():
    global _sentiment_model
    if _sentiment_model is None:
        print(f"  🔄 加载情感模型 {SENTIMENT_MODEL_NAME} ...")
        _sentiment_model = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
        print(f"  ✅ 情感模型已加载")
    return _sentiment_model


def get_sentiment_label(text: str) -> str:
    try:
        return get_sentiment_pipeline()(text[:512])[0]["label"]
    except Exception:
        return "POSITIVE"


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════
#  数据预加载
# ════════════════════════════════════════════════════════════════

def load_sample_embeddings(sample_file_paths: list) -> np.ndarray:
    """加载样本评论并编码，供原创性打分使用。"""
    model = get_sbert()
    texts = []
    for fp in sample_file_paths:
        if not os.path.exists(fp):
            print(f"  [警告] 样本文件不存在，跳过：{fp}")
            continue
        data = load_json(fp)
        for item in data:
            for i in range(1, 6):
                c = item.get(f"comment_{i}", "").strip()
                if c:
                    texts.append(c)
            if isinstance(item.get("comments"), list):
                for c_obj in item["comments"]:
                    c = c_obj.get("content", c_obj.get("comment","")).strip()
                    if c:
                        texts.append(c)
            c = item.get("comment","").strip()
            if c and c not in texts:
                texts.append(c)

    if not texts:
        print("  [警告] 未找到任何样本评论，原创性得分将为 0")
        return np.array([])

    print(f"  🔄 编码 {len(texts)} 条样本评论 ...")
    vecs = model.encode(texts, show_progress_bar=False, batch_size=64)
    print(f"  ✅ 样本 embedding 完成（{len(texts)} 条）")
    return vecs


def load_reference_specificities(reference_file: str) -> dict:
    """
    读取参考数据集，计算每个 video_id 的
    「人类真实评论 vs 视频描述」平均语义相似度，作为具体性基线。
    """
    model = get_sbert()
    if not os.path.exists(reference_file):
        print(f"  [警告] 参考文件不存在：{reference_file}，具体性得分将为 0")
        return {}

    data    = load_json(reference_file)
    ref_map = {}
    print(f"  🔄 计算参考具体性基线（{len(data)} 条）...")

    for item in tqdm(data, desc="  参考基线", leave=False):
        vid  = str(item.get("id", item.get("video_id",""))).strip()
        desc = item.get("video_description","").strip()
        if not vid or not desc:
            continue
        desc_vec = model.encode([desc])
        sims = []
        for i in range(1, 6):
            c = item.get(f"comment_{i}","").strip()
            if c:
                sims.append(cosine_similarity(model.encode([c]), desc_vec).item())
        if isinstance(item.get("comments"), list):
            for c_obj in item["comments"]:
                c = c_obj.get("content", c_obj.get("comment","")).strip()
                if c:
                    sims.append(cosine_similarity(model.encode([c]), desc_vec).item())
        if sims:
            ref_map[vid] = float(np.mean(sims))

    print(f"  ✅ 参考基线完成（{len(ref_map)} 个视频）")
    return ref_map


# ════════════════════════════════════════════════════════════════
#  三个评分维度（与原版 evaluate_for_douyin/youtube.py 对齐）
# ════════════════════════════════════════════════════════════════

def score_originality(comment: str, comment_vec: np.ndarray,
                      sample_vecs: np.ndarray, all_comments: list,
                      summary_kws: list) -> float:
    """原创性（0~10）。"""
    sim = float(cosine_similarity(comment_vec, sample_vecs).max()) \
          if sample_vecs.size > 0 else 0.0

    is_summary_like = any(kw in comment for kw in summary_kws)
    is_repetitive   = all_comments.count(comment) > 1

    score = 10.0 - sim * 10.0
    if is_summary_like: score -= 3.0
    if is_repetitive:   score -= 3.0
    return round(max(0.0, score), 2)


def score_specificity(comment_vec: np.ndarray, video_description: str,
                      video_id: str, reference_specificities: dict) -> float:
    """具体性（0~10）。"""
    model = get_sbert()
    if not video_description.strip() or video_id not in reference_specificities:
        return 0.0
    desc_vec    = model.encode([video_description])
    current_sim = cosine_similarity(comment_vec, desc_vec).item()
    ref_sim     = reference_specificities[video_id]
    return round(max(0.0, 10.0 - abs(current_sim - ref_sim) * 10.0), 2)


def score_style(comment: str, comment_vec: np.ndarray,
                video_description: str, all_comments: list,
                ideal_length: int) -> float:
    """风格符合性（0~10）。"""
    model = get_sbert()

    length_penalty   = abs(len(comment) - ideal_length) / max(ideal_length, 1)
    length_deduction = min(length_penalty * 5.0, 5.0)

    try:
        c_sent = get_sentiment_label(comment)
        d_sent = get_sentiment_label(video_description)
        sentiment_score = 10.0 if c_sent == d_sent else 5.0
    except Exception:
        sentiment_score = 5.0

    is_repetitive = all_comments.count(comment) > 1
    if len(all_comments) > 1:
        others = [c for c in all_comments if c != comment] or all_comments
        max_sim = float(cosine_similarity(comment_vec,
                        model.encode(others, show_progress_bar=False))[0].max())
        is_sem_rep = max_sim > 0.75
    else:
        is_sem_rep = False

    if is_repetitive or is_sem_rep:
        sentiment_score = min(sentiment_score, 1.0)

    return round(max(0.0, min(10.0, sentiment_score - length_deduction)), 2)


def score_one_comment(comment: str, video_id: str, video_description: str,
                      all_comments: list, sample_vecs: np.ndarray,
                      reference_specificities: dict, cfg: dict) -> dict:
    """对单条评论打全部三维度的分。总分 = 三维度均值（满分 10）。"""
    model = get_sbert()

    if not comment or not comment.strip():
        return {"原创性": 0.0, "具体性": 0.0, "风格符合性": 0.0, "总分": 0.0}

    comment_vec = model.encode([comment])
    orig  = score_originality(comment, comment_vec, sample_vecs,
                              all_comments, cfg["summary_kws"])
    spec  = score_specificity(comment_vec, video_description,
                              video_id, reference_specificities)
    style = score_style(comment, comment_vec, video_description,
                        all_comments, cfg["ideal_length"])

    return {"原创性": orig, "具体性": spec, "风格符合性": style,
            "总分": round((orig + spec + style) / 3.0, 2)}


# ════════════════════════════════════════════════════════════════
#  通用：对一批记录中所有 GENERATED_COMMENT_FIELDS 打分
# ════════════════════════════════════════════════════════════════

def evaluate_records(
    records:                 list,
    sample_vecs:             np.ndarray,
    reference_specificities: dict,
    cfg:                     dict,
    source_type:             str,   # "ablation" | "baseline" | "extra"
    desc: str = "",
) -> list:
    """
    遍历记录列表，对每条记录的 GENERATED_COMMENT_FIELDS 各打一次分。
    source_type 用于在报告中区分三组数据。

    记录结构（ablation/baseline 相同）：
      id / video_description / video_introduction / label /
      ablation_exp_type / ablation_exp_name / ablation_variable / ablation_exp_pipeline /
      qwen3.5_generated_comment / glm_generated_comment / ...
    """
    # 收集本批次所有生成评论（用于重复检测）
    all_batch_comments = [
        rec.get(f, "").strip()
        for rec in records
        for f in GENERATED_COMMENT_FIELDS
        if rec.get(f, "").strip()
    ]

    detail_rows = []
    label = desc or source_type

    for rec in tqdm(records, desc=f"  {label}", unit="视频"):
        vid_id      = str(rec.get("id","")).strip()
        exp_type    = rec.get("ablation_exp_type", "")
        exp_name    = rec.get("ablation_exp_name", "")
        variable    = rec.get("ablation_variable", "")
        pipeline_   = rec.get("ablation_exp_pipeline", "")
        video_desc  = rec.get("video_description", "")
        video_intro = rec.get("video_introduction", "")
        label_val   = rec.get("label", "")
        plat        = rec.get("ablation_platform", cfg.get("platform",""))

        for field in GENERATED_COMMENT_FIELDS:
            gen_model = field.replace("_generated_comment", "")
            comment   = rec.get(field, "").strip()

            scores = score_one_comment(
                comment                 = comment,
                video_id                = vid_id,
                video_description       = video_desc,
                all_comments            = all_batch_comments,
                sample_vecs             = sample_vecs,
                reference_specificities = reference_specificities,
                cfg                     = cfg,
            )

            detail_rows.append({
                "source_type":       source_type,
                "source_label":      f"{exp_type} / {gen_model}" if exp_type else gen_model,
                "id":                vid_id,
                "platform":          plat,
                "label":             label_val,
                "video_description": video_desc,
                "video_introduction":video_intro,
                "exp_type":          exp_type,
                "exp_name":          exp_name,
                "ablation_variable": variable,
                "ablation_pipeline": pipeline_,
                "gen_model":         gen_model,
                "comment":           comment,
                "comment_length":    len(comment),
                **scores,
            })

    return detail_rows


# ════════════════════════════════════════════════════════════════
#  额外评论通道
# ════════════════════════════════════════════════════════════════

def evaluate_extra_comments(
    extra_file_paths:        list,
    sample_vecs:             np.ndarray,
    reference_specificities: dict,
    cfg:                     dict,
) -> list:
    """
    额外评论格式（每文件是一个 list）：
      [{"video_id":"1","url":"1.mp4","label":"","comment":"...","video_description":"..."}, ...]
    文件名（去掉后缀）作为 source_label。
    """
    all_rows = []
    for fp in extra_file_paths:
        if not os.path.exists(fp):
            print(f"  [警告] 额外评论文件不存在：{fp}")
            continue
        data        = load_json(fp)
        source_name = os.path.basename(fp).replace(".json","")
        all_comments = [item.get("comment","").strip() for item in data
                        if item.get("comment","").strip()]

        print(f"\n  额外评论：{source_name}（{len(data)} 条）")

        for item in tqdm(data, desc=f"  {source_name}"):
            vid_id     = str(item.get("video_id", item.get("id",""))).strip()
            comment    = item.get("comment","").strip()
            video_desc = item.get("video_description","").strip()

            scores = score_one_comment(
                comment                 = comment,
                video_id                = vid_id,
                video_description       = video_desc,
                all_comments            = all_comments,
                sample_vecs             = sample_vecs,
                reference_specificities = reference_specificities,
                cfg                     = cfg,
            )

            all_rows.append({
                "source_type":       "extra",
                "source_label":      source_name,
                "id":                vid_id,
                "platform":          cfg.get("platform",""),
                "label":             item.get("label",""),
                "url":               item.get("url",""),
                "video_description": video_desc,
                "video_introduction":"",
                "exp_type":          "",
                "exp_name":          "",
                "ablation_variable": "",
                "ablation_pipeline": "",
                "gen_model":         source_name,
                "comment":           comment,
                "comment_length":    len(comment),
                **scores,
            })
    return all_rows


# ════════════════════════════════════════════════════════════════
#  保存结果 & 控制台报告
# ════════════════════════════════════════════════════════════════

SCORE_COLS = ["原创性", "具体性", "风格符合性", "总分"]


def save_and_print_report(all_rows: list, eval_dir: str, platform: str):
    os.makedirs(eval_dir, exist_ok=True)

    # 详细结果 JSON
    detail_path = os.path.join(eval_dir, "ablation_eval_detail.json")
    save_json(all_rows, detail_path)
    print(f"\n  ✅ 详细结果 → {detail_path}")

    df = pd.DataFrame(all_rows)

    # ── 消融实验汇总（按实验类型 × 生成模型）──────────────────
    abl_df = df[df["source_type"] == "ablation"]
    if not abl_df.empty:
        abl_sum = (
            abl_df.groupby(["exp_type","exp_name","gen_model"])[SCORE_COLS]
            .agg(["mean","std"]).round(3)
        )
        abl_sum.columns = ["_".join(c) for c in abl_sum.columns]
        abl_sum.reset_index().to_csv(
            os.path.join(eval_dir, "ablation_eval_summary.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"  ✅ 消融实验汇总 → {os.path.join(eval_dir, 'ablation_eval_summary.csv')}")

    # ── BASELINE 汇总（按生成模型）────────────────────────────
    base_df = df[df["source_type"] == "baseline"]
    if not base_df.empty:
        base_sum = (
            base_df.groupby("gen_model")[SCORE_COLS]
            .agg(["mean","std"]).round(3)
        )
        base_sum.columns = ["_".join(c) for c in base_sum.columns]
        base_sum.reset_index().to_csv(
            os.path.join(eval_dir, "baseline_eval_summary.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"  ✅ BASELINE 汇总 → {os.path.join(eval_dir, 'baseline_eval_summary.csv')}")

    # ── 额外评论汇总 ──────────────────────────────────────────
    ext_df = df[df["source_type"] == "extra"]
    if not ext_df.empty:
        ext_sum = (
            ext_df.groupby("source_label")[SCORE_COLS]
            .agg(["mean","std"]).round(3)
        )
        ext_sum.columns = ["_".join(c) for c in ext_sum.columns]
        ext_sum.reset_index().to_csv(
            os.path.join(eval_dir, "ablation_eval_extra.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"  ✅ 额外评论汇总 → {os.path.join(eval_dir, 'ablation_eval_extra.csv')}")

    # ── 三组统一对比表 ─────────────────────────────────────────
    combined_rows = []
    # 消融实验：实验类型 × 生成模型
    if not abl_df.empty:
        for (et, en, gm), grp in abl_df.groupby(["exp_type","exp_name","gen_model"]):
            row = {"实验组": f"{et}（{en}）", "生成模型": gm, "来源类型": "消融实验"}
            row.update(grp[SCORE_COLS].mean().round(3).to_dict())
            combined_rows.append(row)
    # BASELINE：生成模型
    if not base_df.empty:
        for gm, grp in base_df.groupby("gen_model"):
            row = {"实验组": "BASELINE（完整RAG）", "生成模型": gm, "来源类型": "BASELINE"}
            row.update(grp[SCORE_COLS].mean().round(3).to_dict())
            combined_rows.append(row)
    # 额外评论：来源文件
    if not ext_df.empty:
        for src, grp in ext_df.groupby("source_label"):
            row = {"实验组": f"额外评论（{src}）", "生成模型": src, "来源类型": "额外评论"}
            row.update(grp[SCORE_COLS].mean().round(3).to_dict())
            combined_rows.append(row)

    if combined_rows:
        pd.DataFrame(combined_rows).to_csv(
            os.path.join(eval_dir, "ablation_eval_combined.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"  ✅ 三组对比表 → {os.path.join(eval_dir, 'ablation_eval_combined.csv')}")

    # ────────────────────────────────────────────────────────
    #  控制台报告
    # ────────────────────────────────────────────────────────
    W = 20

    def _header(first_col, first_w):
        return f"  {first_col:<{first_w}}" + "".join(f"{c:<{W}}" for c in SCORE_COLS)

    def _data_row(label, row, first_w):
        cells = "".join(f"{row[c]:<{W}.2f}" for c in SCORE_COLS)
        bar   = "█" * int(row["总分"] * 2)
        return f"  {str(label):<{first_w}}{cells}  {bar}"

    print(f"\n\n{'★'*66}")
    print(f"  ★ 评估报告 — {platform.upper()}  （各维度满分 10，总分 = 三维度均值）")
    print(f"{'★'*66}")

    # ── ① BASELINE：按生成模型 ────────────────────────────────
    if not base_df.empty:
        print(f"\n  ① BASELINE — 完整 RAG Pipeline / 按生成模型")
        print(f"  {'─'*60}")
        gen_avg = base_df.groupby("gen_model")[SCORE_COLS].mean().round(2)
        print(_header("生成模型", 18))
        print("  " + "─" * (18 + W * len(SCORE_COLS)))
        for g, row in gen_avg.iterrows():
            print(_data_row(g, row, 18))

    # ── ② 消融实验：按实验类型（所有模型均值）─────────────────
    if not abl_df.empty:
        print(f"\n  ② 消融实验 — 按实验类型（所有生成模型均值）")
        print(f"  {'─'*60}")
        exp_avg = abl_df.groupby(["exp_type","exp_name"])[SCORE_COLS].mean().round(2)
        print(f"  {'实验类型':<10}{'实验名称':<26}" + "".join(f"{c:<{W}}" for c in SCORE_COLS))
        print("  " + "─" * (36 + W * len(SCORE_COLS)))
        for (et, en), row in exp_avg.iterrows():
            cells = "".join(f"{row[c]:<{W}.2f}" for c in SCORE_COLS)
            bar   = "█" * int(row["总分"] * 2)
            print(f"  {et:<10}{en:<26}{cells}  {bar}")

        print(f"\n  ② 消融实验 — 按生成模型（所有实验类型均值）")
        print(f"  {'─'*60}")
        gen_avg_abl = abl_df.groupby("gen_model")[SCORE_COLS].mean().round(2)
        print(_header("生成模型", 18))
        print("  " + "─" * (18 + W * len(SCORE_COLS)))
        for g, row in gen_avg_abl.iterrows():
            print(_data_row(g, row, 18))

    # ── ③ BASELINE vs 消融实验 整体对比 ───────────────────────
    if not base_df.empty and not abl_df.empty:
        print(f"\n  ③ BASELINE vs 消融实验 — 整体均值对比")
        print(f"  {'─'*60}")
        print(_header("来源", 30))
        print("  " + "─" * (30 + W * len(SCORE_COLS)))
        base_avg_row = base_df[SCORE_COLS].mean().round(2)
        abl_avg_row  = abl_df[SCORE_COLS].mean().round(2)
        print(_data_row("BASELINE（完整RAG，四模型均值）", base_avg_row, 30))
        print(_data_row("消融实验（EXP-1~4，四模型均值）", abl_avg_row, 30))

    # ── ④ 额外评论 ────────────────────────────────────────────
    if not ext_df.empty:
        print(f"\n  ④ 额外评论 — 按来源文件")
        print(f"  {'─'*60}")
        ext_avg = ext_df.groupby("source_label")[SCORE_COLS].mean().round(2)
        print(_header("来源", 28))
        print("  " + "─" * (28 + W * len(SCORE_COLS)))
        for src, row in ext_avg.iterrows():
            print(_data_row(src, row, 28))

    # ── ⑤ 消融实验最佳组合 ────────────────────────────────────
    if not abl_df.empty:
        best = (
            abl_df.groupby(["exp_type","gen_model"])["总分"]
            .mean().reset_index()
            .sort_values("总分", ascending=False)
        )
        print(f"\n  ⑤ 消融实验 — 最佳实验 × 生成模型 Top 8")
        print(f"  {'─'*60}")
        print(f"  {'实验类型':<12}{'生成模型':<22}{'总分均值'}")
        print("  " + "─" * 44)
        for _, r in best.head(8).iterrows():
            bar = "█" * int(r["总分"] * 2)
            print(f"  {r['exp_type']:<12}{r['gen_model']:<22}{r['总分']:.2f}  {bar}")

    print(f"\n  输出目录：{eval_dir}")


# ════════════════════════════════════════════════════════════════
#  平台交互选择
# ════════════════════════════════════════════════════════════════

def select_platform_interactive() -> str:
    print("\n" + "╔" + "═"*54 + "╗")
    print("║" + "  消融实验 + BASELINE 评论质量评估脚本".center(56) + "║")
    print("╚" + "═"*54 + "╝\n")
    print("  请选择目标平台：")
    print("    1. 🎵  抖音 (Douyin)")
    print("    2. 📺  YouTube\n")
    while True:
        c = input("  输入 1 或 2（或 douyin/youtube）：").strip().lower()
        if c in ("1","douyin"):  return "douyin"
        if c in ("2","youtube"): return "youtube"
        print("  ⚠️  无效输入")


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="消融实验 + BASELINE 多模型对比 评论质量评估",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
三种打分对象：
  ① BASELINE 多模型对比  baseline_results.json  （完整 RAG，四模型正常输出）
  ② 消融实验             ablation_all_results.json  （EXP-1/2/3/4）
  ③ 额外评论通道         --extra 指定的 JSON 文件

额外评论 JSON 格式（每文件是一个 list）：
  [{"video_id":"1","url":"1.mp4","label":"","comment":"...","video_description":"..."}, ...]

输出文件：
  ablation_eval_detail.json     所有条目详细得分
  ablation_eval_summary.csv     消融实验：实验类型 × 生成模型
  baseline_eval_summary.csv     BASELINE：按生成模型
  ablation_eval_extra.csv       额外评论（有 --extra 时）
  ablation_eval_combined.csv    三组统一对比表

示例：
  python ablation_evaluator.py --platform douyin
  python ablation_evaluator.py --platform douyin --extra GPT-4o.json LOLgorithm.json
  python ablation_evaluator.py --platform douyin --no-ablation --extra my.json
"""
    )
    parser.add_argument("--platform", choices=["douyin","youtube"], default=None)
    parser.add_argument("--input", default=None, metavar="FILE",
                        help="消融实验结果文件（默认自动推断）")
    parser.add_argument("--baseline-input", default=None, metavar="FILE",
                        help="BASELINE 结果文件（默认自动推断）")
    parser.add_argument("--extra", nargs="+", default=[], metavar="FILE",
                        help="额外评论 JSON 文件（可多个，文件名作为来源标签）")
    parser.add_argument("--sample-files", nargs="+", default=None, metavar="FILE",
                        help="学习样本文件（可多个，覆盖默认）")
    parser.add_argument("--reference-file", default=None, metavar="FILE",
                        help="参考基线文件（覆盖默认）")
    parser.add_argument("--no-ablation", action="store_true",
                        help="跳过消融实验结果评分")
    parser.add_argument("--no-baseline", action="store_true",
                        help="跳过 BASELINE 多模型对比评分")
    parser.add_argument("--eval-dir", default=None, metavar="DIR",
                        help="输出目录（覆盖默认）")

    args = parser.parse_args()

    # ── 平台 & 配置 ───────────────────────────────────────────
    platform = args.platform or select_platform_interactive()
    cfg      = dict(PLATFORM_CFG[platform])
    cfg["platform"] = platform

    ablation_path  = args.input          or cfg["ablation_all"]
    baseline_path  = args.baseline_input or cfg["baseline_file"]
    reference_file = args.reference_file or cfg["reference_file"]
    sample_files   = args.sample_files   or cfg["sample_files"]
    eval_dir       = args.eval_dir       or cfg["eval_dir"]

    # ── 文件存在性检查 ─────────────────────────────────────────
    run_ablation = not args.no_ablation and os.path.exists(ablation_path)
    run_baseline = not args.no_baseline and os.path.exists(baseline_path)

    if not args.no_ablation and not os.path.exists(ablation_path):
        print(f"  [警告] 消融实验文件不存在：{ablation_path}，跳过消融实验评分")
    if not args.no_baseline and not os.path.exists(baseline_path):
        print(f"  [警告] BASELINE 文件不存在：{baseline_path}，跳过 BASELINE 评分")

    if not run_ablation and not run_baseline and not args.extra:
        print("  [错误] 无任何可评估的数据，请检查输入文件。"); return

    # ── 打印配置 ──────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print(f"  平台              ：{platform.upper()}")
    print(f"  消融实验文件      ：{'跳过' if not run_ablation else ablation_path}")
    print(f"  BASELINE 文件     ：{'跳过' if not run_baseline else baseline_path}")
    print(f"  额外评论文件      ：{args.extra if args.extra else '[无]'}")
    print(f"  输出目录          ：{eval_dir}")
    print(f"{'═'*64}\n")

    # ── 初始化模型 ────────────────────────────────────────────
    get_sbert()
    get_sentiment_pipeline()

    # ── 预加载数据 ────────────────────────────────────────────
    sample_vecs             = load_sample_embeddings(sample_files)
    reference_specificities = load_reference_specificities(reference_file)

    # ── 评分 ──────────────────────────────────────────────────
    all_rows = []

    # ① BASELINE
    if run_baseline:
        records = load_json(baseline_path)
        print(f"\n  ── ① BASELINE 评分（{len(records)} 条视频 × 4 模型）")
        rows = evaluate_records(records, sample_vecs, reference_specificities,
                                cfg, source_type="baseline", desc="BASELINE")
        all_rows.extend(rows)
        print(f"  ✅ BASELINE 打分完成（{len(rows)} 条）")

    # ② 消融实验
    if run_ablation:
        records = load_json(ablation_path)
        print(f"\n  ── ② 消融实验评分（{len(records)} 条视频 × 4 模型）")
        rows = evaluate_records(records, sample_vecs, reference_specificities,
                                cfg, source_type="ablation", desc="消融实验")
        all_rows.extend(rows)
        print(f"  ✅ 消融实验打分完成（{len(rows)} 条）")

    # ③ 额外评论
    if args.extra:
        ext_rows = evaluate_extra_comments(
            args.extra, sample_vecs, reference_specificities, cfg)
        all_rows.extend(ext_rows)
        print(f"  ✅ 额外评论打分完成（{len(ext_rows)} 条）")

    if not all_rows:
        print("\n  [警告] 没有任何评论参与评分"); return

    # ── 保存 & 报告 ───────────────────────────────────────────
    save_and_print_report(all_rows, eval_dir, platform)


if __name__ == "__main__":
    main()