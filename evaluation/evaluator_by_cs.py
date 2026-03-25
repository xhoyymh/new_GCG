"""
评论质量统一评估脚本
====================

一次调用同时评估以下所有评论来源：

  ┌─── 抖音 (Douyin) ────────────────────────────────────────────┐
  │  外部系统：evaluation/baseline/douyin/                        │
  │    douyin_comments_GPT-4o.json                                │
  │    douyin_comments_LOLgorithm.json                            │
  │    douyin_comments_V2Xum-LLM.json                             │
  │    douyin_comments_livechat.json                              │
  │  消融实验：ablation_results/douyin/ablation_all_results.json  │
  │  BASELINE ：ablation_results/douyin/baseline_results.json     │
  └───────────────────────────────────────────────────────────────┘

  ┌─── YouTube ───────────────────────────────────────────────────┐
  │  外部系统：evaluation/baseline/youtube/                       │
  │    youtube_comments_GPT-4o.json                               │
  │    youtube_comments_LOLgorithm.json                            │
  │    youtube_comments_V2Xum-LLM.json                            │
  │    youtube_comments_livechat.json                             │
  │  消融实验：ablation_results/youtube/ablation_all_results.json │
  │  BASELINE ：ablation_results/youtube/baseline_results.json    │
  └───────────────────────────────────────────────────────────────┘

参考样本（原创性 & 具体性基线）—— 抖音和 YouTube 各自独立子目录：
  evaluation/original_comment_from_platform/
    douyin/    ← 抖音原评论，目录内所有 .json 文件均会被读取
    youtube/   ← YouTube 原评论，同上

评分维度（各满分 10，总分 = 三维度均值）：
  原创性     与原平台评论的语义相似度（越低越好 → 得分越高）
             惩罚：含总结关键词 -3，批次内重复 -3
  具体性     当前评论与视频描述相关度 vs 参考人类基线的差距
  风格符合性 情感一致性（10/5）- 长度惩罚（0~5），重复时压至 1

输出（所有平台合并到一个 JSON，CSV 按平台分文件）：
  evaluation/result/
    all_eval_detail.json           ← 全平台全来源所有条目详细得分
    douyin_eval_summary.csv        ← 抖音：按来源 × 模型汇总
    youtube_eval_summary.csv       ← YouTube：按来源 × 模型汇总
    douyin_eval_combined.csv       ← 抖音：各来源均值对比表
    youtube_eval_combined.csv      ← YouTube：各来源均值对比表

用法：
  python evaluator.py                          # 交互式选择平台（推荐）
  python evaluator.py --platform douyin        # 直接指定抖音
  python evaluator.py --platform youtube       # 直接指定 YouTube
  python evaluator.py --platform both          # 两个平台都评，分开输出
  python evaluator.py --platform douyin --no-ablation   # 抖音，跳过消融实验
  python evaluator.py --no-baseline            # 跳过 BASELINE 评分
  python evaluator.py --no-external            # 跳过外部系统评分
  python evaluator.py --eval-dir path/to/out   # 自定义输出目录
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# ════════════════════════════════════════════════════════════════
#  ★ 路径配置
# ════════════════════════════════════════════════════════════════

BASE_DIR = r"D:\Desktop\video_comment_generation\ALLinone"

# 外部对比系统评论文件夹
EXTERNAL_DIRS = {
    "douyin":  os.path.join(BASE_DIR, "evaluation", "baseline", "douyin"),
    "youtube": os.path.join(BASE_DIR, "evaluation", "baseline", "youtube"),
}

# 消融实验结果
ABLATION_FILES = {
    "douyin":  os.path.join(BASE_DIR, "ablation_results", "douyin", "ablation_all_results.json"),
    "youtube": os.path.join(BASE_DIR, "ablation_results", "youtube", "ablation_all_results.json"),
}

# BASELINE 多模型对比结果
BASELINE_FILES = {
    "douyin":  os.path.join(BASE_DIR, "ablation_results", "douyin", "baseline_results.json"),
    "youtube": os.path.join(BASE_DIR, "ablation_results", "youtube", "baseline_results.json"),
}

# 原平台评论（参考基线）— 抖音和 YouTube 各自独立子目录
ORIGINAL_COMMENT_DIRS = {
    "douyin":  os.path.join(BASE_DIR, "evaluation", "original_comment_from_platform", "douyin"),
    "youtube": os.path.join(BASE_DIR, "evaluation", "original_comment_from_platform", "youtube"),
}

# 输出目录
DEFAULT_EVAL_DIR = os.path.join(BASE_DIR, "evaluation", "result")

# 消融/BASELINE 记录中四个模型的评论字段
GENERATED_COMMENT_FIELDS = [
    "qwen3.5_generated_comment",
    "glm_generated_comment",
    "deepseek-r1_generated_comment",
    "minimax_generated_comment",
]

# ── 平台专属评分参数 ──────────────────────────────────────────────
PLATFORM_PARAMS = {
    "douyin": {
        "ideal_length": 30,       # 理想评论长度（字符数）
        "summary_kws":  ["视频", "画面", "场景", "描述", "故事"],
    },
    "youtube": {
        "ideal_length": 72,
        "summary_kws":  ["video", "scene", "description", "story", "footage"],
    },
}

# ── 外部系统文件名 → 来源标签 ─────────────────────────────────────
# 文件名中包含以下字符串时，映射到对应的标签（方便报告显示）
SOURCE_NAME_MAP = {
    "GPT-4o":     "GPT-4o",
    "LOLgorithm": "LOLgorithm",
    "V2Xum-LLM":  "V2Xum-LLM",
    "livechat":   "LiveChat",
}

# ════════════════════════════════════════════════════════════════
#  模型懒加载（全局共享，只初始化一次）
# ════════════════════════════════════════════════════════════════

SBERT_MODEL_NAME     = "all-MiniLM-L6-v2"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_sbert     = None
_sentiment = None


def get_sbert() -> SentenceTransformer:
    global _sbert
    if _sbert is None:
        print(f"  🔄 加载语义模型 {SBERT_MODEL_NAME} ...")
        _sbert = SentenceTransformer(SBERT_MODEL_NAME)
        print(f"  ✅ 语义模型已加载")
    return _sbert


def get_sentiment():
    global _sentiment
    if _sentiment is None:
        print(f"  🔄 加载情感模型 {SENTIMENT_MODEL_NAME} ...")
        _sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
        print(f"  ✅ 情感模型已加载")
    return _sentiment


def sentiment_label(text: str) -> str:
    try:
        return get_sentiment()(text[:512])[0]["label"]
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


def extract_comment(item: dict) -> str:
    """兼容不同来源文件的评论字段名。"""
    for key in ("comment", "comment_1", "generated_comment"):
        val = item.get(key, "")
        if val and str(val).strip():
            return str(val).strip()
    return ""


def extract_video_id(item: dict) -> str:
    """兼容 video_id / id 两种字段名。"""
    return str(item.get("video_id", item.get("id", ""))).strip()


def extract_video_desc(item: dict) -> str:
    return item.get("video_description", "").strip()


def map_source_name(filename: str) -> str:
    """根据文件名推断标准来源标签。"""
    base = os.path.splitext(os.path.basename(filename))[0]
    for pattern, label in SOURCE_NAME_MAP.items():
        if pattern.lower() in base.lower():
            return label
    return base


def find_original_comment_files(platform: str) -> list:
    """
    返回指定平台原评论目录下所有 .json 文件的路径列表。
    目录结构：
      evaluation/original_comment_from_platform/
        douyin/    ← 抖音原评论（本函数 platform="douyin" 时读取此目录）
        youtube/   ← YouTube 原评论（platform="youtube" 时读取此目录）
    目录不存在时返回空列表并给出警告。
    """
    target_dir = ORIGINAL_COMMENT_DIRS.get(platform, "")
    if not target_dir or not os.path.isdir(target_dir):
        print(f"  [警告] 原评论目录不存在：{target_dir or '（未配置）'}")
        return []

    files = [
        os.path.join(target_dir, fn)
        for fn in sorted(os.listdir(target_dir))
        if fn.endswith(".json")
    ]
    if not files:
        print(f"  [警告] 原评论目录为空，无 .json 文件：{target_dir}")
    else:
        print(f"  📂 原评论目录：{target_dir}（{len(files)} 个文件）")
    return files


# ════════════════════════════════════════════════════════════════
#  参考数据预加载
# ════════════════════════════════════════════════════════════════

def load_sample_embeddings(ref_file_paths: list) -> np.ndarray:
    """
    从原平台评论文件中提取所有评论文本并编码。
    兼容格式：
      comment_1~5 字段（样本格式）
      comment 字段（外部系统格式 / 原评论格式）
      comments 列表
    """
    model = get_sbert()
    texts = []
    for fp in ref_file_paths:
        if not os.path.exists(fp):
            print(f"  [警告] 样本文件不存在，跳过：{fp}")
            continue
        data = load_json(fp)
        for item in data:
            # comment_1 ~ comment_5
            for i in range(1, 6):
                c = item.get(f"comment_{i}", "").strip()
                if c:
                    texts.append(c)
            # comments 列表
            if isinstance(item.get("comments"), list):
                for c_obj in item["comments"]:
                    c = c_obj.get("content", c_obj.get("comment", "")).strip()
                    if c:
                        texts.append(c)
            # 直接 comment 字段
            c = item.get("comment", "").strip()
            if c and c not in texts:
                texts.append(c)

    if not texts:
        print("  [警告] 未找到样本评论，原创性得分将为 0")
        return np.array([])

    print(f"  🔄 编码 {len(texts)} 条原平台评论 ...")
    vecs = model.encode(texts, show_progress_bar=False, batch_size=64)
    print(f"  ✅ 完成（{len(texts)} 条）")
    return vecs


def load_reference_specificities(ref_file_paths: list) -> dict:
    """
    计算每个 video_id 的「人类真实评论 vs 视频描述」平均语义相似度。
    作为具体性评分的参考基线。
    同一 video_id 出现在多个文件时取均值。
    """
    model = get_sbert()
    raw: dict[str, list] = {}   # video_id -> [sim, sim, ...]

    for fp in ref_file_paths:
        if not os.path.exists(fp):
            continue
        data = load_json(fp)
        print(f"  🔄 计算具体性基线 ← {os.path.basename(fp)}（{len(data)} 条）")

        for item in tqdm(data, desc="  具体性基线", leave=False):
            vid  = extract_video_id(item)
            desc = extract_video_desc(item)
            if not vid or not desc:
                continue
            desc_vec = model.encode([desc])

            sims = []
            for i in range(1, 6):
                c = item.get(f"comment_{i}", "").strip()
                if c:
                    sims.append(cosine_similarity(model.encode([c]), desc_vec).item())
            if isinstance(item.get("comments"), list):
                for c_obj in item["comments"]:
                    c = c_obj.get("content", c_obj.get("comment", "")).strip()
                    if c:
                        sims.append(cosine_similarity(model.encode([c]), desc_vec).item())
            # 直接 comment 字段
            c = item.get("comment", "").strip()
            if c:
                sims.append(cosine_similarity(model.encode([c]), desc_vec).item())

            if sims:
                raw.setdefault(vid, []).extend(sims)

    ref_map = {vid: float(np.mean(sims)) for vid, sims in raw.items()}
    print(f"  ✅ 具体性基线完成（{len(ref_map)} 个视频）")
    return ref_map


# ════════════════════════════════════════════════════════════════
#  三个评分维度
# ════════════════════════════════════════════════════════════════

def score_originality(comment: str, comment_vec: np.ndarray,
                      sample_vecs: np.ndarray, all_comments: list,
                      summary_kws: list) -> float:
    """
    原创性（0~10）：
      基础分 = 10 - max_sim_to_samples × 10
      -3 若含总结性关键词
      -3 若在批次中重复
    """
    if sample_vecs.size > 0:
        sim = float(cosine_similarity(comment_vec, sample_vecs).max())
    else:
        sim = 0.0

    score = 10.0 - sim * 10.0
    if any(kw in comment for kw in summary_kws):
        score -= 3.0
    if all_comments.count(comment) > 1:
        score -= 3.0
    return round(max(0.0, score), 2)


def score_specificity(comment_vec: np.ndarray, video_description: str,
                      video_id: str, ref_spec: dict) -> float:
    """
    具体性（0~10）：
      specificity = 10 - |current_sim - ref_sim| × 10
      若 video_id 不在参考基线中，返回 0。
    """
    model = get_sbert()
    if not video_description.strip() or video_id not in ref_spec:
        return 0.0
    desc_vec    = model.encode([video_description])
    current_sim = cosine_similarity(comment_vec, desc_vec).item()
    ref_sim     = ref_spec[video_id]
    return round(max(0.0, 10.0 - abs(current_sim - ref_sim) * 10.0), 2)


def score_style(comment: str, comment_vec: np.ndarray,
                video_description: str, all_comments: list,
                ideal_length: int) -> float:
    """
    风格符合性（0~10）：
      情感得分（10 若一致，5 若不同）- 长度惩罚（0~5）
      语义重复（max_sim > 0.75）或完全重复时，情感分压至 1
    """
    model = get_sbert()

    # 长度惩罚
    length_deduction = min(abs(len(comment) - ideal_length) / max(ideal_length, 1) * 5.0, 5.0)

    # 情感一致性
    try:
        sent_score = 10.0 if sentiment_label(comment) == sentiment_label(video_description) else 5.0
    except Exception:
        sent_score = 5.0

    # 重复检测
    is_repeat = all_comments.count(comment) > 1
    if len(all_comments) > 1:
        others  = [c for c in all_comments if c != comment] or all_comments
        max_sim = float(cosine_similarity(
            comment_vec, model.encode(others, show_progress_bar=False))[0].max())
        is_sem_rep = max_sim > 0.75
    else:
        is_sem_rep = False

    if is_repeat or is_sem_rep:
        sent_score = min(sent_score, 1.0)

    return round(max(0.0, min(10.0, sent_score - length_deduction)), 2)


def score_comment(comment: str, video_id: str, video_description: str,
                  all_comments: list, sample_vecs: np.ndarray,
                  ref_spec: dict, params: dict) -> dict:
    """对单条评论打全部三维度的分。总分 = 三维度均值（满分 10）。"""
    model = get_sbert()

    if not comment:
        return {"原创性": 0.0, "具体性": 0.0, "风格符合性": 0.0, "总分": 0.0}

    comment_vec = model.encode([comment])
    orig  = score_originality(comment, comment_vec, sample_vecs,
                              all_comments, params["summary_kws"])
    spec  = score_specificity(comment_vec, video_description, video_id, ref_spec)
    style = score_style(comment, comment_vec, video_description,
                        all_comments, params["ideal_length"])

    return {
        "原创性":    orig,
        "具体性":    spec,
        "风格符合性": style,
        "总分":      round((orig + spec + style) / 3.0, 2),
    }


# ════════════════════════════════════════════════════════════════
#  各来源批量打分
# ════════════════════════════════════════════════════════════════

def eval_external_system(data: list, source_name: str, platform: str,
                         sample_vecs: np.ndarray, ref_spec: dict,
                         params: dict) -> list:
    """
    评分外部系统评论（GPT-4o / LOLgorithm / V2Xum-LLM / livechat）。

    兼容字段：
      video_id / id        → 视频 ID
      comment              → 评论文本
      video_description    → 视频描述（用于具体性计算）
      label / url          → 可选元数据
    """
    all_comments = [extract_comment(x) for x in data if extract_comment(x)]
    rows = []

    for item in tqdm(data, desc=f"  {source_name}", unit="条"):
        vid    = extract_video_id(item)
        comment= extract_comment(item)
        desc   = extract_video_desc(item)

        scores = score_comment(comment, vid, desc, all_comments,
                               sample_vecs, ref_spec, params)
        rows.append({
            # 来源标注
            "platform":          platform,
            "source_type":       "external",
            "source_label":      source_name,
            "exp_type":          "",
            "exp_name":          "",
            "gen_model":         source_name,
            # 视频元信息
            "video_id":          vid,
            "label":             item.get("label", ""),
            "video_description": desc,
            "video_introduction":item.get("video_introduction", ""),
            # 评论
            "comment":           comment,
            "comment_length":    len(comment),
            # 得分
            **scores,
        })
    return rows


def eval_ablation_or_baseline(records: list, source_type: str,
                               platform: str, sample_vecs: np.ndarray,
                               ref_spec: dict, params: dict) -> list:
    """
    评分消融实验（EXP-1/2/3/4）或 BASELINE 记录。
    每条记录含四个模型字段，各打一次分。
    """
    # 收集全批次所有生成评论（重复检测用）
    all_batch = [
        rec.get(f, "").strip()
        for rec in records
        for f in GENERATED_COMMENT_FIELDS
        if rec.get(f, "").strip()
    ]

    rows = []
    desc_label = "BASELINE" if source_type == "baseline" else "消融实验"

    for rec in tqdm(records, desc=f"  {desc_label}({platform})", unit="视频"):
        vid      = str(rec.get("id", "")).strip()
        exp_type = rec.get("ablation_exp_type", "")
        exp_name = rec.get("ablation_exp_name", "")
        desc     = rec.get("video_description", "")
        intro    = rec.get("video_introduction", "")
        label    = rec.get("label", "")

        for field in GENERATED_COMMENT_FIELDS:
            gen_model = field.replace("_generated_comment", "")
            comment   = rec.get(field, "").strip()

            scores = score_comment(comment, vid, desc, all_batch,
                                   sample_vecs, ref_spec, params)
            rows.append({
                "platform":          platform,
                "source_type":       source_type,
                "source_label":      f"{exp_type}/{gen_model}" if exp_type else f"BASELINE/{gen_model}",
                "exp_type":          exp_type,
                "exp_name":          exp_name,
                "gen_model":         gen_model,
                "video_id":          vid,
                "label":             label,
                "video_description": desc,
                "video_introduction":intro,
                "comment":           comment,
                "comment_length":    len(comment),
                **scores,
            })
    return rows


# ════════════════════════════════════════════════════════════════
#  输出：保存 & 报告
# ════════════════════════════════════════════════════════════════

SCORE_COLS = ["原创性", "具体性", "风格符合性", "总分"]


def save_results(all_rows: list, eval_dir: str):
    """
    保存全部结果：
      all_eval_detail.json           ← 所有平台所有来源明细
      {platform}_eval_summary.csv   ← 按来源均值汇总（含 std）
      {platform}_eval_combined.csv  ← 各来源一行对比表
    """
    os.makedirs(eval_dir, exist_ok=True)

    # ── 全量明细 JSON ──────────────────────────────────────────
    all_json_path = os.path.join(eval_dir, "all_eval_detail.json")
    save_json(all_rows, all_json_path)
    print(f"\n  ✅ 全量明细 → {all_json_path}  （{len(all_rows)} 条记录）")

    df = pd.DataFrame(all_rows)

    for platform in ["douyin", "youtube"]:
        plat_df = df[df["platform"] == platform]
        if plat_df.empty:
            continue

        prefix = os.path.join(eval_dir, platform)

        # ── 汇总 CSV（来源 × 维度，含 mean & std）──────────────
        grp = plat_df.groupby(["source_type", "source_label", "gen_model"])
        summary_rows = []
        for (stype, slabel, gmodel), g in grp:
            row = {
                "来源类型":  stype,
                "来源标签":  slabel,
                "生成模型":  gmodel,
            }
            for col in SCORE_COLS:
                row[f"{col}_mean"] = round(g[col].mean(), 3)
                row[f"{col}_std"]  = round(g[col].std(),  3)
            row["样本数"] = len(g)
            summary_rows.append(row)

        pd.DataFrame(summary_rows).to_csv(
            f"{prefix}_eval_summary.csv", index=False, encoding="utf-8-sig"
        )
        print(f"  ✅ {platform.upper()} 汇总 → {prefix}_eval_summary.csv")

        # ── 对比表 CSV（每个来源一行，方便论文直接引用）──────────
        # 分组粒度：source_label（同一来源多模型则先均值）
        combined_rows = []

        # 外部系统（每个文件一个 source_label）
        ext = plat_df[plat_df["source_type"] == "external"]
        if not ext.empty:
            for slabel, g in ext.groupby("source_label"):
                row = {"来源类型": "外部系统", "来源": slabel}
                row.update(g[SCORE_COLS].mean().round(3).to_dict())
                combined_rows.append(row)

        # BASELINE（按生成模型拆分）
        base = plat_df[plat_df["source_type"] == "baseline"]
        if not base.empty:
            for gm, g in base.groupby("gen_model"):
                row = {"来源类型": "BASELINE", "来源": f"BASELINE/{gm}"}
                row.update(g[SCORE_COLS].mean().round(3).to_dict())
                combined_rows.append(row)
            # 全 BASELINE 均值行
            row = {"来源类型": "BASELINE", "来源": "BASELINE（四模型均值）"}
            row.update(base[SCORE_COLS].mean().round(3).to_dict())
            combined_rows.append(row)

        # 消融实验（按实验类型 × 生成模型）
        abl = plat_df[plat_df["source_type"] == "ablation"]
        if not abl.empty:
            for (et, gm), g in abl.groupby(["exp_type", "gen_model"]):
                row = {"来源类型": "消融实验", "来源": f"{et}/{gm}"}
                row.update(g[SCORE_COLS].mean().round(3).to_dict())
                combined_rows.append(row)
            # 各实验类型均值行
            for et, g in abl.groupby("exp_type"):
                row = {"来源类型": "消融实验", "来源": f"{et}（四模型均值）"}
                row.update(g[SCORE_COLS].mean().round(3).to_dict())
                combined_rows.append(row)

        pd.DataFrame(combined_rows).to_csv(
            f"{prefix}_eval_combined.csv", index=False, encoding="utf-8-sig"
        )
        print(f"  ✅ {platform.upper()} 对比表 → {prefix}_eval_combined.csv")


def print_report(all_rows: list):
    """控制台输出双平台评估报告。"""
    df = pd.DataFrame(all_rows)
    W  = 22

    def _hdr(first, fw):
        return f"  {first:<{fw}}" + "".join(f"{c:<{W}}" for c in SCORE_COLS)

    def _row(label, g, fw):
        avgs = g[SCORE_COLS].mean()
        bar  = "█" * int(avgs["总分"] * 2)
        return f"  {str(label):<{fw}}" + "".join(f"{avgs[c]:<{W}.2f}" for c in SCORE_COLS) + f"  {bar}"

    for platform in ["douyin", "youtube"]:
        plat_df = df[df["platform"] == platform]
        if plat_df.empty:
            continue

        print(f"\n\n{'★'*70}")
        print(f"  ★ 评估报告 — {platform.upper()}  （各维度满分 10，总分 = 三维度均值）")
        print(f"{'★'*70}")

        # ① 外部系统
        ext = plat_df[plat_df["source_type"] == "external"]
        if not ext.empty:
            print(f"\n  ① 外部对比系统（GPT-4o / LOLgorithm / V2Xum-LLM / LiveChat）")
            print(f"  {'─'*65}")
            print(_hdr("来源系统", 20))
            print("  " + "─" * (20 + W * len(SCORE_COLS)))
            for slabel, g in ext.groupby("source_label"):
                print(_row(slabel, g, 20))

        # ② BASELINE
        base = plat_df[plat_df["source_type"] == "baseline"]
        if not base.empty:
            print(f"\n  ② BASELINE — 完整 RAG / 按生成模型")
            print(f"  {'─'*65}")
            print(_hdr("生成模型", 22))
            print("  " + "─" * (22 + W * len(SCORE_COLS)))
            for gm, g in base.groupby("gen_model"):
                print(_row(gm, g, 22))
            # 均值行
            print("  " + "─" * (22 + W * len(SCORE_COLS)))
            print(_row("【BASELINE 四模型均值】", base, 22))

        # ③ 消融实验：按实验类型
        abl = plat_df[plat_df["source_type"] == "ablation"]
        if not abl.empty:
            print(f"\n  ③ 消融实验 — 按实验类型（四模型均值）")
            print(f"  {'─'*65}")
            print(f"  {'实验类型':<10}{'实验名称':<28}" + "".join(f"{c:<{W}}" for c in SCORE_COLS))
            print("  " + "─" * (38 + W * len(SCORE_COLS)))
            for (et, en), g in abl.groupby(["exp_type", "exp_name"]):
                avgs = g[SCORE_COLS].mean()
                bar  = "█" * int(avgs["总分"] * 2)
                print(f"  {et:<10}{en:<28}" +
                      "".join(f"{avgs[c]:<{W}.2f}" for c in SCORE_COLS) + f"  {bar}")

            print(f"\n  ③ 消融实验 — 按生成模型（四实验类型均值）")
            print(f"  {'─'*65}")
            print(_hdr("生成模型", 22))
            print("  " + "─" * (22 + W * len(SCORE_COLS)))
            for gm, g in abl.groupby("gen_model"):
                print(_row(gm, g, 22))

        # ④ 整体对比（外部 vs BASELINE vs 消融实验）
        if not ext.empty or not base.empty or not abl.empty:
            print(f"\n  ④ 整体均值对比（各来源类型）")
            print(f"  {'─'*65}")
            print(_hdr("来源类型", 30))
            print("  " + "─" * (30 + W * len(SCORE_COLS)))
            if not ext.empty:
                print(_row("外部对比系统（四系统均值）", ext, 30))
            if not base.empty:
                print(_row("BASELINE（完整RAG，四模型均值）", base, 30))
            if not abl.empty:
                print(_row("消融实验（EXP-1~4，四模型均值）", abl, 30))

        # ⑤ 消融实验最佳组合 Top8
        if not abl.empty:
            best = (
                abl.groupby(["exp_type", "gen_model"])["总分"]
                .mean().reset_index()
                .sort_values("总分", ascending=False)
            )
            print(f"\n  ⑤ 消融实验 — 最佳实验 × 生成模型 Top 8")
            print(f"  {'─'*65}")
            print(f"  {'实验类型':<12}{'生成模型':<24}{'总分均值'}")
            print("  " + "─" * 50)
            for _, r in best.head(8).iterrows():
                bar = "█" * int(r["总分"] * 2)
                print(f"  {r['exp_type']:<12}{r['gen_model']:<24}{r['总分']:.2f}  {bar}")


# ════════════════════════════════════════════════════════════════
#  ★ 单平台评估流程
# ════════════════════════════════════════════════════════════════

def evaluate_platform(
    platform:      str,
    run_external:  bool,
    run_baseline:  bool,
    run_ablation:  bool,
) -> list:
    """
    对单个平台执行完整评估流程，返回该平台的所有得分行。
    """
    params = PLATFORM_PARAMS[platform]
    print(f"\n{'═'*70}")
    print(f"  开始评估平台：{platform.upper()}")
    print(f"{'═'*70}")

    # ── 加载参考数据 ──────────────────────────────────────────
    ref_files = find_original_comment_files(platform)
    if not ref_files:
        print(f"  [警告] {platform} 原评论目录为空或不存在：{ORIGINAL_COMMENT_DIRS.get(platform, '')}")
        print(f"  将使用空参考（具体性得分为 0，原创性基线为空）")

    sample_vecs = load_sample_embeddings(ref_files)
    ref_spec    = load_reference_specificities(ref_files)

    all_rows = []

    # ── ① 外部系统评论 ────────────────────────────────────────
    if run_external:
        ext_dir = EXTERNAL_DIRS[platform]
        if os.path.isdir(ext_dir):
            json_files = sorted(
                [f for f in os.listdir(ext_dir) if f.endswith(".json")]
            )
            if json_files:
                print(f"\n  ── 外部对比系统（{len(json_files)} 个文件）")
                for fn in json_files:
                    fp          = os.path.join(ext_dir, fn)
                    source_name = map_source_name(fn)
                    data        = load_json(fp)
                    print(f"  📄 {fn} → [{source_name}]（{len(data)} 条）")
                    rows = eval_external_system(
                        data, source_name, platform,
                        sample_vecs, ref_spec, params
                    )
                    all_rows.extend(rows)
                    print(f"     ✅ 评分完成（{len(rows)} 条）")
            else:
                print(f"  [警告] 外部系统目录为空：{ext_dir}")
        else:
            print(f"  [警告] 外部系统目录不存在：{ext_dir}")

    # ── ② BASELINE 多模型对比 ─────────────────────────────────
    if run_baseline:
        fp = BASELINE_FILES[platform]
        if os.path.exists(fp):
            records = load_json(fp)
            print(f"\n  ── BASELINE 多模型对比（{len(records)} 条视频 × {len(GENERATED_COMMENT_FIELDS)} 模型）")
            rows = eval_ablation_or_baseline(
                records, "baseline", platform, sample_vecs, ref_spec, params
            )
            all_rows.extend(rows)
            print(f"     ✅ 评分完成（{len(rows)} 条）")
        else:
            print(f"  [警告] BASELINE 文件不存在，跳过：{fp}")

    # ── ③ 消融实验（EXP-1/2/3/4）────────────────────────────
    if run_ablation:
        fp = ABLATION_FILES[platform]
        if os.path.exists(fp):
            records = load_json(fp)
            print(f"\n  ── 消融实验（{len(records)} 条视频 × {len(GENERATED_COMMENT_FIELDS)} 模型）")
            rows = eval_ablation_or_baseline(
                records, "ablation", platform, sample_vecs, ref_spec, params
            )
            all_rows.extend(rows)
            print(f"     ✅ 评分完成（{len(rows)} 条）")
        else:
            print(f"  [警告] 消融实验文件不存在，跳过：{fp}")

    print(f"\n  {platform.upper()} 全部评分完成，共 {len(all_rows)} 条记录")
    return all_rows


# ════════════════════════════════════════════════════════════════
#  ★ 交互式平台选择
# ════════════════════════════════════════════════════════════════

def select_platform_interactive() -> str:
    """
    启动时交互式询问用户要评估哪个平台。
    返回 "douyin"、"youtube" 或 "both"。
    """
    print(f"\n{'╔'+'═'*60+'╗'}")
    print(f"║{'  评论质量评估脚本'.center(62)}║")
    print(f"{'╚'+'═'*60+'╝'}\n")
    print("  请选择要评估的平台：\n")
    print("    1.  🎵  抖音 (Douyin)")
    print("    2.  📺  YouTube")
    print("    3.  🎵📺  两个平台都评（分开输出）\n")

    mapping = {
        "1": "douyin",  "douyin":  "douyin",
        "2": "youtube", "youtube": "youtube",
        "3": "both",    "both":    "both",
    }
    while True:
        choice = input("  请输入选项（1 / 2 / 3）：").strip().lower()
        if choice in mapping:
            selected = mapping[choice]
            labels   = {"douyin": "抖音", "youtube": "YouTube", "both": "抖音 + YouTube"}
            print(f"\n  ✅ 已选择：{labels[selected]}\n")
            return selected
        print("  ⚠️  无效输入，请输入 1、2 或 3")


# ════════════════════════════════════════════════════════════════
#  ★ 主入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="评论质量评估脚本（支持抖音 / YouTube / 两者分开评估）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
平台选择：
  不指定 --platform 时进入交互式选择菜单。
  --platform douyin   仅评估抖音
  --platform youtube  仅评估 YouTube
  --platform both     两个平台都评，分开输出

评估来源（默认全部开启）：
  外部系统    evaluation/baseline/{platform}/ 下的所有 .json 文件
              包含 GPT-4o / LOLgorithm / V2Xum-LLM / livechat
  BASELINE    ablation_results/{platform}/baseline_results.json
              完整 RAG Pipeline，四模型正常输出
  消融实验    ablation_results/{platform}/ablation_all_results.json
              EXP-1/2/3/4，各四模型

参考基线（原平台评论）—— 两平台各自独立子目录：
  evaluation/original_comment_from_platform/
    douyin/    ← 该目录下所有 .json 用于抖音打分
    youtube/   ← 该目录下所有 .json 用于 YouTube 打分

输出文件（evaluation/result/）：
  all_eval_detail.json           全来源所有条目明细（一个 JSON）
  douyin_eval_summary.csv        抖音：来源 × 模型 均值/std 汇总
  youtube_eval_summary.csv       YouTube：来源 × 模型 均值/std 汇总
  douyin_eval_combined.csv       抖音：各来源一行对比表
  youtube_eval_combined.csv      YouTube：各来源一行对比表

示例：
  python evaluator.py                        # 交互选平台，全量评估
  python evaluator.py --platform douyin      # 直接指定抖音
  python evaluator.py --platform youtube     # 直接指定 YouTube
  python evaluator.py --platform both        # 两个平台都评
  python evaluator.py --platform douyin --no-ablation   # 抖音，跳过消融实验
"""
    )
    parser.add_argument(
        "--platform",
        choices=["douyin", "youtube", "both"],
        default=None,
        metavar="PLATFORM",
        help="要评估的平台：douyin / youtube / both（不指定则交互选择）",
    )
    parser.add_argument("--no-external", action="store_true", help="跳过外部对比系统评分")
    parser.add_argument("--no-baseline", action="store_true", help="跳过 BASELINE 评分")
    parser.add_argument("--no-ablation", action="store_true", help="跳过消融实验评分")
    parser.add_argument(
        "--eval-dir", default=None, metavar="DIR",
        help=f"输出目录（默认 {DEFAULT_EVAL_DIR}）",
    )
    args = parser.parse_args()

    # ── 平台选择 ──────────────────────────────────────────────
    platform_choice = args.platform or select_platform_interactive()

    if platform_choice == "both":
        platforms = ["douyin", "youtube"]
    else:
        platforms = [platform_choice]

    eval_dir     = args.eval_dir or DEFAULT_EVAL_DIR
    run_external = not args.no_external
    run_baseline = not args.no_baseline
    run_ablation = not args.no_ablation

    os.makedirs(eval_dir, exist_ok=True)

    # ── 打印配置摘要 ──────────────────────────────────────────
    plat_label = " + ".join(p.upper() for p in platforms)
    print(f"\n{'═'*70}")
    print(f"  平台          ：{plat_label}")
    print(f"  外部系统评分  ：{'✅ 开启' if run_external else '⏭ 跳过'}")
    print(f"  BASELINE 评分 ：{'✅ 开启' if run_baseline else '⏭ 跳过'}")
    print(f"  消融实验评分  ：{'✅ 开启' if run_ablation else '⏭ 跳过'}")
    print(f"  输出目录      ：{eval_dir}")
    print(f"{'═'*70}")

    # ── 预加载模型（统一提前，两平台共享同一实例）────────────
    get_sbert()
    get_sentiment()

    # ── 逐平台评估 ────────────────────────────────────────────
    all_rows = []
    for platform in platforms:
        rows = evaluate_platform(
            platform     = platform,
            run_external = run_external,
            run_baseline = run_baseline,
            run_ablation = run_ablation,
        )
        all_rows.extend(rows)

        # 每个平台评完立即打印该平台的报告，不等另一个平台完成
        if rows:
            print(f"\n{'─'*70}")
            print(f"  {platform.upper()} 评分完成，生成报告 ...")
            print_report(rows)   # 只传当前平台的行，报告只显示该平台

    if not all_rows:
        print("\n  [错误] 没有任何评论参与评分，请检查输入文件路径。")
        return

    # ── 保存全量结果（all_eval_detail.json 含所有平台）────────
    print(f"\n{'═'*70}")
    print(f"  保存结果中 ...")
    save_results(all_rows, eval_dir)

    print(f"\n{'═'*70}")
    total_plat = len(set(r["platform"] for r in all_rows))
    print(f"  ✅ 全部评估完成！")
    print(f"     平台数    ：{total_plat}（{plat_label}）")
    print(f"     总记录数  ：{len(all_rows)} 条")
    print(f"     输出目录  ：{eval_dir}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()