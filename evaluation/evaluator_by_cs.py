import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from jieba import cut
from collections import Counter
import openai

# 设置 OpenAI API 密钥
openai.api_key = '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'

# 读取文件并解析为json
def load_json_files(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    return data

# 使用 OpenAI API 进行情绪分析
def compute_sentiment(comment):
    try:
        # 调用 OpenAI API，使用 GPT 模型来做情感分析
        response = openai.Completion.create(
            model="gpt-4",  # 可以根据需要选择不同的模型
            prompt=f"请分析以下评论的情感并判断为程度如何：\n\n评论：{comment}\n\n情感：",
            max_tokens=10,
            temperature=0.3
        )
        sentiment = response.choices[0].text.strip()
        
        # 根据返回的情感进行评分
        if "负面" in sentiment:
            return -1
        elif "正面" in sentiment:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0  # 如果出错，默认返回 0

# 计算评论字数得分
def comment_length_score(comments):
    score = 0
    for comment in comments:
        length = len(comment)
        if 20 <= length <= 50:
            score += 1
    return min(score, 20)

# 计算评论与视频描述的主题相关性
def compute_topic_relevance(comment, description):
    # 使用词频的余弦相似度来计算
    comment_words = Counter(cut(comment))
    description_words = Counter(cut(description))
    all_words = set(comment_words) | set(description_words)
    
    comment_vector = np.array([comment_words.get(word, 0) for word in all_words])
    description_vector = np.array([description_words.get(word, 0) for word in all_words])
    
    similarity = cosine_similarity([comment_vector], [description_vector])
    return similarity[0][0] * 10  # 满分10

# 计算评论和sample的相似度（自然度）
def compute_naturalness(comment, sample_comments):
    best_similarity = 0
    for sample_comment in sample_comments:
        if sample_comment:  # 跳过空评论
            similarity = compute_similarity(comment, sample_comment)
            sentiment_diff = abs(compute_sentiment(comment) - compute_sentiment(sample_comment))
            adjusted_similarity = similarity * (1 - sentiment_diff / 10)  # 调整情绪差异
            best_similarity = max(best_similarity, adjusted_similarity)
    
    return best_similarity * 20  # 满分20

def compute_similarity(cmt1, cmt2):
    cmt1_words = Counter(cut(cmt1))
    cmt2_words = Counter(cut(cmt2))
    all_words = set(cmt1_words) | set(cmt2_words)
    
    cmt1_vector = np.array([cmt1_words.get(word, 0) for word in all_words])
    cmt2_vector = np.array([cmt2_words.get(word, 0) for word in all_words])
    
    return cosine_similarity([cmt1_vector], [cmt2_vector])[0][0]

# 计算每个文件的得分
def compute_scores(json_files, sample_comments):
    results = []
    for file_path in json_files:
        # 读取json数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        comments = [item['comment'] for item in data]
        descriptions = [item['video_description'] for item in data]
        
        # 计算评论字数得分
        length_score = comment_length_score(comments)
        
        # 计算主题相关性
        relevance_scores = [compute_topic_relevance(c, d) for c, d in zip(comments, descriptions)]
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
        
        # 计算自然度
        naturalness_scores = [compute_naturalness(c, sample_comments) for c in comments]
        avg_naturalness = np.mean(naturalness_scores) if naturalness_scores else 0
        
        # 计算总分
        total_score = length_score + avg_relevance + avg_naturalness
        
        results.append({
            'file': file_path,
            'length_score': length_score,
            'avg_relevance': avg_relevance,
            'avg_naturalness': avg_naturalness,
            'total_score': total_score
        })
    
    return results

# 主函数
def main():
    # 输入文件路径
    json_files = [
        r'D:\Desktop\video_comment_generation\json\result\douyin\douyin_chinese_comments_V2Xum-LLM.json',
        r'D:\Desktop\video_comment_generation\json\result\douyin\douyin_comments_livechat.json',
        r'D:\Desktop\video_comment_generation\json\result\douyin\douyin_video_comments.json',
        r'D:\Desktop\video_comment_generation\json\result\douyin\model_douyin_commentgeneration_directly.json'
    ]
    
    sample_file_path = r'D:\Desktop\video_comment_generation\json\sample\douyin\chouzhenyong.json'
    
    # 读取sample数据
    with open(sample_file_path, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    # 假设 sample_data 是一个列表，每个元素包含一个包含 comment_1, comment_2 等评论的字典
    sample_comments = sample_data[0]  # 获取列表中的第一个字典
    comments = [sample_comments.get(f"comment_{i}") for i in range(1, 6)]
    
    # 计算得分
    scores = compute_scores(json_files, comments)
    
    # 确保输出路径存在
    output_dir = r'D:\Desktop\video_comment_generation\evaluation\result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 输出为CSV
    output_file = os.path.join(output_dir, 'scores.csv')
    df = pd.DataFrame(scores)
    df.to_csv(output_file, index=False)

    print(f"Scores have been saved to '{output_file}'")

if __name__ == '__main__':
    main()
