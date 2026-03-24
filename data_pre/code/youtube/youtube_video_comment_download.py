import os
import json
import time
import yt_dlp
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build

# 路径配置
INPUT_JSON_PATH = r'D:\Desktop\OYX\video_comment_generation_en\json\new\video_new_youtube.json'
OUTPUT_MAIN_JSON_PATH = r'D:\Desktop\OYX\video_comment_generation_en\json\new\youtube_new_top5comments.json'
OUTPUT_COMMENTS_JSON_PATH = r'D:\Desktop\OYX\video_comment_generation_en\json\new\youtube_new_top5comments_detail.json'
VIDEO_SAVE_FOLDER = r'D:\Desktop\OYX\video_comment_generation_en\video\new'

# API 设置
API_KEY = 'AIzaSyAp0cKrDn6M3--UQaSHlfJF1UcGfanWsug'
youtube = build('youtube', 'v3', developerKey=API_KEY)

# 🧠 提取 videoId，兼容 Shorts 和标准链接
def extract_video_id(video_url):
    if 'shorts' in video_url:
        return video_url.split('/')[-1]
    elif 'watch?v=' in video_url:
        return video_url.split('v=')[-1].split('&')[0]
    else:
        parsed = urlparse(video_url)
        return parse_qs(parsed.query).get('v', [''])[0]

# 🎬 下载视频
def download_video(url, save_id):
    os.makedirs(VIDEO_SAVE_FOLDER, exist_ok=True)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(VIDEO_SAVE_FOLDER, f'{save_id}.mp4'),
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# ℹ️ 获取视频简介和标题
def get_video_info(video_id):
    try:
        response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()
        if response['items']:
            snippet = response['items'][0]['snippet']
            return snippet.get('description', ''), snippet.get('title', '')
    except Exception as e:
        print(f"❌ 获取视频信息失败：{video_id}", e)
    return "", ""

# 💬 获取点赞最多的前5条评论（从最多 1000 条中筛选）
def get_top_comments(video_id, max_count=5, max_total=50000):
    comments = []
    next_page_token = None
    try_count = 0

    while True:
        try:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat='plainText'
            ).execute()

            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': snippet.get('textDisplay', ''),
                    'likeCount': snippet.get('likeCount', 0)
                })

            next_page_token = response.get('nextPageToken')
            if not next_page_token or len(comments) >= max_total:
                break
        except Exception as e:
            try_count += 1
            print(f"⚠️ 评论抓取失败尝试 {try_count} 次：{video_id}", e)
            if try_count >= 3:
                break
            time.sleep(1)

    sorted_comments = sorted(comments, key=lambda x: x['likeCount'], reverse=True)
    return sorted_comments[:max_count], sorted_comments

# 🏷️ 评论标签函数（可接入你的模型）
def label_comment(comment):
    return "待标注" if comment else ""

# 📦 保存评论详情 JSON（带点赞数）
def save_comments_by_video(video_comments_dict, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(video_comments_dict, f, indent=4, ensure_ascii=False)

# 🚀 主流程：处理所有视频并生成两个 JSON 输出
def process_all_videos(input_path, output_main_path, output_comment_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        video_list = json.load(f)

    all_results = []
    all_comments_by_video = {}

    for idx, video_data in enumerate(video_list, start=1):
        video_url = video_data['video_url']
        video_id = extract_video_id(video_url)
        save_id = video_data['id']

        print(f"\n🔄 处理视频 {idx}：{video_id}")
        download_video(video_url, save_id)
        description, title = get_video_info(video_id)
        top_5, full_comments = get_top_comments(video_id)

        # 主输出结构
        result = {
            "id": save_id,
            "video_url": video_url,
            "video_introduction": title,
            "video_description": description,
            "label": video_data.get("label", "")
        }

        for i, comment in enumerate(top_5):
            result[f"comment_{i+1}"] = comment.get('text', '')
            result[f"C{i+1}_label"] = label_comment(comment.get('text', ''))

        all_results.append(result)

        # 评论详情输出结构
        all_comments_by_video[str(idx)] = [
            {
                "text": c.get("text", ""),
                "digg_count": c.get("likeCount", 0)
            } for c in top_5
        ]

    # 保存主 JSON
    with open(output_main_path, 'w', encoding='utf-8') as f_out:
        json.dump(all_results, f_out, indent=4, ensure_ascii=False)

    # 保存评论详情 JSON
    save_comments_by_video(all_comments_by_video, output_comment_path)
    print(f"\n✅ 所有视频处理完毕\n📄 主数据保存至：{output_main_path}\n💬 评论详情保存至：{output_comment_path}")

# 🟩 启动脚本
if __name__ == '__main__':
    process_all_videos(INPUT_JSON_PATH, OUTPUT_MAIN_JSON_PATH, OUTPUT_COMMENTS_JSON_PATH)
