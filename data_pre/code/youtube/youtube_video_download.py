import yt_dlp
import json
import os
from tqdm import tqdm

# 输入输出文件路径
input_json_path = r'D:\Desktop\video_comment_generation\json\new\youtube\youtube_video_test.json'
output_json_path = r'D:\Desktop\video_comment_generation\json\new\youtube\youtube_test_introduction.json'
video_save_dir = r'D:\Desktop\video_comment_generation\video\youtube_test'

# 创建保存目录（如果不存在）
os.makedirs(video_save_dir, exist_ok=True)

# 读取原始 JSON
with open(input_json_path, 'r', encoding='utf-8') as f:
    video_data = json.load(f)

# 新的 JSON 数据列表
updated_video_data = []

# 遍历每个视频项
with tqdm(total=len(video_data), desc="下载中", unit="video") as pbar:
    for entry in video_data:
        video_id = entry.get("id")
        url = entry.get("video_url")
        label = entry.get("label", "")

        # 构建输出路径
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)

        # 下载设置
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': video_path,
            'quiet': True,
            'noplaylist': True,
            'skip_download': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'postprocessors': [],
        }

        # 初始化简介
        video_description = ""

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_description = info.get("description", "")
        except Exception as e:
            print(f"[错误] 视频 {video_id} 下载失败: {e}")
            video_description = "下载失败或无法获取简介"

        # 构建新的条目
        updated_entry = {
            "id": video_id,
            "video_url": url,
            "label": label,
            "video_introduction": video_description.strip()
        }
        updated_video_data.append(updated_entry)
        pbar.update(1)

# 保存更新后的 JSON
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(updated_video_data, f, ensure_ascii=False, indent=4)

print(f"\n所有视频下载完成，更新后的 JSON 文件已保存至：{output_json_path}")
