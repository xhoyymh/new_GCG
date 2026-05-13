# `code_active`

英文版：[README.md](./README.md)

这个目录是 `data_pre` 流水线面向日常使用的入口目录。

如果你完全不熟悉这个项目，建议从这里开始看。旧目录 `E:\pn\new_GCG-main\data_pre\code` 仍然保留在仓库里，但它更像历史参考，不是当前推荐的起点。

## 这个项目整体在做什么

这套流程会把短视频原始数据逐步变成结构化训练数据：

1. Step 1 收集候选视频 URL
2. Step 2 下载视频并保存简介
3. Step 3 抓取评论
4. Step 4 抽帧并转录音频
5. Step 5 用 Codex 生成视频描述
6. Step 6 生成最终样本 JSON

真正的数据输出不在这个目录里，而是在这些位置：

- `data_pre\json\...`：结构化 JSON 输出
- `data_pre\video\...`：下载下来的视频
- `data_pre\douyin_image\...` 和 `data_pre\youtube_image\...`：Step 4 的抽帧和转录结果
- `data_pre\logs\...`：运行日志

## 开始之前先准备什么

- 在 PowerShell 中切到仓库根目录：`E:\pn\new_GCG-main`
- 使用这个 Python 解释器：
  - `C:\Users\anaconda3\envs\gcg_douyin311\python.exe`
- 如果要跑 Step 5，还需要：
  - `E:\pn\new_GCG-main\tmp_codex.exe`
  - 可用的 Codex 配额
- 这个整理后的目录只保留 Step 4 的 local 版本，不包含旧的 shared / dual-host 方案。

推荐先这样准备命令行环境：

```powershell
$py = 'C:\Users\anaconda3\envs\gcg_douyin311\python.exe'
Set-Location 'E:\pn\new_GCG-main'
```

## 每个文件是干什么的

- `douyin/douyin_dataset_all_in_one_ollama.py`
  - Douyin 平台的主流程脚本。
  - 在当前整理后的用法里，主要拿它跑 Step 1-3。
  - 它内部仍然保留了旧的 Step 4-6 逻辑，但如果你走现在推荐的主线，Step 4、5、6 请切换到下面的专用脚本。

- `youtube/youtube_dataset_all_in_one_ollama.py`
  - YouTube 平台的主流程脚本。
  - 在当前整理后的用法里，主要拿它跑 Step 1-3。
  - 和 Douyin 一样，不建议把它内部的旧 Step 4-6 当成当前主线入口。

- `douyin/douyin_download_label_target.py`
  - 可选辅助脚本。
  - 当某一个 Douyin label 下载数量不够时，用它定向补齐，不用重跑整条 Douyin 流程。

- `douyin/douyin_comments_label_target.py`
  - 可选辅助脚本。
  - 当 Douyin 某些 label 的评论缺失或不完整时，用它补抓评论。

- `youtube/youtube_download_label_target.py`
  - 可选辅助脚本。
  - 当某一个 YouTube label 需要补下载时，用它定向补齐。

- `douyin/douyin_video_sample.json`
  - 数据文件，不是可执行代码。
  - 被 Douyin Step 1 用作样本和标签提示数据。

- `youtube/youtube_video_sample.json`
  - 数据文件，不是可执行代码。
  - 被 YouTube Step 1 用作样本和标签提示数据。

- `launch_step4_local_v2.ps1`
  - 最适合 Windows 直接使用的 Step 4 启动器。
  - 它会调用 Python 的 Step 4 脚本，并把日志写入 `data_pre\logs\step4_local_v2`。

- `step4_local_v2.py`
  - 真正执行 Step 4 local 的 Python 脚本。
  - 如果你不想用 PowerShell 启动器，而是想直接用 Python 控制参数，就运行它。

- `build_codex_step5_manifest.py`
  - 用 Step 4 已经准备好的材料，构建 Step 5 待处理清单。
  - 一般在 Step 4 跑完之后运行。

- `codex_step5_mcp_server.py`
  - Step 5 的本地辅助服务。
  - 负责准备单个视频的上下文、维护队列状态、保存最终描述。
  - 一般不需要手动启动，Step 5 的运行脚本会自动调用它。

- `run_codex_step5_via_codex_cli.py`
  - 适合先做小规模验证的 Step 5 运行器。
  - 推荐先用它跑 1 条或少量样本，确认环境和配额没问题。

- `run_codex_step5_batches.py`
  - Step 5 的主力批量运行器。
  - 当 manifest 已经准备好，并且你要正式批量跑描述生成时，使用它。

- `run_step6_from_codex_outputs.py`
  - 基于 Codex 生成的 Step 5 描述和 top comments，构建最终 Step 6 样本输出。
  - Step 5 产出 `*_video_description_codex.json` 之后再运行它。

- `run_single_step45_validation.py`
  - 单条记录的 Step 4/5 验证脚本。
  - 适合你有意识地做端到端小验证。
  - 重要：它不是纯 dry-run，会对那条记录写入真实输出。

## 推荐的新手跑法

如果你是第一次接手这个项目，建议严格按这个顺序跑。

### 1. 先跑某个平台的 Step 1-3

Douyin：

```powershell
& $py .\data_pre\code_active\douyin\douyin_dataset_all_in_one_ollama.py --steps 1,2,3
```

YouTube：

```powershell
& $py .\data_pre\code_active\youtube\youtube_dataset_all_in_one_ollama.py --steps 1,2,3
```

如果两个平台都要处理，就把这两个命令依次都跑一遍。

### 2. 跑 Step 4 local

Windows 下最简单的方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\data_pre\code_active\launch_step4_local_v2.ps1 -Platform both
```

如果只处理一个平台，把 `both` 改成 `douyin` 或 `youtube`。

直接用 Python 也可以：

```powershell
& $py .\data_pre\code_active\step4_local_v2.py --platform both
```

### 3. 构建 Step 5 manifest

```powershell
& $py .\data_pre\code_active\build_codex_step5_manifest.py
```

这一步会生成 Step 5 的待处理清单。

### 4. 先小规模验证 Step 5

```powershell
& $py .\data_pre\code_active\run_codex_step5_via_codex_cli.py --limit 1 --model gpt-5.4-mini --reasoning-effort medium
```

建议先这样跑 1 条，先确认环境、路径和 Codex 配额没有问题。

### 5. 正式批量跑 Step 5

```powershell
& $py .\data_pre\code_active\run_codex_step5_batches.py --concurrency 10 --model gpt-5.4-mini --reasoning-effort medium
```

这是当前 Step 5 的主入口。

### 6. 生成 Step 6 输出

```powershell
& $py .\data_pre\code_active\run_step6_from_codex_outputs.py --platform all
```

如果你只跑了一个平台，就把 `all` 改成 `douyin` 或 `youtube`。

## 常见辅助命令

定向补齐某个 Douyin label 的下载：

```powershell
& $py .\data_pre\code_active\douyin\douyin_download_label_target.py --label "Comedy Skits" --target 200
```

补抓指定 Douyin label 的评论：

```powershell
& $py .\data_pre\code_active\douyin\douyin_comments_label_target.py --labels "Comedy Skits" "Daily Life Jokes"
```

定向补齐某个 YouTube label 的下载：

```powershell
& $py .\data_pre\code_active\youtube\youtube_download_label_target.py --label "Comedy Skits"
```

有意识地验证单条 Step 4/5：

```powershell
& $py .\data_pre\code_active\run_single_step45_validation.py --platform douyin
```

## 新手最容易踩的坑

- 不要先从 `data_pre\code` 目录开始，除非你是在排查历史实现。
- 如果你的目标只是跑通当前主线，不要去用旧的 shared / dual-host Step 4 路线。
- 不要把 `run_single_step45_validation.py` 当成无副作用脚本，它会写真实输出。
- 一般不要手动直接启动 `codex_step5_mcp_server.py`，除非你在调 Step 5 内部机制。

## 这个目录没有收进来的旧文件

- `E:\pn\new_GCG-main\data_pre\code\douyin\douyin_all_in_one.py`
- `E:\pn\new_GCG-main\data_pre\code\run_codex_step5_batch.py`
- `E:\pn\new_GCG-main\data_pre\code\step4_local_gpu_runner.py`
- `E:\pn\new_GCG-main\data_pre\code\launch_step4_local_gpu.ps1`
- 仍然只保留在旧 `code` 目录中的历史 Step 4 shared 文件：
  - `step4_shared_parallel.py`
  - `step4_remote_worker.ps1`
  - `STEP4_SHARED_README.md`
