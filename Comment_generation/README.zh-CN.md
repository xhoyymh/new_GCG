# `code_active`

英文版： [README.md](./README.md)

这个目录是当前项目整理后的可运行主入口。如果你是第一次接手这个仓库，建议从这里开始。

下面两个旧目录还保留在仓库里，方便查历史、对比实现、排查问题，但它们已经不是推荐入口：

- `E:\pn\new_GCG-main\data_pre\code`
- `E:\pn\new_GCG-main\comment_generation\code`

## 这个项目整体在做什么

主线 `data_pre` 流程会把短视频原始数据逐步整理成结构化 JSON：

1. Step 1 收集候选视频 URL
2. Step 2 下载视频并保存简介
3. Step 3 抓取原始评论
4. Step 4 在本地做抽帧和语音转录
5. Step 5 用 Codex 生成最终视频描述
6. Step 6 生成最终样本 JSON

在 Step 5 之后还有一条可选分支：

- 评论生成
  - 新手路线：基于最新的 Step 5 Codex 描述生成评论
  - 高级评测路线：基于平台原评论样本，对多个模型做评论生成能力对比

主要输出目录不在 `code_active` 里面，而是在这些位置：

- `data_pre\json\...`：主线结构化 JSON 输出
- `data_pre\video\...`：下载好的视频
- `data_pre\douyin_image\...` 和 `data_pre\youtube_image\...`：Step 4 抽帧和转录结果
- `data_pre\logs\...`：运行日志
- `comment_generation\json\result\...`：评论生成结果
- `comment_generation\cache\...`：评论生成 embedding 缓存

## 开始之前先准备什么

先在仓库根目录打开 PowerShell：

```powershell
$py = 'C:\Users\anaconda3\envs\gcg_douyin311\python.exe'
Set-Location 'E:\pn\new_GCG-main'
```

另外还需要：

- Step 5 需要 `E:\pn\new_GCG-main\tmp_codex.exe`
- Step 5 批量跑时需要可用的 Codex 配额
- 本地评论生成路线需要本机 Ollama 可用

如果你要跑高级 API 评论生成路线，需要二选一完成 OpenRouter 配置：

1. 在脚本目录旁边放本地配置文件

```powershell
Copy-Item `
  .\data_pre\code_active\comment_generation\openrouter_api_pool.example.json `
  .\data_pre\code_active\comment_generation\openrouter_api_pool.json
```

然后手动编辑 `openrouter_api_pool.json`，把里面的占位 key 换成你自己的。

2. 用环境变量

```powershell
$env:OPENROUTER_API_KEY = 'your-key'
$env:OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
```

如果你有多个 key，也可以用 `OPENROUTER_API_KEYS_JSON`。

## 文件说明

这一节按“你什么时候会用到它”来解释每个重要文件的作用。

### 主线数据处理文件

- `douyin/douyin_dataset_all_in_one_ollama.py`
  - Douyin 主流程脚本。
  - 在当前整理后的用法里，主要拿它跑 Step 1-3。

- `youtube/youtube_dataset_all_in_one_ollama.py`
  - YouTube 主流程脚本。
  - 在当前整理后的用法里，主要拿它跑 Step 1-3。

- `douyin/douyin_download_label_target.py`
  - Douyin 补量工具。
  - 当某个标签下载数量不够时，用它定向补，不用重跑整条流程。

- `douyin/douyin_comments_label_target.py`
  - Douyin 评论补抓工具。
  - 当某些标签的评论缺失时，用它补抓。

- `youtube/youtube_download_label_target.py`
  - YouTube 补量工具。
  - 当某个标签还需要补下载时，用它定向补。

- `douyin/douyin_video_sample.json`
  - 数据文件，不是代码。
  - 给 Douyin Step 1 提供种子样本。

- `youtube/youtube_video_sample.json`
  - 数据文件，不是代码。
  - 给 YouTube Step 1 提供种子样本。

- `launch_step4_local_v2.ps1`
  - Windows 下最容易直接运行的 Step 4 local 入口。

- `step4_local_v2.py`
  - Step 4 local 的 Python 入口。
  - 如果你想直接用 Python 控制参数，就跑它。

- `build_codex_step5_manifest.py`
  - 根据 Step 4 已经准备好的材料，构建 Step 5 待处理清单。

- `codex_step5_mcp_server.py`
  - Step 5 内部使用的辅助服务。
  - 一般不需要手动启动。

- `run_codex_step5_via_codex_cli.py`
  - Step 5 小规模试跑入口。
  - 推荐第一次先用它跑 1 条或少量样本，先确认环境没问题。

- `run_codex_step5_batches.py`
  - Step 5 正式批量入口。

- `run_step6_from_codex_outputs.py`
  - 根据 Step 5 的描述结果和评论，生成最终 Step 6 样本。

- `run_single_step45_validation.py`
  - 单条记录的 Step 4/5 验证工具。
  - 它会写真实输出，不是完全无副作用的 dry-run。

### 评论生成：新手路线

- `comment_generation/run_phase2_from_descriptions.py`
  - 最推荐的新手入口。
  - 直接读取最新的 Step 5 Codex 描述文件来生成评论。

- `comment_generation/run_phase2_from_test.py`
  - 测试入口。
  - 读取 `test\*.json`，做归一化后跑本地评论生成，不会直接依赖最新 Step 5 产物。

- `comment_generation/test_input_loader.py`
  - 测试输入清洗和归一化工具。
  - 给 `run_phase2_from_test.py` 和高级评测脚本共用。

- `comment_generation/douyin/comment_generate.py`
  - Douyin 本地评论生成后端。
  - 真正的生成逻辑在这里，但新手通常不建议直接从这里起跑。

- `comment_generation/youtube/youtube_comment_generate_hotmeme.py`
  - YouTube 本地评论生成后端。
  - YouTube 路线可选支持 meme search。

### 评论生成：高级评测路线

- `comment_generation/run_phase2_from_test_api_ready.py`
  - API-ready 测试入口。
  - 基于归一化后的 `test\*.json` 输入，可以选择用本地 Ollama 或 OpenRouter API 模型生成评论。
  - 如果你想先验证某一个对比模型，这通常是最好的起点。

- `comment_generation/run_phase2_from_original_comments_compare.py`
  - 多模型对比入口。
  - 基于平台原评论样本做 benchmark，对多个模型的评论生成效果做横向比较。
  - 这是当前模型对比的主入口。

- `comment_generation/run_phase2_from_original_comments_api_batch.py`
  - 固定付费模型的批量评测入口。
  - 用预先定义好的付费 OpenRouter 模型跑同一套 benchmark。

- `comment_generation/model_api_adapter.py`
  - `ollama` / `api` 的统一后端适配层。
  - 负责 OpenRouter key 读取、模型别名映射、重试和 key 轮换。

- `comment_generation/original_comment_loader.py`
  - 加载平台原评论 benchmark 文件，并和归一化后的测试记录合并。

- `comment_generation/openrouter_api_pool.example.json`
  - OpenRouter 本地配置模板。
  - 仓库只提供模板，不包含真实 key。

- `comment_generation/douyin/comment_generate_api_ready.py`
  - Douyin 的 API-ready 后端实现。
  - 它复用了当前 active 版 Douyin 本地生成器，并补上原评论 prompt 和 API 后端支持。

- `comment_generation/youtube/youtube_comment_generate_api_ready.py`
  - YouTube 的 API-ready 后端实现。
  - 它复用了当前 active 版 YouTube 本地生成器，并补上原评论 prompt 和 API 后端支持。

### 高级评测路线会用到、但没有复制进 `code_active` 的运行数据

这两份文件是运行时输入，不是代码，所以不会被复制进 `code_active`：

- `E:\pn\new_GCG-main\comment_generation\original_comments_for_douyin.json`
- `E:\pn\new_GCG-main\comment_generation\original_comments_for_youtube.json`

它们是原评论对比路线使用的 benchmark 数据。

## 如果你是新手，建议这样跑主线

如果你的目标是把主线 Step 1-6 跑通，推荐按下面顺序来。

### 1. 先跑 Step 1-3

Douyin：

```powershell
& $py .\data_pre\code_active\douyin\douyin_dataset_all_in_one_ollama.py --steps 1,2,3
```

YouTube：

```powershell
& $py .\data_pre\code_active\youtube\youtube_dataset_all_in_one_ollama.py --steps 1,2,3
```

### 2. 跑 Step 4 local

```powershell
powershell -ExecutionPolicy Bypass -File .\data_pre\code_active\launch_step4_local_v2.ps1 -Platform both
```

如果只想处理一个平台，把 `both` 改成 `douyin` 或 `youtube`。

### 3. 构建 Step 5 manifest

```powershell
& $py .\data_pre\code_active\build_codex_step5_manifest.py
```

### 4. 先用 1 条记录试跑 Step 5

```powershell
& $py .\data_pre\code_active\run_codex_step5_via_codex_cli.py --limit 1 --model gpt-5.4-mini --reasoning-effort medium
```

### 5. 正式批量跑 Step 5

```powershell
& $py .\data_pre\code_active\run_codex_step5_batches.py --concurrency 10 --model gpt-5.4-mini --reasoning-effort medium
```

### 6. 可选：基于 Step 5 结果生成评论

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_descriptions.py --platform all --limit 5
```

这一步是可选分支，不会替代 Step 6。

### 7. 生成 Step 6 输出

```powershell
& $py .\data_pre\code_active\run_step6_from_codex_outputs.py --platform all
```

## 如果你的目标是评论生成，该怎么选入口

这一节只看评论生成，不看主线 Step 1-6。

### A. 基于最新 Step 5 结果生成评论

当你已经有这些文件时，用这一条：

- `data_pre\json\douyin\data_pre\douyin_video_description_codex.json`
- `data_pre\json\youtube\data_pre\youtube_video_description_codex.json`

推荐命令：

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_descriptions.py --platform all --limit 5
```

输出位置：

- `comment_generation\json\result\douyin_output_comments.json`
- `comment_generation\json\result\youtube_output_comments.json`

### B. 基于 `test\*.json` 调试本地评论生成流程

当你想调试评论生成，但不想依赖最新 Step 5 结果时，用这一条：

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_test.py --platform all --limit 5
```

它还会把归一化后的测试输入写到 `comment_generation\json\douyin\...` 和 `comment_generation\json\youtube\...`。

### C. 先验证单个 API 模型

当你想先验证一个 API 模型是否能正常跑通，推荐先用这个入口：

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_test_api_ready.py `
  --platform douyin `
  --generation-backend api `
  --model-alias glm `
  --limit 3
```

如果你想复用同一条 API-ready 路线，但先用本地模型，也可以把 `--generation-backend` 改成 `ollama`。

适配层里常见可用的模型别名包括：

- `glm`
- `qwen`
- `dsr1`
- `llama`
- `gptoss`
- `r1`
- `gpt54`

输出位置：

- `comment_generation\json\result\douyin_output_comments_api_ready.json`
- `comment_generation\json\result\youtube_output_comments_api_ready.json`

### D. 用原评论 benchmark 做多模型对比

当你想做横向评测，而且希望避免把“同一条视频的原评论”直接泄漏进 prompt 时，用这个入口：

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_original_comments_compare.py `
  --platform all `
  --models qwen3.5_9b qwen_free glm_air `
  --limit 5 `
  --resume
```

这一条会输出：

- `comment_generation\json\result\douyin_original_comments_model_compare.json`
- `comment_generation\json\result\douyin_original_comments_model_compare_meta.json`
- `comment_generation\json\result\youtube_original_comments_model_compare.json`
- `comment_generation\json\result\youtube_original_comments_model_compare_meta.json`

### E. 跑固定付费模型批量评测

如果你想直接跑预先定义好的付费 OpenRouter 模型（`r1`、`llama`、`gpt54`），用这个入口：

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_original_comments_api_batch.py `
  --platform all `
  --limit 5 `
  --resume
```

这一条会输出：

- `comment_generation\json\result\douyin_original_comments_paid_model_outputs.json`
- `comment_generation\json\result\youtube_original_comments_paid_model_outputs.json`

## 新手最容易踩的坑

- 不要一上来就从 `data_pre\code` 开始，除非你是在查历史实现。
- 不要一上来就从 `comment_generation\code` 开始，除非你是在对比旧源码。
- 如果你的目标只是跑通当前主线，不要去碰旧的 shared / dual-host Step 4 路线。
- 如果你是第一次接手，不要优先直接跑评论生成后端脚本。
- 不要把真实 API key 写进仓库里的模板文件版本。

## 这些旧文件没有被收进 `code_active`

- `E:\pn\new_GCG-main\data_pre\code\douyin\douyin_all_in_one.py`
- `E:\pn\new_GCG-main\data_pre\code\run_codex_step5_batch.py`
- `E:\pn\new_GCG-main\data_pre\code\step4_local_gpu_runner.py`
- `E:\pn\new_GCG-main\data_pre\code\launch_step4_local_gpu.ps1`
- `E:\pn\new_GCG-main\comment_generation\code\douyin\douyin_comment_generate_hotmeme.py`
- 缓存文件、`.gitkeep` 和 `__pycache__` 目录不会复制进 `code_active`
- 旧的 Step 4 shared 文件仍然只保留在历史 `data_pre\code` 目录里
