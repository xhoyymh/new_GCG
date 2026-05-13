# `code_active`

Chinese version: [README.zh-CN.md](./README.zh-CN.md)

This folder is the beginner-friendly runnable entrypoint for the `data_pre` pipeline.

If you are new to this project, start here. The older folder `E:\pn\new_GCG-main\data_pre\code` is historical reference. It is kept on disk, but it is not the recommended place to begin.

## What This Project Does

At a high level, this pipeline turns short-video data into structured training data:

1. Step 1 collects candidate video URLs.
2. Step 2 downloads videos and saves basic intro text.
3. Step 3 fetches comments.
4. Step 4 extracts frames and speech transcription.
5. Step 5 uses Codex to write a polished video description.
6. Step 6 builds final sample JSON outputs.

The main outputs are written outside this folder:

- `data_pre\json\...` for structured JSON files
- `data_pre\video\...` for downloaded videos
- `data_pre\douyin_image\...` and `data_pre\youtube_image\...` for Step 4 frames and transcripts
- `data_pre\logs\...` for runtime logs

## Before You Run Anything

- Open PowerShell in the repo root: `E:\pn\new_GCG-main`
- Use this Python interpreter:
  - `C:\Users\anaconda3\envs\gcg_douyin311\python.exe`
- For Step 5, you also need:
  - `E:\pn\new_GCG-main\tmp_codex.exe`
  - available Codex quota
- In this curated folder, Step 4 is local-only. The old shared / dual-host Step 4 route is intentionally not included here.

Recommended shell setup:

```powershell
$py = 'C:\Users\anaconda3\envs\gcg_douyin311\python.exe'
Set-Location 'E:\pn\new_GCG-main'
```

## What Each File Is For

- `douyin/douyin_dataset_all_in_one_ollama.py`
  - Main Douyin pipeline script.
  - In this curated setup, use it mainly for Step 1-3.
  - It still contains older Step 4-6 logic internally, but the recommended path in `code_active` is to switch to the dedicated Step 4 / Step 5 / Step 6 scripts below.

- `youtube/youtube_dataset_all_in_one_ollama.py`
  - Main YouTube pipeline script.
  - In this curated setup, use it mainly for Step 1-3.
  - Same note as above: do not treat its built-in Step 4-6 path as the recommended active route.

- `douyin/douyin_download_label_target.py`
  - Optional helper.
  - Use this when one Douyin label is under-downloaded and you want to top it up without rerunning the whole Douyin pipeline.

- `douyin/douyin_comments_label_target.py`
  - Optional helper.
  - Use this when Douyin comments are missing or incomplete for specific labels.

- `youtube/youtube_download_label_target.py`
  - Optional helper.
  - Use this when one YouTube label needs more downloaded videos without rerunning the whole YouTube flow.

- `douyin/douyin_video_sample.json`
  - Data file, not code.
  - Seed examples used by the Douyin Step 1 collection logic.

- `youtube/youtube_video_sample.json`
  - Data file, not code.
  - Seed examples used by the YouTube Step 1 collection logic.

- `launch_step4_local_v2.ps1`
  - The easiest Windows entrypoint for Step 4.
  - It wraps the Python Step 4 runner and writes logs under `data_pre\logs\step4_local_v2`.

- `step4_local_v2.py`
  - The actual Step 4 local runner.
  - Use this if you want direct Python control instead of the PowerShell launcher.

- `build_codex_step5_manifest.py`
  - Builds the Step 5 backlog manifest from videos that already have Step 4 materials.
  - Run this after Step 4 has finished for the videos you want to describe.

- `codex_step5_mcp_server.py`
  - Local helper service for Step 5.
  - It prepares one video's context, manages queue state, and saves final descriptions.
  - Usually you do not start this manually; the Step 5 runners call it for you.

- `run_codex_step5_via_codex_cli.py`
  - Good first Step 5 runner to test on one or a few videos.
  - Use this before running the large batch job.

- `run_codex_step5_batches.py`
  - Main Step 5 production runner.
  - Use this when the manifest is ready and you want to process many videos with Codex.

- `run_step6_from_codex_outputs.py`
  - Builds the final Step 6 sample outputs from Codex Step 5 descriptions plus top comments.
  - Run this after Step 5 has produced `*_video_description_codex.json`.

- `run_single_step45_validation.py`
  - Validation helper for one record.
  - Use this only when you want to intentionally test one Step 4/5 path end to end.
  - Important: this is not a pure dry-run; it writes real outputs for that record.

## Recommended Way To Run The Whole Pipeline

If you want the safest path for a new person, use this order.

### 1. Run Step 1-3 for one platform

Douyin:

```powershell
& $py .\data_pre\code_active\douyin\douyin_dataset_all_in_one_ollama.py --steps 1,2,3
```

YouTube:

```powershell
& $py .\data_pre\code_active\youtube\youtube_dataset_all_in_one_ollama.py --steps 1,2,3
```

If you need both platforms, run both commands one after the other.

### 2. Run Step 4 locally

The easiest option on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\data_pre\code_active\launch_step4_local_v2.ps1 -Platform both
```

If you only want one platform, replace `both` with `douyin` or `youtube`.

Direct Python alternative:

```powershell
& $py .\data_pre\code_active\step4_local_v2.py --platform both
```

### 3. Build the Step 5 manifest

```powershell
& $py .\data_pre\code_active\build_codex_step5_manifest.py
```

This creates the backlog file that Step 5 will read.

### 4. Validate Step 5 on one item first

```powershell
& $py .\data_pre\code_active\run_codex_step5_via_codex_cli.py --limit 1 --model gpt-5.4-mini --reasoning-effort medium
```

Do this first so you can catch environment or quota problems before a larger run.

### 5. Run Step 5 in batch

```powershell
& $py .\data_pre\code_active\run_codex_step5_batches.py --concurrency 10 --model gpt-5.4-mini --reasoning-effort medium
```

This is the normal large-run Step 5 entrypoint.

### 6. Build Step 6 outputs

```powershell
& $py .\data_pre\code_active\run_step6_from_codex_outputs.py --platform all
```

If you only processed one platform, replace `all` with `douyin` or `youtube`.

## Common Helper Commands

Top up one Douyin label:

```powershell
& $py .\data_pre\code_active\douyin\douyin_download_label_target.py --label "Comedy Skits" --target 200
```

Refresh Douyin comments for selected labels:

```powershell
& $py .\data_pre\code_active\douyin\douyin_comments_label_target.py --labels "Comedy Skits" "Daily Life Jokes"
```

Top up one YouTube label:

```powershell
& $py .\data_pre\code_active\youtube\youtube_download_label_target.py --label "Comedy Skits"
```

Validate one Step 4/5 record intentionally:

```powershell
& $py .\data_pre\code_active\run_single_step45_validation.py --platform douyin
```

## What Not To Use First

- Do not start from `data_pre\code` unless you are debugging history.
- Do not use the older shared / dual-host Step 4 route if your goal is simply to run the active pipeline.
- Do not treat `run_single_step45_validation.py` as harmless; it writes real outputs.
- Do not start `codex_step5_mcp_server.py` manually unless you are debugging the Step 5 internals.

## Legacy Files Not Included Here

- `E:\pn\new_GCG-main\data_pre\code\douyin\douyin_all_in_one.py`
- `E:\pn\new_GCG-main\data_pre\code\run_codex_step5_batch.py`
- `E:\pn\new_GCG-main\data_pre\code\step4_local_gpu_runner.py`
- `E:\pn\new_GCG-main\data_pre\code\launch_step4_local_gpu.ps1`
- historical shared Step 4 files kept only in the old `code` directory:
  - `step4_shared_parallel.py`
  - `step4_remote_worker.ps1`
  - `STEP4_SHARED_README.md`
