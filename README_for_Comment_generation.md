# `code_active`

Chinese version: [README.zh-CN.md](./README.zh-CN.md)

This folder is the curated runnable entrypoint for the current project. If you are new to this repository, start here.

The older folders below are still kept for history, comparison, and debugging, but they are no longer the recommended place to begin:

- `E:\pn\new_GCG-main\data_pre\code`
- `E:\pn\new_GCG-main\comment_generation\code`

## What This Project Does

The main `data_pre` pipeline turns raw short-video data into structured JSON samples:

1. Step 1 collects candidate video URLs.
2. Step 2 downloads videos and saves introductions.
3. Step 3 fetches source comments.
4. Step 4 extracts frames and speech locally.
5. Step 5 uses Codex to write final video descriptions.
6. Step 6 builds the final sample JSON outputs.

There is also an optional branch after Step 5:

- Comment generation
  - Beginner route: generate comments from the latest Step 5 Codex descriptions.
  - Advanced evaluation route: compare multiple models on benchmark inputs based on original platform comments.

Important output folders live outside `code_active`:

- `data_pre\json\...`: main structured JSON outputs
- `data_pre\video\...`: downloaded videos
- `data_pre\douyin_image\...` and `data_pre\youtube_image\...`: Step 4 frame/transcript outputs
- `data_pre\logs\...`: runtime logs
- `comment_generation\json\result\...`: generated-comment outputs
- `comment_generation\cache\...`: embedding caches for comment generation

## Before You Run Anything

Open PowerShell in the repo root:

```powershell
$py = 'C:\Users\anaconda3\envs\gcg_douyin311\python.exe'
Set-Location 'E:\pn\new_GCG-main'
```

You will also need:

- `E:\pn\new_GCG-main\tmp_codex.exe` for Step 5
- available Codex quota for Step 5 batch runs
- local Ollama running for the local comment-generation paths

For the advanced API-based comment routes, choose one of these setup options:

1. Local JSON config next to the scripts

```powershell
Copy-Item `
  .\data_pre\code_active\comment_generation\openrouter_api_pool.example.json `
  .\data_pre\code_active\comment_generation\openrouter_api_pool.json
```

Then edit `openrouter_api_pool.json` and replace the placeholder key.

2. Environment variables

```powershell
$env:OPENROUTER_API_KEY = 'your-key'
$env:OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
```

You can also use `OPENROUTER_API_KEYS_JSON` if you want a rotating key pool.

## File Guide

This section explains what each important script is for and when to use it.

### Main pipeline files

- `douyin/douyin_dataset_all_in_one_ollama.py`
  - Main Douyin collector/downloader script.
  - In the active setup, use it mainly for Step 1-3.

- `youtube/youtube_dataset_all_in_one_ollama.py`
  - Main YouTube collector/downloader script.
  - In the active setup, use it mainly for Step 1-3.

- `douyin/douyin_download_label_target.py`
  - Douyin helper for topping up one label without rerunning the whole pipeline.

- `douyin/douyin_comments_label_target.py`
  - Douyin helper for refreshing comments for specific labels.

- `youtube/youtube_download_label_target.py`
  - YouTube helper for topping up one label without rerunning the full flow.

- `douyin/douyin_video_sample.json`
  - Seed data used by the Douyin Step 1 collection logic.

- `youtube/youtube_video_sample.json`
  - Seed data used by the YouTube Step 1 collection logic.

- `launch_step4_local_v2.ps1`
  - The easiest Windows entrypoint for Step 4 local processing.

- `step4_local_v2.py`
  - The direct Python runner behind the Step 4 local launcher.

- `build_codex_step5_manifest.py`
  - Builds the Step 5 backlog manifest from records that already have Step 4 materials.

- `codex_step5_mcp_server.py`
  - Internal Step 5 helper service used by the Step 5 runners.
  - Most users do not start this directly.

- `run_codex_step5_via_codex_cli.py`
  - The safest Step 5 starter command for a small test run.

- `run_codex_step5_batches.py`
  - The main Step 5 batch runner for larger Codex description jobs.

- `run_step6_from_codex_outputs.py`
  - Builds the final Step 6 sample outputs from Step 5 Codex descriptions plus comments.

- `run_single_step45_validation.py`
  - Validation helper for one record.
  - This writes real outputs and is not a harmless dry-run.

### Comment generation: beginner route

- `comment_generation/run_phase2_from_descriptions.py`
  - Recommended entrypoint for beginners.
  - Reads the latest Step 5 Codex description JSON and generates comments.

- `comment_generation/run_phase2_from_test.py`
  - Test-only entrypoint.
  - Reads `test\*.json`, normalizes the records, and runs the local comment generator without touching the latest Step 5 outputs.

- `comment_generation/test_input_loader.py`
  - Shared helper that cleans, repairs, and normalizes `test\*.json` inputs.

- `comment_generation/douyin/comment_generate.py`
  - Current Douyin local comment-generation backend.
  - Most people should not start here directly.

- `comment_generation/youtube/youtube_comment_generate_hotmeme.py`
  - Current YouTube local comment-generation backend.
  - Supports meme search on the YouTube path when enabled.

### Comment generation: advanced evaluation route

- `comment_generation/run_phase2_from_test_api_ready.py`
  - API-ready test runner.
  - Uses normalized `test\*.json` inputs and can generate comments with either local Ollama or an OpenRouter API model.
  - Best first step if you want to verify one compare model.

- `comment_generation/run_phase2_from_original_comments_compare.py`
  - Multi-model benchmark runner.
  - Uses original-comment benchmark inputs and writes one file with outputs from several models side by side.
  - This is the main script for model comparison.

- `comment_generation/run_phase2_from_original_comments_api_batch.py`
  - Fixed-model paid API batch runner.
  - Runs the benchmark route with the predefined paid OpenRouter models.

- `comment_generation/model_api_adapter.py`
  - Shared backend adapter for `ollama` vs `api`.
  - Handles OpenRouter key loading, alias resolution, retries, and key rotation.

- `comment_generation/original_comment_loader.py`
  - Loads the benchmark original-comment files and merges them with normalized test records.

- `comment_generation/openrouter_api_pool.example.json`
  - Safe template config file.
  - The repo ships this example only. It does not include a real key.

- `comment_generation/douyin/comment_generate_api_ready.py`
  - Douyin API-ready backend used by the advanced entrypoints.
  - Wraps the active Douyin local generator and adds original-comment prompt support plus API backend support.

- `comment_generation/youtube/youtube_comment_generate_api_ready.py`
  - YouTube API-ready backend used by the advanced entrypoints.
  - Wraps the active YouTube local generator and adds original-comment prompt support plus API backend support.

### External runtime inputs used by the advanced route

These files are required at runtime, but they are not copied into `code_active` because they are data, not code:

- `E:\pn\new_GCG-main\comment_generation\original_comments_for_douyin.json`
- `E:\pn\new_GCG-main\comment_generation\original_comments_for_youtube.json`

They are the benchmark inputs used for the original-comments comparison route.

## Recommended Run Order For A New User

If your goal is to run the main pipeline end to end, use this order.

### 1. Run Step 1-3

Douyin:

```powershell
& $py .\data_pre\code_active\douyin\douyin_dataset_all_in_one_ollama.py --steps 1,2,3
```

YouTube:

```powershell
& $py .\data_pre\code_active\youtube\youtube_dataset_all_in_one_ollama.py --steps 1,2,3
```

### 2. Run Step 4 locally

```powershell
powershell -ExecutionPolicy Bypass -File .\data_pre\code_active\launch_step4_local_v2.ps1 -Platform both
```

If you only want one platform, replace `both` with `douyin` or `youtube`.

### 3. Build the Step 5 manifest

```powershell
& $py .\data_pre\code_active\build_codex_step5_manifest.py
```

### 4. Test Step 5 on one item first

```powershell
& $py .\data_pre\code_active\run_codex_step5_via_codex_cli.py --limit 1 --model gpt-5.4-mini --reasoning-effort medium
```

### 5. Run Step 5 in batch

```powershell
& $py .\data_pre\code_active\run_codex_step5_batches.py --concurrency 10 --model gpt-5.4-mini --reasoning-effort medium
```

### 6. Optional: generate comments from Step 5 outputs

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_descriptions.py --platform all --limit 5
```

This is optional. It does not replace Step 6.

### 7. Build Step 6 outputs

```powershell
& $py .\data_pre\code_active\run_step6_from_codex_outputs.py --platform all
```

## Comment Generation: How To Choose The Right Entry Script

Use this section if your goal is comment generation, not the full Step 1-6 pipeline.

### A. Generate comments from the latest Step 5 results

Use this when you already have:

- `data_pre\json\douyin\data_pre\douyin_video_description_codex.json`
- `data_pre\json\youtube\data_pre\youtube_video_description_codex.json`

Recommended command:

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_descriptions.py --platform all --limit 5
```

Outputs:

- `comment_generation\json\result\douyin_output_comments.json`
- `comment_generation\json\result\youtube_output_comments.json`

### B. Test the local comment flow from `test\*.json`

Use this when you want to debug the generator without using the latest Step 5 outputs.

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_test.py --platform all --limit 5
```

This also writes normalized test inputs under `comment_generation\json\douyin\...` and `comment_generation\json\youtube\...`.

### C. Test one API-ready model on benchmark-like inputs

Use this when you want to verify an API model before running a larger compare job.

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_test_api_ready.py `
  --platform douyin `
  --generation-backend api `
  --model-alias glm `
  --limit 3
```

You can also set `--generation-backend ollama` to reuse the same API-ready path with a local model.

Common aliases supported by the adapter include:

- `glm`
- `qwen`
- `dsr1`
- `llama`
- `gptoss`
- `r1`
- `gpt54`

Outputs:

- `comment_generation\json\result\douyin_output_comments_api_ready.json`
- `comment_generation\json\result\youtube_output_comments_api_ready.json`

### D. Compare multiple models on the original-comments benchmark

Use this when you want a side-by-side evaluation across several models without leaking the same-video original comments into the prompt.

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_original_comments_compare.py `
  --platform all `
  --models qwen3.5_9b qwen_free glm_air `
  --limit 5 `
  --resume
```

This route writes:

- `comment_generation\json\result\douyin_original_comments_model_compare.json`
- `comment_generation\json\result\douyin_original_comments_model_compare_meta.json`
- `comment_generation\json\result\youtube_original_comments_model_compare.json`
- `comment_generation\json\result\youtube_original_comments_model_compare_meta.json`

### E. Run the fixed paid-model API batch

Use this when you want the predefined paid OpenRouter models (`r1`, `llama`, `gpt54`) on the same no-leakage benchmark route.

```powershell
& $py .\data_pre\code_active\comment_generation\run_phase2_from_original_comments_api_batch.py `
  --platform all `
  --limit 5 `
  --resume
```

This route writes:

- `comment_generation\json\result\douyin_original_comments_paid_model_outputs.json`
- `comment_generation\json\result\youtube_original_comments_paid_model_outputs.json`

## What Not To Use First

- Do not start from `data_pre\code` unless you are intentionally debugging historical implementations.
- Do not start from `comment_generation\code` unless you need the historical source tree for comparison.
- Do not use the old shared / dual-host Step 4 route if your goal is just to run the active pipeline.
- Do not start from the backend comment scripts if you are new to the repo.
- Do not copy a real API key into the repository version of the template file.

## Legacy Files Intentionally Not Included In `code_active`

- `E:\pn\new_GCG-main\data_pre\code\douyin\douyin_all_in_one.py`
- `E:\pn\new_GCG-main\data_pre\code\run_codex_step5_batch.py`
- `E:\pn\new_GCG-main\data_pre\code\step4_local_gpu_runner.py`
- `E:\pn\new_GCG-main\data_pre\code\launch_step4_local_gpu.ps1`
- `E:\pn\new_GCG-main\comment_generation\code\douyin\douyin_comment_generate_hotmeme.py`
- cached embeddings, `.gitkeep`, and `__pycache__` folders are intentionally not copied into `code_active`
- old Step 4 shared files remain only in the historical `data_pre\code` folder
