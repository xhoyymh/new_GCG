from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_codex_step5_via_codex_cli import (
    DEFAULT_CODEX,
    DEFAULT_DOUYIN_OUT,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_PYTHON,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVER,
    DEFAULT_STATE,
    DEFAULT_YOUTUBE_OUT,
    McpClient,
    UsageLimitError,
    build_codex_command,
    build_prompt,
    ensure_json_array,
    usage_limit_in_text,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows path
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows path
    msvcrt = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO_ROOT / "data_pre" / "json" / "materials" / "codex_step5_codex_batches.log"
DEFAULT_LOCK = REPO_ROOT / "data_pre" / "json" / "materials" / "codex_step5_codex_batches.lock"
KILL_TIMEOUT_SECONDS = 5


@dataclass
class ActiveWorker:
    platform: str
    video_id: str
    preview_image_path: str
    launched_at: str
    command: list[str]
    process: subprocess.Popen[str]
    output_path: Path
    stdout_path: Path
    stderr_path: Path
    stdout_handle: Any
    stderr_handle: Any


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _counts(state: dict[str, Any]) -> dict[str, int]:
    records = state.get("records", [])
    result = {"pending": 0, "in_progress": 0, "done": 0, "failed": 0}
    for item in records:
        status = str(item.get("status", "pending"))
        if status in result:
            result[status] += 1
    return result


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _processed_cap(args: argparse.Namespace, concurrency: int) -> int:
    if args.max_total_processed > 0:
        return args.max_total_processed
    if args.max_batches > 0:
        return args.max_batches * concurrency
    return 0


def _resolve_concurrency(args: argparse.Namespace) -> int:
    if args.batch_size and args.batch_size > 0:
        return args.batch_size
    return args.concurrency


def _create_temp_path(suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        return Path(handle.name)


def _close_worker_handles(worker: ActiveWorker) -> None:
    for handle in (worker.stdout_handle, worker.stderr_handle):
        try:
            if handle and not handle.closed:
                handle.close()
        except Exception:
            pass


def _cleanup_worker(worker: ActiveWorker) -> None:
    _close_worker_handles(worker)
    for path in (worker.output_path, worker.stdout_path, worker.stderr_path):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _launch_worker(
    codex_path: Path,
    platform: str,
    video_id: str,
    prompt: str,
    preview_image_path: str,
    cwd: Path,
    model: str,
    reasoning_effort: str,
) -> ActiveWorker:
    output_path = _create_temp_path(".txt")
    stdout_path = _create_temp_path(".stdout.log")
    stderr_path = _create_temp_path(".stderr.log")
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    command = build_codex_command(
        codex_path=codex_path,
        output_path=output_path,
        cwd=cwd,
        preview_image_path=preview_image_path,
        model=model,
        reasoning_effort=reasoning_effort,
    )

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd),
        )
        if process.stdin is None:
            raise RuntimeError("Failed to open stdin for codex worker.")
        process.stdin.write(prompt)
        process.stdin.close()
    except Exception:
        try:
            if "process" in locals() and process.poll() is None:
                process.kill()
                process.wait(timeout=1)
        except Exception:
            pass
        worker = ActiveWorker(
            platform=platform,
            video_id=video_id,
            preview_image_path=preview_image_path,
            launched_at=_now(),
            command=command,
            process=process if "process" in locals() else None,  # type: ignore[arg-type]
            output_path=output_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )
        _cleanup_worker(worker)
        raise

    return ActiveWorker(
        platform=platform,
        video_id=video_id,
        preview_image_path=preview_image_path,
        launched_at=_now(),
        command=command,
        process=process,
        output_path=output_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )


def _finalize_worker(worker: ActiveWorker) -> str:
    _close_worker_handles(worker)
    returncode = worker.process.poll()
    if returncode is None:
        raise RuntimeError(f"Worker {worker.platform}/{worker.video_id} has not finished yet.")

    stdout_text = _read_text(worker.stdout_path)
    stderr_text = _read_text(worker.stderr_path)
    combined_output = f"{stdout_text}\n{stderr_text}".strip()

    if returncode != 0:
        if usage_limit_in_text(combined_output):
            raise UsageLimitError(combined_output[-1200:])
        raise RuntimeError(
            f"codex exec failed with code {returncode}. stdout={stdout_text[-400:]} stderr={stderr_text[-400:]}"
        )

    description = _read_text(worker.output_path).strip()
    if not description:
        raise RuntimeError("codex exec returned an empty final message.")
    return description


def _terminate_worker(worker: ActiveWorker) -> None:
    if worker.process.poll() is not None:
        return
    worker.process.terminate()
    try:
        worker.process.wait(timeout=KILL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        worker.process.kill()
        worker.process.wait(timeout=KILL_TIMEOUT_SECONDS)


def _requeue_worker(client: McpClient, worker: ActiveWorker, reason: str) -> None:
    client.call_tool(
        "step5_requeue_video",
        {
            "platform": worker.platform,
            "id": worker.video_id,
            "reason": reason,
        },
    )


def _stop_and_requeue_active_workers(
    active_workers: dict[tuple[str, str], ActiveWorker],
    client: McpClient,
    log_path: Path,
    reason: str,
) -> int:
    requeued = 0
    for key, worker in list(active_workers.items()):
        try:
            _terminate_worker(worker)
        except Exception as exc:
            _append_log(
                log_path,
                f"[{_now()}] worker_terminate_error platform={worker.platform} id={worker.video_id} error={str(exc)!r}\n",
            )
        try:
            _requeue_worker(client, worker, reason)
            requeued += 1
            _append_log(
                log_path,
                f"[{_now()}] requeue_after_global_stop platform={worker.platform} id={worker.video_id}\n",
            )
        except Exception as exc:
            _append_log(
                log_path,
                f"[{_now()}] requeue_error platform={worker.platform} id={worker.video_id} error={str(exc)!r}\n",
            )
        finally:
            _cleanup_worker(worker)
            active_workers.pop(key, None)
    return requeued


@contextmanager
def _coordinator_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            if msvcrt is None:  # pragma: no cover
                raise RuntimeError("msvcrt is unavailable on this Windows host.")
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            if fcntl is None:  # pragma: no cover
                raise RuntimeError("fcntl is unavailable on this host.")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except OSError as exc:
        raise RuntimeError(f"Another Step5 coordinator is already running. lock_path={lock_path}") from exc
    finally:
        try:
            handle.seek(0)
            if os.name == "nt" and msvcrt is not None:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step5 through a single-writer Codex worker pool.")
    parser.add_argument("--python-path", default=str(DEFAULT_PYTHON))
    parser.add_argument("--server-path", default=str(DEFAULT_SERVER))
    parser.add_argument("--codex-path", default=str(DEFAULT_CODEX))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--run-state-path", default=str(DEFAULT_STATE))
    parser.add_argument("--douyin-output-path", default=str(DEFAULT_DOUYIN_OUT))
    parser.add_argument("--youtube-output-path", default=str(DEFAULT_YOUTUBE_OUT))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK))
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Legacy alias for --concurrency. When provided, it overrides --concurrency.",
    )
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--max-total-processed",
        type=int,
        default=0,
        help="Soft stop after this many successful saves in the current coordinator session.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Codex model name.")
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=("low", "medium", "high", "xhigh"),
        help="Reasoning effort passed to Codex via config override.",
    )
    args = parser.parse_args()

    concurrency = _resolve_concurrency(args)
    if concurrency <= 0:
        raise SystemExit("--concurrency must be greater than 0.")

    processed_cap = _processed_cap(args, concurrency)
    python_path = Path(args.python_path)
    server_path = Path(args.server_path)
    codex_path = Path(args.codex_path)
    manifest_path = Path(args.manifest_path)
    run_state_path = Path(args.run_state_path)
    douyin_out = Path(args.douyin_output_path)
    youtube_out = Path(args.youtube_output_path)
    log_path = Path(args.log_path)
    lock_path = Path(args.lock_path)

    ensure_json_array(douyin_out)
    ensure_json_array(youtube_out)

    initial_state = _load_state(run_state_path)
    initial_counts = _counts(initial_state)
    initial_done = initial_counts["done"]
    processed = 0
    failed = 0
    launched = 0
    usage_limit_hit = False
    usage_limit_message = ""
    active_workers: dict[tuple[str, str], ActiveWorker] = {}

    _append_log(
        log_path,
        f"[{_now()}] coordinator_start concurrency={concurrency} processed_cap={processed_cap} model={args.model} reasoning_effort={args.reasoning_effort}\n",
    )

    with _coordinator_lock(lock_path):
        client = McpClient(
            python_path=python_path,
            server_path=server_path,
            manifest_path=manifest_path,
            state_path=run_state_path,
            douyin_description_path=douyin_out,
            youtube_description_path=youtube_out,
        )
        try:
            while True:
                while not usage_limit_hit and len(active_workers) < concurrency:
                    if processed_cap and processed + len(active_workers) >= processed_cap:
                        break

                    next_item = client.call_tool("step5_next_pending_video")
                    if not next_item:
                        break

                    platform = next_item["platform"]
                    video_id = str(next_item["id"])
                    key = (platform, video_id)

                    try:
                        context = client.call_tool("step5_prepare_video_context", {"platform": platform, "id": video_id})
                        prompt = build_prompt(context)
                        worker = _launch_worker(
                            codex_path=codex_path,
                            platform=platform,
                            video_id=video_id,
                            prompt=prompt,
                            preview_image_path=context.get("preview_image_path", ""),
                            cwd=REPO_ROOT,
                            model=args.model,
                            reasoning_effort=args.reasoning_effort,
                        )
                        active_workers[key] = worker
                        launched += 1
                        _append_log(
                            log_path,
                            f"[{_now()}] worker_launch platform={platform} id={video_id} active_workers={len(active_workers)} launched={launched}\n",
                        )
                    except Exception as exc:
                        failed += 1
                        try:
                            client.call_tool(
                                "step5_mark_failed",
                                {"platform": platform, "id": video_id, "reason": str(exc)},
                            )
                        except Exception as mark_exc:
                            _append_log(
                                log_path,
                                f"[{_now()}] worker_prepare_mark_failed_error platform={platform} id={video_id} error={str(mark_exc)!r}\n",
                            )
                        print(
                            json.dumps(
                                {
                                    "ok": False,
                                    "platform": platform,
                                    "id": video_id,
                                    "error": str(exc),
                                    "processed": processed,
                                    "failed": failed,
                                    "stage": "prepare_or_launch",
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        _append_log(
                            log_path,
                            f"[{_now()}] worker_prepare_failed platform={platform} id={video_id} error={str(exc)!r}\n",
                        )

                completed_keys = [
                    key
                    for key, worker in active_workers.items()
                    if worker.process.poll() is not None
                ]
                if not completed_keys:
                    status = client.call_tool("step5_run_status") or {}
                    if not active_workers:
                        if usage_limit_hit:
                            break
                        if processed_cap and processed >= processed_cap:
                            _append_log(
                                log_path,
                                f"[{_now()}] processed_cap_pause processed={processed} cap={processed_cap} pending={status.get('pending')} done={status.get('done')} failed={status.get('failed')}\n",
                            )
                            break
                        if status.get("pending", 0) <= 0 and status.get("in_progress", 0) <= 0:
                            _append_log(
                                log_path,
                                f"[{_now()}] stop pending={status.get('pending')} in_progress={status.get('in_progress')} failed={status.get('failed')} done={status.get('done')}\n",
                            )
                            break
                    time.sleep(max(args.pause_seconds, 0.0))
                    continue

                for key in completed_keys:
                    worker = active_workers.pop(key)
                    try:
                        description = _finalize_worker(worker)
                        result = client.call_tool(
                            "step5_save_description",
                            {"platform": worker.platform, "id": worker.video_id, "description": description},
                        )
                        processed += 1
                        print(
                            json.dumps(
                                {
                                    "ok": True,
                                    "platform": worker.platform,
                                    "id": worker.video_id,
                                    "last_duration_seconds": result.get("last_duration_seconds"),
                                    "processed": processed,
                                    "active_workers": len(active_workers),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        _append_log(
                            log_path,
                            f"[{_now()}] worker_complete platform={worker.platform} id={worker.video_id} processed={processed} active_workers={len(active_workers)}\n",
                        )
                    except UsageLimitError as exc:
                        usage_limit_hit = True
                        usage_limit_message = str(exc)
                        try:
                            _requeue_worker(
                                client,
                                worker,
                                "Paused and re-queued because Codex usage limit was reached.",
                            )
                        except Exception as requeue_exc:
                            _append_log(
                                log_path,
                                f"[{_now()}] usage_limit_requeue_error platform={worker.platform} id={worker.video_id} error={str(requeue_exc)!r}\n",
                            )
                        print(
                            json.dumps(
                                {
                                    "ok": False,
                                    "platform": worker.platform,
                                    "id": worker.video_id,
                                    "error": str(exc),
                                    "processed": processed,
                                    "failed": failed,
                                    "usage_limit_hit": True,
                                    "action": "global_stop_and_requeue",
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        _append_log(
                            log_path,
                            f"[{_now()}] usage_limit_pause platform={worker.platform} id={worker.video_id} active_workers={len(active_workers) + 1}\n",
                        )
                        _cleanup_worker(worker)
                        requeued = _stop_and_requeue_active_workers(
                            active_workers,
                            client,
                            log_path,
                            "Requeued after global stop because another worker hit Codex usage limit.",
                        )
                        _append_log(
                            log_path,
                            f"[{_now()}] usage_limit_global_stop requeued={requeued + 1} killed_workers={requeued}\n",
                        )
                        break
                    except Exception as exc:
                        failed += 1
                        try:
                            client.call_tool(
                                "step5_mark_failed",
                                {"platform": worker.platform, "id": worker.video_id, "reason": str(exc)},
                            )
                        except Exception as mark_exc:
                            _append_log(
                                log_path,
                                f"[{_now()}] worker_failed_mark_error platform={worker.platform} id={worker.video_id} error={str(mark_exc)!r}\n",
                            )
                        print(
                            json.dumps(
                                {
                                    "ok": False,
                                    "platform": worker.platform,
                                    "id": worker.video_id,
                                    "error": str(exc),
                                    "processed": processed,
                                    "failed": failed,
                                    "stage": "finalize_or_save",
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        _append_log(
                            log_path,
                            f"[{_now()}] worker_failed platform={worker.platform} id={worker.video_id} error={str(exc)!r}\n",
                        )
                    finally:
                        _cleanup_worker(worker)

                if usage_limit_hit:
                    break
        finally:
            if active_workers:
                requeued = _stop_and_requeue_active_workers(
                    active_workers,
                    client,
                    log_path,
                    "Requeued after coordinator shutdown before result persistence.",
                )
                _append_log(
                    log_path,
                    f"[{_now()}] coordinator_shutdown_requeue requeued={requeued}\n",
                )
            status = client.call_tool("step5_run_status") or {}
            print(
                json.dumps(
                    {
                        "summary": {
                            key: status.get(key)
                            for key in ("record_count", "pending", "in_progress", "done", "failed")
                        },
                        "processed_in_this_run": processed,
                        "failed_in_this_run": failed,
                        "launched_in_this_run": launched,
                        "usage_limit_hit": usage_limit_hit,
                        "usage_limit_message": usage_limit_message,
                        "concurrency": concurrency,
                        "model": args.model,
                        "reasoning_effort": args.reasoning_effort,
                        "douyin_output_path": str(douyin_out),
                        "youtube_output_path": str(youtube_out),
                        "run_state_path": str(run_state_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            _append_log(
                log_path,
                f"[{_now()}] coordinator_end processed={processed} failed={failed} usage_limit_hit={usage_limit_hit}\n",
            )
            client.close()

    if usage_limit_hit:
        raise SystemExit(2)

    final_done = _counts(_load_state(run_state_path))["done"]
    if processed_cap and max(final_done - initial_done, 0) >= processed_cap:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
