from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parent

DEFAULT_PYTHON = Path(r"C:\Users\bobzhou\anaconda3\envs\gcg_douyin311\python.exe")
DEFAULT_SERVER = CODE_ROOT / "codex_step5_mcp_server.py"
DEFAULT_CODEX = REPO_ROOT / "tmp_codex.exe"
DEFAULT_MANIFEST = REPO_ROOT / "data_pre" / "json" / "materials" / "successful_step5_candidates_20260329.json"
DEFAULT_STATE = REPO_ROOT / "data_pre" / "json" / "materials" / "codex_step5_codex_run_20260329.json"
DEFAULT_DOUYIN_OUT = REPO_ROOT / "data_pre" / "json" / "douyin" / "data_pre" / "douyin_video_description_codex.json"
DEFAULT_YOUTUBE_OUT = REPO_ROOT / "data_pre" / "json" / "youtube" / "data_pre" / "youtube_video_description_codex.json"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TIMEOUT_SECONDS = 1800


class UsageLimitError(RuntimeError):
    pass


class McpClient:
    def __init__(
        self,
        python_path: Path,
        server_path: Path,
        manifest_path: Path,
        state_path: Path,
        douyin_description_path: Path,
        youtube_description_path: Path,
    ) -> None:
        self.proc = subprocess.Popen(
            [
                str(python_path),
                str(server_path),
                "--manifest-path",
                str(manifest_path),
                "--run-state-path",
                str(state_path),
                "--douyin-description-path",
                str(douyin_description_path),
                "--youtube-description-path",
                str(youtube_description_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._message_id = 1
        self._initialize()

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        assert self.proc.stdin is not None
        self.proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
        self.proc.stdin.write(body)
        self.proc.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        headers: dict[str, str] = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr = b""
                if self.proc.stderr is not None:
                    stderr = self.proc.stderr.read()
                raise RuntimeError(f"MCP server closed unexpectedly. stderr={stderr.decode('utf-8', errors='replace')}")
            if line in (b"\r\n", b"\n"):
                break
            key, value = line.decode("utf-8").split(":", 1)
            headers[key.strip().lower()] = value.strip()
        payload = self.proc.stdout.read(int(headers["content-length"]))
        return json.loads(payload.decode("utf-8"))

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        message_id = self._message_id
        self._message_id += 1
        self._send({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params or {}})
        response = self._recv()
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def _initialize(self) -> None:
        _ = self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "codex-cli-batch-runner", "version": "1.0"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any] | None:
        result = self._request("tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            text = ""
            for item in result.get("content", []):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    break
            raise RuntimeError(text or f"Tool call failed: {name}")
        return result.get("structuredContent")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def ensure_json_array(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]\n", encoding="utf-8")


def build_prompt(context: dict[str, Any]) -> str:
    meta = context["video_meta"]
    top_comments = context.get("top_comments", [])
    comments_block = "\n".join(f"- {comment}" for comment in top_comments[:5]) or "- None"
    sampled_frames = "\n".join(context.get("sampled_frame_names", [])[:24])

    return f"""You are writing the final Step5 video description for one short video.

Output requirements:
- Return exactly one polished paragraph in plain text.
- No Markdown, bullets, JSON, headings, or meta commentary.
- Describe the video itself, not the extraction process.
- Use the same language that best matches the source material. If the intro/transcript are mainly Chinese, write Chinese. Otherwise write English.
- Reconcile noisy transcript text with the storyboard image and comments.
- Do not mention frames, screenshots, OCR, ASR, prompts, or uncertainty.

Video metadata:
- platform: {meta.get("platform", "")}
- id: {meta.get("id", "")}
- label: {meta.get("label", "")}
- video_url: {meta.get("video_url", "")}
- video_introduction: {meta.get("video_introduction", "")}

Transcript:
{context.get("transcript", "")}

Top comments:
{comments_block}

Storyboard frame file names:
{sampled_frames}
"""


def usage_limit_in_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return "usage limit" in lowered or "you've hit your usage limit" in lowered


def build_codex_command(
    codex_path: Path,
    output_path: Path,
    cwd: Path,
    preview_image_path: str = "",
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> list[str]:
    command = [
        str(codex_path),
        "exec",
        "-C",
        str(cwd),
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--color",
        "never",
        "-o",
        str(output_path),
    ]
    if model:
        command.extend(["-m", model])
    if reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    if preview_image_path:
        command.extend(["-i", preview_image_path])
    command.append("-")
    return command


def run_codex_once(
    codex_path: Path,
    prompt: str,
    preview_image_path: str,
    cwd: Path,
    model: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as output_file:
        output_path = Path(output_file.name)

    try:
        command = build_codex_command(
            codex_path=codex_path,
            output_path=output_path,
            cwd=cwd,
            preview_image_path=preview_image_path,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            cwd=str(cwd),
        )
        if completed.returncode != 0:
            combined_output = f"{completed.stdout}\n{completed.stderr}".strip()
            if usage_limit_in_text(combined_output):
                raise UsageLimitError(combined_output[-1200:])
            raise RuntimeError(
                f"codex exec failed with code {completed.returncode}. stdout={completed.stdout[-400:]} stderr={completed.stderr[-400:]}"
            )
        description = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if not description:
            raise RuntimeError("codex exec returned an empty final message.")
        return description
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step5 through real Codex CLI calls and save results via the local MCP server.")
    parser.add_argument("--python-path", default=str(DEFAULT_PYTHON))
    parser.add_argument("--server-path", default=str(DEFAULT_SERVER))
    parser.add_argument("--codex-path", default=str(DEFAULT_CODEX))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--run-state-path", default=str(DEFAULT_STATE))
    parser.add_argument("--douyin-output-path", default=str(DEFAULT_DOUYIN_OUT))
    parser.add_argument("--youtube-output-path", default=str(DEFAULT_YOUTUBE_OUT))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Codex model name.")
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=("low", "medium", "high", "xhigh"),
        help="Reasoning effort passed to Codex via config override.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of items to process this run.")
    args = parser.parse_args()

    python_path = Path(args.python_path)
    server_path = Path(args.server_path)
    codex_path = Path(args.codex_path)
    manifest_path = Path(args.manifest_path)
    state_path = Path(args.run_state_path)
    douyin_out = Path(args.douyin_output_path)
    youtube_out = Path(args.youtube_output_path)

    ensure_json_array(douyin_out)
    ensure_json_array(youtube_out)

    client = McpClient(
        python_path=python_path,
        server_path=server_path,
        manifest_path=manifest_path,
        state_path=state_path,
        douyin_description_path=douyin_out,
        youtube_description_path=youtube_out,
    )

    processed = 0
    failed = 0
    usage_limit_hit = False
    usage_limit_message = ""
    try:
        while True:
            if args.limit and processed >= args.limit:
                break

            next_item = client.call_tool("step5_next_pending_video")
            if not next_item:
                break

            platform = next_item["platform"]
            video_id = str(next_item["id"])
            try:
                context = client.call_tool("step5_prepare_video_context", {"platform": platform, "id": video_id})
                prompt = build_prompt(context)
                description = run_codex_once(
                    codex_path=codex_path,
                    prompt=prompt,
                    preview_image_path=context.get("preview_image_path", ""),
                    cwd=REPO_ROOT,
                    model=args.model or None,
                    reasoning_effort=args.reasoning_effort or None,
                )
                result = client.call_tool(
                    "step5_save_description",
                    {"platform": platform, "id": video_id, "description": description},
                )
                processed += 1
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "platform": platform,
                            "id": video_id,
                            "last_duration_seconds": result.get("last_duration_seconds"),
                            "processed": processed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except UsageLimitError as exc:
                usage_limit_hit = True
                usage_limit_message = str(exc)
                client.call_tool(
                    "step5_requeue_video",
                    {
                        "platform": platform,
                        "id": video_id,
                        "reason": "Paused and re-queued because Codex usage limit was reached.",
                    },
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
                            "usage_limit_hit": True,
                            "action": "requeued_and_stop",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break
            except Exception as exc:
                failed += 1
                client.call_tool("step5_mark_failed", {"platform": platform, "id": video_id, "reason": str(exc)})
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "platform": platform,
                            "id": video_id,
                            "error": str(exc),
                            "processed": processed,
                            "failed": failed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        status = client.call_tool("step5_run_status") or {}
        print(
            json.dumps(
                {
                    "summary": {key: status.get(key) for key in ("record_count", "pending", "in_progress", "done", "failed")},
                    "processed_in_this_run": processed,
                    "failed_in_this_run": failed,
                    "usage_limit_hit": usage_limit_hit,
                    "usage_limit_message": usage_limit_message,
                    "douyin_output_path": str(douyin_out),
                    "youtube_output_path": str(youtube_out),
                    "run_state_path": str(state_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        client.close()

    if usage_limit_hit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
