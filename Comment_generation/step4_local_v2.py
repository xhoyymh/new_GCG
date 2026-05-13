import argparse
import filecmp
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np

try:
    import av
except Exception:
    av = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

import torch
import whisper


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SAMPLE_SECONDS = 3.0
DEFAULT_EXTRACT_WORKERS = 3
DEFAULT_STATUS_SECONDS = 60
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_LOG_DIR = REPO_ROOT / "data_pre" / "logs" / "step4_local_v2"
COMMON_LABEL_SLUG_MAP = {
    "Comedy Skits": "comedy_skits",
    "Daily Life Jokes": "daily_life_jokes",
    "Funny Animal Videos": "funny_animal_videos",
    "Humorous Commentary": "humorous_commentary",
    "Talk Shows / Stand-Up Comedy / Cross-Talk": "talk_show_standup_crosstalk",
}


@dataclass(frozen=True)
class PlatformProfile:
    name: str
    intro_json: Path
    chouzhen_json: Path
    failed_json: Path
    image_dir_name: str
    video_root: Path
    label_slug_map: dict[str, str]
    include_image_list: bool = False
    extra_fields: tuple[str, ...] = ()


@dataclass
class ToolPaths:
    ffmpeg: str | None = None
    ffprobe: str | None = None


@dataclass
class ProbeInfo:
    has_video: bool
    has_audio: bool
    duration_seconds: float
    fps: float
    frame_count: int


@dataclass
class PreparedItem:
    record: dict
    resolved_video_path: Path
    image_root_rel: str
    final_root_abs: Path
    temp_root: Path
    frame_files: list[Path]
    audio_input: object
    cleanup_paths: list[Path]
    metrics: dict
    frame_backend: str


@dataclass
class FailureItem:
    record: dict
    resolved_video_path: str
    reason_code: str
    message: str
    metrics: dict
    temp_root: Path | None = None
    cleanup_paths: list[Path] = field(default_factory=list)


@dataclass
class WorkerDone:
    worker_name: str


class Step4Error(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class InterruptController:
    def __init__(self):
        self.stop_requested = threading.Event()
        self._count = 0
        self._lock = threading.Lock()

    def handle_signal(self, signum, _frame):
        with self._lock:
            self._count += 1
            current = self._count
        if current == 1:
            self.stop_requested.set()
            print(f"[interrupt] signal={signum} | graceful-stop requested; current in-flight videos will finish.")
            return
        print(f"[interrupt] signal={signum} | force exit requested.")
        raise SystemExit(130)


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class PendingSource:
    def __init__(self, records: list[dict], stop_event: threading.Event):
        self._records = records
        self._stop_event = stop_event
        self._index = 0
        self._lock = threading.Lock()

    def next_record(self) -> dict | None:
        with self._lock:
            if self._stop_event.is_set():
                return None
            if self._index >= len(self._records):
                return None
            record = self._records[self._index]
            self._index += 1
            return record


class ProgressReporter(threading.Thread):
    def __init__(self, platform: str, stats: dict, stats_lock: threading.Lock, stop_event: threading.Event, interval_seconds: int):
        super().__init__(daemon=True)
        self.platform = platform
        self.stats = stats
        self.stats_lock = stats_lock
        self.stop_event = stop_event
        self.interval_seconds = max(int(interval_seconds), 1)

    def run(self):
        while not self.stop_event.wait(self.interval_seconds):
            with self.stats_lock:
                snapshot = dict(self.stats)
            print(
                f"[{self.platform}] progress processed={snapshot['processed']}/{snapshot['total']} "
                f"success={snapshot['success']} failed={snapshot['failed']} "
                f"extract_inflight={snapshot['extract_inflight']} transcribe_inflight={snapshot['transcribe_inflight']}"
            )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_relpath(value: str | Path) -> str:
    if not value:
        return ""
    return os.path.normpath(str(value))


def sort_records_by_id(records: list[dict]) -> list[dict]:
    def key(item: dict):
        value = str(item.get("id", "")).strip()
        if value.isdigit():
            return (0, int(value))
        return (1, value)

    return sorted(records, key=key)


def slugify_label(label: str, label_slug_map: dict[str, str]) -> str:
    label = str(label or "").strip()
    if label in label_slug_map:
        return label_slug_map[label]
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def to_repo_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return normalize_relpath(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return normalize_relpath(path)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tmp_path = Path(tf.name)
    os.replace(tmp_path, path)


def build_profiles() -> dict[str, PlatformProfile]:
    return {
        "douyin": PlatformProfile(
            name="douyin",
            intro_json=REPO_ROOT / "data_pre" / "json" / "douyin" / "data_pre" / "douyin_video_introduction.json",
            chouzhen_json=REPO_ROOT / "data_pre" / "json" / "douyin" / "data_pre" / "douyin_chouzhen.json",
            failed_json=REPO_ROOT / "data_pre" / "json" / "douyin" / "data_pre" / "douyin_step4_failed.json",
            image_dir_name="douyin_image",
            video_root=REPO_ROOT / "data_pre" / "video" / "douyin",
            label_slug_map=COMMON_LABEL_SLUG_MAP,
            include_image_list=True,
        ),
        "youtube": PlatformProfile(
            name="youtube",
            intro_json=REPO_ROOT / "data_pre" / "json" / "youtube" / "data_pre" / "youtube_video_introduction.json",
            chouzhen_json=REPO_ROOT / "data_pre" / "json" / "youtube" / "data_pre" / "youtube_chouzhen.json",
            failed_json=REPO_ROOT / "data_pre" / "json" / "youtube" / "data_pre" / "youtube_step4_failed.json",
            image_dir_name="youtube_image",
            video_root=REPO_ROOT / "data_pre" / "video" / "youtube",
            label_slug_map=COMMON_LABEL_SLUG_MAP,
            include_image_list=False,
            extra_fields=("video_api_description",),
        ),
    }


def load_output_map(path: Path) -> dict[str, dict]:
    data = load_json(path, [])
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    if isinstance(data, list):
        result = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("id", "")).strip()
            if record_id:
                result[record_id] = item
        return result
    return {}


def load_failure_map(path: Path) -> dict[str, dict]:
    data = load_json(path, [])
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    if isinstance(data, list):
        result = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("id", "")).strip()
            if record_id:
                result[record_id] = item
        return result
    return {}


def save_failure_map(path: Path, failure_map: dict[str, dict]) -> None:
    dump_json_atomic(path, sort_records_by_id(list(failure_map.values())))


def build_image_root_relpath(profile: PlatformProfile, record_id: str, label: str) -> str:
    slug = slugify_label(label, profile.label_slug_map)
    return normalize_relpath(Path("data_pre") / profile.image_dir_name / slug / str(record_id))


def resolve_video_path(profile: PlatformProfile, record: dict) -> Path:
    raw_path = str(record.get("video_path", "")).strip()
    candidates: list[Path] = []
    if raw_path:
        raw = Path(raw_path)
        candidates.append(raw if raw.is_absolute() else REPO_ROOT / raw)

    record_id = str(record.get("id", "")).strip()
    label = record.get("label", "")
    slug = slugify_label(label, profile.label_slug_map)
    if record_id:
        candidates.append(profile.video_root / slug / f"{record_id}.mp4")
        candidates.append(profile.video_root / f"{record_id}.mp4")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else Path()


def list_frame_files(frames_dir: Path) -> list[Path]:
    if not frames_dir.is_dir():
        return []
    return sorted(
        [item for item in frames_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda item: item.name,
    )


def is_valid_step4_record(record: dict) -> bool:
    image_root_rel = normalize_relpath(record.get("image_root", ""))
    transcript = str(record.get("all_transcription", "")).strip()
    if not image_root_rel or not transcript:
        return False
    image_root_abs = REPO_ROOT / image_root_rel
    frames_dir = image_root_abs / "frames"
    if not list_frame_files(frames_dir):
        return False
    txt_path = image_root_abs / "transcription.txt"
    if not txt_path.is_file():
        return False
    return bool(txt_path.read_text(encoding="utf-8").strip())


def find_binary(name: str, extra_candidates: list[str]) -> str | None:
    found = shutil.which(name)
    if found and os.path.isfile(found):
        return found
    for candidate in extra_candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def find_ffmpeg() -> str | None:
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft",
            "WinGet",
            "Packages",
            "Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe",
            "ffmpeg-8.1-essentials_build",
            "bin",
            "ffmpeg.exe",
        ),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""), "ffmpeg", "bin", "ffmpeg.exe"),
    ]
    return find_binary("ffmpeg", candidates)


def find_ffprobe(ffmpeg_path: str | None) -> str | None:
    candidates = [
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe",
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft",
            "WinGet",
            "Packages",
            "Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe",
            "ffmpeg-8.1-essentials_build",
            "bin",
            "ffprobe.exe",
        ),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "ffmpeg", "bin", "ffprobe.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""), "ffmpeg", "bin", "ffprobe.exe"),
    ]
    if ffmpeg_path:
        sibling = ffmpeg_path.replace("ffmpeg.EXE", "ffprobe.EXE").replace("ffmpeg.exe", "ffprobe.exe")
        candidates.insert(0, sibling)
    return find_binary("ffprobe", candidates)


def detect_tool_paths() -> ToolPaths:
    ffmpeg_path = find_ffmpeg()
    return ToolPaths(ffmpeg=ffmpeg_path, ffprobe=find_ffprobe(ffmpeg_path))


def probe_media_pyav(video_path: Path) -> ProbeInfo:
    if av is None:
        raise Step4Error("probe_failed", "PyAV unavailable")
    try:
        with av.open(str(video_path)) as container:
            video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
            audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
            duration_seconds = float(container.duration) / 1_000_000.0 if container.duration is not None else 0.0
            fps = 0.0
            frame_count = 0
            if video_stream is not None:
                avg_rate = getattr(video_stream, "average_rate", None)
                try:
                    if avg_rate is not None:
                        fps = float(avg_rate)
                except Exception:
                    fps = 0.0
                frame_count = int(getattr(video_stream, "frames", 0) or 0)
            return ProbeInfo(
                has_video=video_stream is not None,
                has_audio=audio_stream is not None,
                duration_seconds=max(duration_seconds, 0.0),
                fps=max(fps, 0.0),
                frame_count=max(frame_count, 0),
            )
    except Step4Error:
        raise
    except Exception as exc:
        raise Step4Error("probe_failed", str(exc)) from exc


def probe_media_ffprobe(video_path: Path, ffprobe_path: str) -> ProbeInfo:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=True,
        )
        payload = json.loads(result.stdout or "{}")
    except Exception as exc:
        raise Step4Error("probe_failed", f"ffprobe failed: {exc}") from exc

    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    duration_seconds = 0.0
    try:
        duration_seconds = float((payload.get("format") or {}).get("duration") or 0.0)
    except Exception:
        duration_seconds = 0.0
    fps = 0.0
    if video_stream:
        for key in ("avg_frame_rate", "r_frame_rate"):
            raw = str(video_stream.get(key, "")).strip()
            if not raw or raw in {"0/0", "N/A"}:
                continue
            if "/" in raw:
                left, right = raw.split("/", 1)
                try:
                    denominator = float(right)
                    if denominator:
                        fps = float(left) / denominator
                        break
                except Exception:
                    continue
            else:
                try:
                    fps = float(raw)
                    break
                except Exception:
                    continue
    frame_count = 0
    if video_stream:
        raw = str(video_stream.get("nb_frames", "")).strip()
        if raw.isdigit():
            frame_count = int(raw)
    return ProbeInfo(
        has_video=video_stream is not None,
        has_audio=audio_stream is not None,
        duration_seconds=max(duration_seconds, 0.0),
        fps=max(fps, 0.0),
        frame_count=max(frame_count, 0),
    )


def probe_media(video_path: Path, tool_paths: ToolPaths) -> ProbeInfo:
    pyav_error = None
    if av is not None:
        try:
            return probe_media_pyav(video_path)
        except Step4Error as exc:
            pyav_error = exc
    if tool_paths.ffprobe:
        return probe_media_ffprobe(video_path, tool_paths.ffprobe)
    if pyav_error is not None:
        raise pyav_error
    raise Step4Error("probe_failed", "no probe backend available")


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def extract_tail_frame_ffmpeg(video_path: Path, output_path: Path, ffmpeg_path: str, duration_seconds: float, use_cuda: bool) -> bool:
    seek_seconds = max(duration_seconds - 0.1, 0.0)
    cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y"]
    if use_cuda:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-ss", f"{seek_seconds:.3f}", "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(output_path)]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=60)
        return output_path.is_file() and output_path.stat().st_size > 0
    except Exception:
        output_path.unlink(missing_ok=True)
        return False


def sample_frames_ffmpeg(video_path: Path, frames_dir: Path, sample_seconds: float, probe: ProbeInfo, ffmpeg_path: str) -> tuple[list[Path], str]:
    clear_directory(frames_dir)
    output_pattern = frames_dir / "frame_%06d.jpg"

    def run_extract(use_cuda: bool) -> tuple[list[Path], str]:
        cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y"]
        if use_cuda:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-i", str(video_path), "-vf", f"fps=1/{sample_seconds}", "-q:v", "2", str(output_pattern)]
        timeout_seconds = max(120, int(max(probe.duration_seconds, sample_seconds) * 2) + 30)
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise Step4Error("frame_extract_failed", f"ffmpeg timeout: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
            raise Step4Error("frame_extract_failed", stderr or "ffmpeg frame extract failed") from exc

        frame_files = list_frame_files(frames_dir)
        if not frame_files:
            raise Step4Error("empty_frames", "ffmpeg extracted zero frames")

        tail_path = frames_dir / "__tail__.jpg"
        if probe.duration_seconds > sample_seconds and extract_tail_frame_ffmpeg(video_path, tail_path, ffmpeg_path, probe.duration_seconds, use_cuda):
            last_frame = frame_files[-1]
            if not filecmp.cmp(str(last_frame), str(tail_path), shallow=False):
                appended = frames_dir / f"frame_{len(frame_files) + 1:06d}.jpg"
                os.replace(tail_path, appended)
                frame_files.append(appended)
            else:
                tail_path.unlink(missing_ok=True)
        return frame_files, "ffmpeg-cuda" if use_cuda else "ffmpeg"

    try:
        return run_extract(use_cuda=True)
    except Step4Error:
        clear_directory(frames_dir)
        return run_extract(use_cuda=False)


def sample_frames_cv2(video_path: Path, frames_dir: Path, sample_seconds: float, probe: ProbeInfo) -> tuple[list[Path], str]:
    clear_directory(frames_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Step4Error("frame_extract_failed", "opencv failed to open video")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or probe.fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or probe.frame_count
        if fps <= 0:
            raise Step4Error("frame_extract_failed", "invalid video fps")

        interval_frames = max(int(round(fps * max(sample_seconds, 0.1))), 1)
        target_indices = {0}
        if frame_count > 0:
            current = interval_frames
            while current < frame_count:
                target_indices.add(current)
                current += interval_frames
            target_indices.add(max(frame_count - 1, 0))

        frame_files: list[Path] = []
        frame_id = 0
        saved = 0
        last_frame = None
        last_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
            last_index = frame_id
            should_save = frame_id in target_indices if frame_count > 0 else (frame_id == 0 or frame_id % interval_frames == 0)
            if should_save:
                saved += 1
                output_path = frames_dir / f"frame_{saved:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise Step4Error("frame_extract_failed", f"failed to write frame {saved}")
                frame_files.append(output_path)
            frame_id += 1

        if last_frame is not None and last_index not in target_indices:
            saved += 1
            output_path = frames_dir / f"frame_{saved:06d}.jpg"
            if not cv2.imwrite(str(output_path), last_frame):
                raise Step4Error("frame_extract_failed", "failed to write last frame")
            frame_files.append(output_path)

        if not frame_files:
            raise Step4Error("empty_frames", "opencv extracted zero frames")
        return frame_files, "opencv"
    finally:
        cap.release()


def sample_frames(video_path: Path, frames_dir: Path, sample_seconds: float, probe: ProbeInfo, tool_paths: ToolPaths) -> tuple[list[Path], str]:
    if tool_paths.ffmpeg:
        try:
            return sample_frames_ffmpeg(video_path, frames_dir, sample_seconds, probe, tool_paths.ffmpeg)
        except Step4Error:
            clear_directory(frames_dir)
    return sample_frames_cv2(video_path, frames_dir, sample_seconds, probe)


def extract_audio_file(video_path: Path, ffmpeg_path: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        audio_path = Path(tf.name)
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=120)
    except Exception as exc:
        audio_path.unlink(missing_ok=True)
        raise Step4Error("transcribe_failed", f"ffmpeg audio extract failed: {exc}") from exc
    return audio_path


def decode_audio_input(video_path: Path, tool_paths: ToolPaths) -> tuple[object, list[Path]]:
    if av is not None:
        try:
            with av.open(str(video_path)) as container:
                audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
                if audio_stream is None:
                    raise Step4Error("missing_audio_stream", "missing audio stream")
                resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
                chunks: list[np.ndarray] = []
                for frame in container.decode(audio_stream):
                    resampled = resampler.resample(frame)
                    if resampled is None:
                        continue
                    frames = resampled if isinstance(resampled, list) else [resampled]
                    for item in frames:
                        chunk = item.to_ndarray()
                        if chunk is None:
                            continue
                        chunk = np.asarray(chunk).reshape(-1)
                        if chunk.size:
                            chunks.append(chunk)
                if not chunks:
                    raise Step4Error("transcribe_failed", "empty decoded audio")
                audio = np.concatenate(chunks).astype(np.float32) / 32768.0
                if audio.size == 0:
                    raise Step4Error("transcribe_failed", "empty decoded audio")
                return audio, []
        except Step4Error:
            raise
        except Exception as exc:
            if not tool_paths.ffmpeg:
                raise Step4Error("transcribe_failed", f"audio decode failed: {exc}") from exc

    if not tool_paths.ffmpeg:
        raise Step4Error("transcribe_failed", "no audio decode backend available")
    audio_path = extract_audio_file(video_path, tool_paths.ffmpeg)
    return audio_path, [audio_path]


class Transcriber:
    def __init__(self, model_size: str):
        self.backend = "whisper"
        self.device = "cpu"
        self.model = None

        if WhisperModel is not None and torch.cuda.is_available():
            try:
                self.model = WhisperModel(
                    model_size,
                    device="cuda",
                    compute_type="float16",
                    device_index=0,
                    local_files_only=True,
                )
                self.backend = "faster-whisper"
                self.device = "cuda"
                return
            except Exception:
                self.model = None

        self.model = whisper.load_model(model_size)
        self.device = str(self.model.device)

    def transcribe(self, audio_input) -> str:
        if self.backend == "faster-whisper":
            segments, _ = self.model.transcribe(
                audio_input,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            texts = [segment.text.strip() for segment in segments if segment.text.strip()]
            return " ".join(texts).strip()

        result = self.model.transcribe(audio_input, fp16=torch.cuda.is_available())
        return str(result.get("text", "")).strip()


def make_temp_root(base_dir: Path, final_name: str) -> Path:
    stamp = f".{final_name}.tmp-{os.getpid()}-{threading.get_ident()}-{int(time.time() * 1000)}"
    temp_root = base_dir / stamp
    temp_root.mkdir(parents=True, exist_ok=False)
    return temp_root


def cleanup_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except Exception:
            pass


def build_output_record(profile: PlatformProfile, source: dict, image_root_rel: str, video_path: Path, frame_files: list[Path], transcript: str) -> dict:
    record = {
        "id": str(source.get("id", "")).strip(),
        "video_url": source.get("video_url", ""),
        "video_path": to_repo_relative(video_path),
        "video_introduction": source.get("video_introduction", ""),
        "label": source.get("label", ""),
        "image_root": normalize_relpath(image_root_rel),
        "all_transcription": transcript,
    }
    for key in profile.extra_fields:
        if key in source:
            record[key] = source.get(key, "")
    if profile.include_image_list:
        record["image"] = [
            normalize_relpath(Path(image_root_rel) / "frames" / frame_file.name)
            for frame_file in frame_files
        ]
    return record


def commit_output_tree(temp_root: Path, final_root: Path) -> None:
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        shutil.rmtree(final_root, ignore_errors=True)
    os.replace(temp_root, final_root)


def record_failure(failure_map: dict[str, dict], failed_json: Path, record: dict, resolved_video_path: str | Path, reason_code: str, reason: str) -> None:
    record_id = str(record.get("id", "")).strip()
    if not record_id:
        return
    failure_map[record_id] = {
        "id": record_id,
        "label": record.get("label", ""),
        "video_path": normalize_relpath(resolved_video_path) if resolved_video_path else "",
        "reason_code": reason_code,
        "reason": reason,
        "updated_at": utc_now_iso(),
    }
    save_failure_map(failed_json, failure_map)


def remove_failure(failure_map: dict[str, dict], failed_json: Path, record_id: str) -> None:
    if record_id in failure_map:
        failure_map.pop(record_id, None)
        save_failure_map(failed_json, failure_map)


def resolve_candidates(profile: PlatformProfile, args) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    intro_records = load_json(profile.intro_json, [])
    intro_records = sort_records_by_id(intro_records if isinstance(intro_records, list) else [])
    intro_map = {str(item.get("id", "")).strip(): item for item in intro_records if str(item.get("id", "")).strip()}
    output_map = load_output_map(profile.chouzhen_json)
    failure_map = load_failure_map(profile.failed_json)

    selected_ids = [str(item).strip() for item in (args.ids or []) if str(item).strip()]
    selected_id_set = set(selected_ids) if selected_ids else None

    if selected_ids:
        ordered_records = [intro_map[item] for item in selected_ids if item in intro_map]
    else:
        ordered_records = intro_records

    pending: list[dict] = []
    for record in ordered_records:
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            continue
        if selected_id_set is not None and record_id not in selected_id_set:
            continue
        if args.scratch_root and selected_id_set is not None:
            pending.append(record)
            continue
        existing = output_map.get(record_id)
        if existing and is_valid_step4_record(existing):
            continue
        if not args.retry_failed and record_id in failure_map:
            continue
        pending.append(record)

    if args.reverse:
        pending = list(reversed(pending))
    if args.limit > 0:
        pending = pending[: args.limit]

    return pending, output_map, failure_map


def summarize_labels(records: list[dict]) -> Counter:
    return Counter(str(item.get("label", "")).strip() for item in records)


def prepare_record(profile: PlatformProfile, record: dict, args, tool_paths: ToolPaths) -> PreparedItem:
    metrics = {"probe_ms": 0, "frame_ms": 0, "audio_ms": 0}
    record_id = str(record.get("id", "")).strip()
    label = record.get("label", "")
    video_path = resolve_video_path(profile, record)
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise Step4Error("probe_failed", f"video not found: {video_path or 'N/A'}")

    probe_started = time.perf_counter()
    probe = probe_media(video_path, tool_paths)
    metrics["probe_ms"] = round((time.perf_counter() - probe_started) * 1000, 2)
    if not probe.has_video:
        raise Step4Error("missing_video_stream", "missing video stream")
    if not probe.has_audio:
        raise Step4Error("missing_audio_stream", "missing audio stream")

    image_root_rel = build_image_root_relpath(profile, record_id, label)
    if args.scratch_root:
        final_root_abs = Path(args.scratch_root) / profile.name / slugify_label(label, profile.label_slug_map) / record_id
    else:
        final_root_abs = REPO_ROOT / image_root_rel

    temp_root = make_temp_root(final_root_abs.parent, final_root_abs.name)
    frames_dir = temp_root / "frames"
    cleanup = [temp_root]

    try:
        frame_started = time.perf_counter()
        frame_files, frame_backend = sample_frames(video_path, frames_dir, args.sample_seconds, probe, tool_paths)
        metrics["frame_ms"] = round((time.perf_counter() - frame_started) * 1000, 2)

        audio_started = time.perf_counter()
        audio_input, extra_cleanup = decode_audio_input(video_path, tool_paths)
        cleanup.extend(extra_cleanup)
        metrics["audio_ms"] = round((time.perf_counter() - audio_started) * 1000, 2)

        return PreparedItem(
            record=record,
            resolved_video_path=video_path,
            image_root_rel=image_root_rel,
            final_root_abs=final_root_abs,
            temp_root=temp_root,
            frame_files=frame_files,
            audio_input=audio_input,
            cleanup_paths=cleanup,
            metrics=metrics,
            frame_backend=frame_backend,
        )
    except Exception:
        cleanup_paths([temp_root] + cleanup[1:])
        raise


def build_success_event(profile: PlatformProfile, prepared: PreparedItem, transcript: str, transcriber: Transcriber, transcribe_ms: float, total_ms: float) -> dict:
    return {
        "ts": utc_now_iso(),
        "platform": profile.name,
        "id": str(prepared.record.get("id", "")).strip(),
        "label": prepared.record.get("label", ""),
        "status": "success",
        "backend": transcriber.backend,
        "device": transcriber.device,
        "frame_backend": prepared.frame_backend,
        "frame_count": len(prepared.frame_files),
        "transcript_chars": len(transcript),
        "probe_ms": prepared.metrics["probe_ms"],
        "frame_ms": prepared.metrics["frame_ms"],
        "audio_ms": prepared.metrics["audio_ms"],
        "transcribe_ms": round(transcribe_ms, 2),
        "total_ms": round(total_ms, 2),
    }


def build_failure_event(profile: PlatformProfile, failure: FailureItem, backend: str, device: str) -> dict:
    metrics = failure.metrics
    total_ms = sum(float(metrics.get(key, 0) or 0) for key in ("probe_ms", "frame_ms", "audio_ms", "transcribe_ms"))
    return {
        "ts": utc_now_iso(),
        "platform": profile.name,
        "id": str(failure.record.get("id", "")).strip(),
        "label": failure.record.get("label", ""),
        "status": "failed",
        "backend": backend,
        "device": device,
        "reason_code": failure.reason_code,
        "reason": failure.message,
        "frame_count": 0,
        "transcript_chars": 0,
        "probe_ms": metrics.get("probe_ms", 0),
        "frame_ms": metrics.get("frame_ms", 0),
        "audio_ms": metrics.get("audio_ms", 0),
        "transcribe_ms": metrics.get("transcribe_ms", 0),
        "total_ms": round(total_ms, 2),
    }


def extractor_worker(profile: PlatformProfile, source: PendingSource, result_queue: Queue, args, tool_paths: ToolPaths):
    while True:
        record = source.next_record()
        if record is None:
            result_queue.put(WorkerDone(threading.current_thread().name))
            return

        try:
            prepared = prepare_record(profile, record, args, tool_paths)
            result_queue.put(prepared)
        except Step4Error as exc:
            result_queue.put(
                FailureItem(
                    record=record,
                    resolved_video_path=str(resolve_video_path(profile, record) or ""),
                    reason_code=exc.code,
                    message=exc.message,
                    metrics={"probe_ms": 0, "frame_ms": 0, "audio_ms": 0, "transcribe_ms": 0},
                )
            )
        except KeyboardInterrupt:
            result_queue.put(
                FailureItem(
                    record=record,
                    resolved_video_path=str(resolve_video_path(profile, record) or ""),
                    reason_code="keyboard_interrupted",
                    message="keyboard interrupted during extract stage",
                    metrics={"probe_ms": 0, "frame_ms": 0, "audio_ms": 0, "transcribe_ms": 0},
                )
            )
            result_queue.put(WorkerDone(threading.current_thread().name))
            return
        except Exception as exc:
            result_queue.put(
                FailureItem(
                    record=record,
                    resolved_video_path=str(resolve_video_path(profile, record) or ""),
                    reason_code="frame_extract_failed",
                    message=str(exc),
                    metrics={"probe_ms": 0, "frame_ms": 0, "audio_ms": 0, "transcribe_ms": 0},
                )
            )


def run_platform(profile: PlatformProfile, args) -> None:
    pending, output_map, failure_map = resolve_candidates(profile, args)
    label_counts = summarize_labels(pending)
    print(
        f"[{profile.name}] pending={len(pending)} | sample_seconds={args.sample_seconds} | "
        f"extract_workers={args.max_extract_workers} | retry_failed={args.retry_failed} | "
        f"reverse={args.reverse} | scratch={bool(args.scratch_root)}"
    )
    for label, count in sorted(label_counts.items()):
        print(f"  - {label}: {count}")
    if not pending:
        print(f"[{profile.name}] nothing to do")
        return

    tool_paths = detect_tool_paths()
    transcriber = Transcriber(DEFAULT_WHISPER_MODEL)
    print(
        f"[{profile.name}] transcriber={transcriber.backend} | device={transcriber.device} | "
        f"ffmpeg={tool_paths.ffmpeg or 'missing'} | ffprobe={tool_paths.ffprobe or 'missing'}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = (Path(args.log_dir) if args.log_dir else DEFAULT_LOG_DIR) / f"{profile.name}_{timestamp}.jsonl"
    jsonl = JsonlLogger(jsonl_path)

    interrupt = InterruptController()
    signal.signal(signal.SIGINT, interrupt.handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, interrupt.handle_signal)

    stats = {
        "total": len(pending),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "extract_inflight": 0,
        "transcribe_inflight": 0,
    }
    stats_lock = threading.Lock()
    reporter_stop = threading.Event()
    reporter = ProgressReporter(profile.name, stats, stats_lock, reporter_stop, args.status_every)
    reporter.start()

    source = PendingSource(pending, interrupt.stop_requested)
    result_queue: Queue = Queue(maxsize=max(args.max_extract_workers * 2, 2))
    workers = []
    for index in range(max(args.max_extract_workers, 1)):
        thread = threading.Thread(
            target=extractor_worker,
            name=f"{profile.name}-extract-{index + 1}",
            args=(profile, source, result_queue, args, tool_paths),
            daemon=True,
        )
        thread.start()
        workers.append(thread)

    completed_workers = 0
    try:
        while completed_workers < len(workers) or not result_queue.empty():
            try:
                item = result_queue.get(timeout=0.5)
            except Empty:
                continue

            if isinstance(item, WorkerDone):
                completed_workers += 1
                continue

            if isinstance(item, FailureItem):
                if not args.scratch_root:
                    record_failure(
                        failure_map,
                        profile.failed_json,
                        item.record,
                        item.resolved_video_path,
                        item.reason_code,
                        item.message,
                    )
                jsonl.write(build_failure_event(profile, item, transcriber.backend, transcriber.device))
                with stats_lock:
                    stats["processed"] += 1
                    stats["failed"] += 1
                print(f"[{profile.name}] failed id={item.record.get('id')} | {item.reason_code} | {item.message}")
                cleanup_paths(item.cleanup_paths)
                continue

            prepared = item
            start_transcribe = time.perf_counter()
            try:
                with stats_lock:
                    stats["transcribe_inflight"] = 1
                transcript = transcriber.transcribe(prepared.audio_input).strip()
                transcribe_ms = (time.perf_counter() - start_transcribe) * 1000
                if not transcript:
                    raise Step4Error("empty_transcription", "empty transcription")

                transcription_path = prepared.temp_root / "transcription.txt"
                transcription_path.write_text(transcript, encoding="utf-8")
                commit_output_tree(prepared.temp_root, prepared.final_root_abs)

                final_frames = list_frame_files(prepared.final_root_abs / "frames")
                if not final_frames or not (prepared.final_root_abs / "transcription.txt").is_file():
                    raise Step4Error("frame_extract_failed", "committed output missing frames or transcription")

                if not args.scratch_root:
                    record_id = str(prepared.record.get("id", "")).strip()
                    output_map[record_id] = build_output_record(
                        profile,
                        prepared.record,
                        prepared.image_root_rel,
                        prepared.resolved_video_path,
                        final_frames,
                        transcript,
                    )
                    dump_json_atomic(profile.chouzhen_json, sort_records_by_id(list(output_map.values())))
                    remove_failure(failure_map, profile.failed_json, record_id)

                total_ms = prepared.metrics["probe_ms"] + prepared.metrics["frame_ms"] + prepared.metrics["audio_ms"] + transcribe_ms
                jsonl.write(build_success_event(profile, prepared, transcript, transcriber, transcribe_ms, total_ms))
                with stats_lock:
                    stats["processed"] += 1
                    stats["success"] += 1
                print(
                    f"[{profile.name}] success id={prepared.record.get('id')} | "
                    f"frames={len(final_frames)} | chars={len(transcript)} | "
                    f"frame_backend={prepared.frame_backend}"
                )
            except Step4Error as exc:
                failure = FailureItem(
                    record=prepared.record,
                    resolved_video_path=str(prepared.resolved_video_path),
                    reason_code=exc.code,
                    message=exc.message,
                    metrics={**prepared.metrics, "transcribe_ms": round((time.perf_counter() - start_transcribe) * 1000, 2)},
                    temp_root=prepared.temp_root,
                    cleanup_paths=prepared.cleanup_paths,
                )
                if not args.scratch_root:
                    record_failure(
                        failure_map,
                        profile.failed_json,
                        failure.record,
                        failure.resolved_video_path,
                        failure.reason_code,
                        failure.message,
                    )
                jsonl.write(build_failure_event(profile, failure, transcriber.backend, transcriber.device))
                with stats_lock:
                    stats["processed"] += 1
                    stats["failed"] += 1
                print(f"[{profile.name}] failed id={failure.record.get('id')} | {failure.reason_code} | {failure.message}")
                cleanup_paths([prepared.temp_root] + prepared.cleanup_paths[1:])
            except Exception as exc:
                failure = FailureItem(
                    record=prepared.record,
                    resolved_video_path=str(prepared.resolved_video_path),
                    reason_code="transcribe_failed",
                    message=str(exc),
                    metrics={**prepared.metrics, "transcribe_ms": round((time.perf_counter() - start_transcribe) * 1000, 2)},
                    temp_root=prepared.temp_root,
                    cleanup_paths=prepared.cleanup_paths,
                )
                if not args.scratch_root:
                    record_failure(
                        failure_map,
                        profile.failed_json,
                        failure.record,
                        failure.resolved_video_path,
                        failure.reason_code,
                        failure.message,
                    )
                jsonl.write(build_failure_event(profile, failure, transcriber.backend, transcriber.device))
                with stats_lock:
                    stats["processed"] += 1
                    stats["failed"] += 1
                print(f"[{profile.name}] failed id={failure.record.get('id')} | {failure.reason_code} | {failure.message}")
                cleanup_paths([prepared.temp_root] + prepared.cleanup_paths[1:])
            finally:
                with stats_lock:
                    stats["transcribe_inflight"] = 0
                for path in prepared.cleanup_paths:
                    if path.is_file():
                        path.unlink(missing_ok=True)
    finally:
        reporter_stop.set()
        reporter.join(timeout=2)
        for worker in workers:
            worker.join(timeout=2)

    with stats_lock:
        snapshot = dict(stats)
    print(
        f"[{profile.name}] completed | processed={snapshot['processed']}/{snapshot['total']} "
        f"success={snapshot['success']} failed={snapshot['failed']} | jsonl={jsonl_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Local-only Step 4 v2 runner.")
    parser.add_argument("--platform", choices=["douyin", "youtube", "both"], default="both")
    parser.add_argument("--max-extract-workers", type=int, default=DEFAULT_EXTRACT_WORKERS)
    parser.add_argument("--sample-seconds", type=float, default=DEFAULT_SAMPLE_SECONDS)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--scratch-root")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--status-every", type=int, default=DEFAULT_STATUS_SECONDS)
    return parser.parse_args()


def main():
    args = parse_args()
    profiles = build_profiles()
    order = ["youtube", "douyin"] if args.platform == "both" else [args.platform]
    for platform in order:
        run_platform(profiles[platform], args)


if __name__ == "__main__":
    main()
