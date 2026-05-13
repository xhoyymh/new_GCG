import argparse
import importlib.util
import json
import os
import pathlib
import tempfile
from contextlib import contextmanager


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CODE_ROOT = pathlib.Path(__file__).resolve().parent
DOUYIN_SCRIPT = CODE_ROOT / "douyin" / "douyin_dataset_all_in_one_ollama.py"
YOUTUBE_SCRIPT = CODE_ROOT / "youtube" / "youtube_dataset_all_in_one_ollama.py"


def normalize_ollama_host():
    raw_host = (os.environ.get("OLLAMA_HOST") or "").strip()
    if not raw_host:
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
        return
    if "://" not in raw_host:
        raw_host = f"http://{raw_host}"
    if "0.0.0.0" in raw_host:
        raw_host = raw_host.replace("0.0.0.0", "127.0.0.1")
    scheme, rest = raw_host.split("://", 1)
    if "/" not in rest and ":" not in rest:
        raw_host = f"{scheme}://{rest}:11434"
    os.environ["OLLAMA_HOST"] = raw_host


normalize_ollama_host()


def load_module(module_name: str, script_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: pathlib.Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: pathlib.Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@contextmanager
def override_attrs(module, mapping: dict):
    old = {key: getattr(module, key) for key in mapping}
    try:
        for key, value in mapping.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(module, key, value)


def resolve_description_json_attr(module) -> str:
    for name in ("VIDEO_DESCRIPTION_JSON", "DESCRIPTION_JSON"):
        if hasattr(module, name):
            return name
    raise AttributeError("description json attribute not found")


def resolve_step4(module):
    if hasattr(module, "step4_chouzhen"):
        return module.step4_chouzhen
    if hasattr(module, "step4_extract_frames_transcribe"):
        return module.step4_extract_frames_transcribe
    raise AttributeError("step4 function not found")


def pick_sample_record(module, explicit_id: str | None = None) -> dict:
    intro_path = pathlib.Path(module.VIDEO_INTRO_JSON)
    chouzhen_path = pathlib.Path(module.CHOUZHEN_JSON)
    desc_attr = resolve_description_json_attr(module)
    desc_path = pathlib.Path(getattr(module, desc_attr))

    intro_records = load_json(intro_path, [])
    chouzhen_records = load_json(chouzhen_path, [])
    desc_records = load_json(desc_path, [])
    chouzhen_ids = {str(item.get("id")) for item in chouzhen_records if isinstance(item, dict)}
    desc_ids = {str(item.get("id")) for item in desc_records if isinstance(item, dict)}

    for item in intro_records:
        record_id = str(item.get("id", "")).strip()
        if explicit_id and record_id != explicit_id:
            continue
        if not record_id:
            continue
        if not explicit_id and record_id in desc_ids:
            continue
        if not explicit_id and record_id in chouzhen_ids:
            continue
        video_path = pathlib.Path(REPO_ROOT / item.get("video_path", ""))
        if video_path.is_file():
            return item
    raise RuntimeError("no eligible sample record found")


def run_single_platform(platform: str, explicit_id: str | None = None) -> dict:
    if platform == "douyin":
        module = load_module("douyin_step45_module", DOUYIN_SCRIPT)
    else:
        module = load_module("youtube_step45_module", YOUTUBE_SCRIPT)

    record = pick_sample_record(module, explicit_id=explicit_id)
    record_id = str(record["id"])
    desc_attr = resolve_description_json_attr(module)
    step4 = resolve_step4(module)
    step5 = module.step5_generate_descriptions

    actual_chouzhen_path = pathlib.Path(module.CHOUZHEN_JSON)
    actual_desc_path = pathlib.Path(getattr(module, desc_attr))

    with tempfile.TemporaryDirectory(prefix=f"{platform}_step45_") as tmp_dir:
        tmp_root = pathlib.Path(tmp_dir)
        tmp_intro = tmp_root / f"{platform}_intro.json"
        dump_json(tmp_intro, [record])

        with override_attrs(
            module,
            {
                "OLLAMA_MODEL": "qwen3.5:9b",
                "VIDEO_INTRO_JSON": str(tmp_intro),
                "CHOUZHEN_JSON": str(actual_chouzhen_path),
                desc_attr: str(actual_desc_path),
            },
        ):
            step4()

        actual_chouzhen_records = load_json(actual_chouzhen_path, [])
        target_chouzhen = [item for item in actual_chouzhen_records if str(item.get("id", "")).strip() == record_id]
        if not target_chouzhen:
            raise RuntimeError(f"step4 did not write target record {record_id} for {platform}")

        tmp_chouzhen = tmp_root / f"{platform}_chouzhen.json"
        dump_json(tmp_chouzhen, target_chouzhen)

        with override_attrs(
            module,
            {
                "OLLAMA_MODEL": "qwen3.5:9b",
                "CHOUZHEN_JSON": str(tmp_chouzhen),
                desc_attr: str(actual_desc_path),
            },
        ):
            step5()

    desc_records = load_json(actual_desc_path, [])
    target_desc = next((item for item in desc_records if str(item.get("id", "")).strip() == record_id), None)
    if not target_desc or not str(target_desc.get("video_description", "")).strip():
        raise RuntimeError(f"step5 did not produce a non-empty description for {platform} id={record_id}")

    return {
        "platform": platform,
        "id": record_id,
        "label": record.get("label", ""),
        "video_path": record.get("video_path", ""),
        "description_chars": len(str(target_desc.get("video_description", "")).strip()),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Step 4/5 on a single Douyin or YouTube record.")
    parser.add_argument("--platform", choices=["douyin", "youtube", "both"], required=True)
    parser.add_argument("--id", default="", help="Optional explicit id for single-platform validation.")
    args = parser.parse_args()

    targets = ["douyin", "youtube"] if args.platform == "both" else [args.platform]
    for platform in targets:
        result = run_single_platform(platform, explicit_id=args.id or None)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
