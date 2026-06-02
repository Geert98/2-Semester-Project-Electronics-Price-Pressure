from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ai_lab.export_news import export_news_for_ai_lab
from src.utils import load_config, setup_env


AI_LAB_FILES = [
    "README.md",
    "run_ai_lab.sh",
    "enrich_articles.py",
]

AI_LAB_RUNTIME_DIRS = [
    "inputs",
    "logs/out",
    "logs/err",
    "results",
]


def _reset_bundle_dir(bundle_dir: Path) -> None:
    if not bundle_dir.exists():
        return
    for child in bundle_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _copy_ai_lab_files(bundle_dir: Path, model: str, gpus: int, max_model_len: int) -> None:
    source_dir = Path("ai_lab")
    for filename in AI_LAB_FILES:
        source = source_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing AI-LAB file: {source}")
        destination = bundle_dir / filename
        shutil.copy2(source, destination)

    run_script = bundle_dir / "run_ai_lab.sh"
    text = run_script.read_text(encoding="utf-8")
    text = text.replace('AI_LAB_MODEL="${AI_LAB_MODEL:-Qwen/Qwen2.5-7B-Instruct}"', f'AI_LAB_MODEL="${{AI_LAB_MODEL:-{model}}}"')
    text = text.replace('AI_LAB_GPUS="${AI_LAB_GPUS:-1}"', f'AI_LAB_GPUS="${{AI_LAB_GPUS:-{gpus}}}"')
    text = text.replace('AI_LAB_MAX_MODEL_LEN="${AI_LAB_MAX_MODEL_LEN:-8192}"', f'AI_LAB_MAX_MODEL_LEN="${{AI_LAB_MAX_MODEL_LEN:-{max_model_len}}}"')
    run_script.write_text(text, encoding="utf-8")
    run_script.chmod(0o755)


def _create_runtime_dirs(bundle_dir: Path) -> None:
    for relative_dir in AI_LAB_RUNTIME_DIRS:
        path = bundle_dir / relative_dir
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch()


def _split_jsonl(input_path: Path, inputs_dir: Path, batch_size: int) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    records = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    batch_files = []
    for batch_index, start in enumerate(range(0, len(records), batch_size)):
        batch_records = records[start : start + batch_size]
        batch_path = inputs_dir / f"news_to_enrich_{batch_index:05d}.jsonl"
        batch_path.write_text("\n".join(batch_records) + "\n", encoding="utf-8")
        batch_files.append(
            {
                "batch_index": batch_index,
                "path": str(batch_path.relative_to(input_path.parent)),
                "article_count": len(batch_records),
            }
        )

    manifest = {
        "article_count": len(records),
        "batch_size": batch_size,
        "batch_count": len(batch_files),
        "input_file": input_path.name,
        "batch_files": batch_files,
    }
    return manifest


def prepare_ai_lab_batch(
    config_path: str = "configs/config.yaml",
    output_path: str | None = None,
    bundle_dir: str | None = None,
    limit: int | None = None,
    max_chars: int | None = None,
    prefilter: bool | None = None,
    batch_size: int | None = None,
    model: str | None = None,
    gpus: int | None = None,
    max_model_len: int | None = None,
) -> Path:
    setup_env()
    config = load_config(config_path)
    export_cfg = config.get("ai_lab_export", {})

    output = output_path or export_cfg.get("output_path", "ai_lab/upload_bundle/news_to_enrich.jsonl")
    bundle = Path(bundle_dir or export_cfg.get("bundle_dir", "ai_lab/upload_bundle"))
    effective_limit = limit if limit is not None else export_cfg.get("limit")
    effective_max_chars = max_chars if max_chars is not None else export_cfg.get("max_chars", 12000)
    effective_prefilter = prefilter if prefilter is not None else export_cfg.get("prefilter", True)
    effective_batch_size = batch_size if batch_size is not None else int(export_cfg.get("batch_size", 100))
    effective_model = model or export_cfg.get("default_model", "Qwen/Qwen2.5-7B-Instruct")
    effective_gpus = gpus if gpus is not None else int(export_cfg.get("default_gpus", 1))
    effective_max_model_len = max_model_len if max_model_len is not None else int(export_cfg.get("default_max_model_len", 8192))

    bundle.mkdir(parents=True, exist_ok=True)
    _reset_bundle_dir(bundle)
    _create_runtime_dirs(bundle)
    export_path = export_news_for_ai_lab(
        config_path=config_path,
        output_path=output,
        limit=effective_limit,
        max_chars=effective_max_chars,
        prefilter=effective_prefilter,
    )

    bundle_export_path = bundle / "news_to_enrich.jsonl"
    if export_path.resolve() != bundle_export_path.resolve():
        shutil.copy2(export_path, bundle_export_path)
    manifest = _split_jsonl(bundle_export_path, bundle / "inputs", effective_batch_size)
    (bundle / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _copy_ai_lab_files(bundle, effective_model, effective_gpus, effective_max_model_len)

    print("\nAI-LAB batch bundle is ready.")
    print(f"Bundle folder: {bundle}")
    print(f"Articles: {manifest['article_count']}")
    print(f"Batches: {manifest['batch_count']} x up to {effective_batch_size} articles")
    print(f"Default model: {effective_model}")
    print(f"Default GPUs: {effective_gpus}")
    print(f"Default max model length: {effective_max_model_len}")
    print("Bundle folders: inputs, logs/out, logs/err, results")
    print("\nUpload the contents of that folder to AI-LAB, then run:")
    print("  sbatch run_ai_lab.sh qwen25-32b")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export news and prepare an AI-LAB upload bundle.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output")
    parser.add_argument("--bundle-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--no-prefilter", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--all", action="store_true", help="Export all pending articles, ignoring configured limit.")
    parser.add_argument("--model")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--max-model-len", type=int)
    args = parser.parse_args()

    prepare_ai_lab_batch(
        config_path=args.config,
        output_path=args.output,
        bundle_dir=args.bundle_dir,
        limit=None if args.all else args.limit,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        prefilter=False if args.no_prefilter else None,
        model=args.model,
        gpus=args.gpus,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
