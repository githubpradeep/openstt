from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm


SOURCES = {
    "ljspeech": {
        "repo_id": "flexthink/ljspeech",
    },
    "blog_timit": {
        "repo_id": "m-aliabbas/idrak_timit_subsample1",
        "splits": ["train", "test"],
        "text_key": "transcription",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a supported speech dataset from Hugging Face and write it to the standard metadata.csv + wavs/ layout."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=sorted(SOURCES.keys()),
        default="ljspeech",
        help="Named dataset source to prepare.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Optional override for the Hugging Face dataset repo id.",
    )
    parser.add_argument("--output-dir", type=str, default="data/LJSpeech-1.1", help="Destination folder.")
    parser.add_argument("--force", action="store_true", help="Overwrite output dir if it already exists.")
    return parser.parse_args()


def ensure_clean_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wavs").mkdir(parents=True, exist_ok=True)


def prepare_standard_wav_repo(repo_id: str, output_dir: Path) -> None:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=["metadata.csv", "wavs/*.wav"],
    )


def prepare_parquet_audio_repo(repo_id: str, output_dir: Path, splits: list[str], text_key: str) -> None:
    wavs_dir = output_dir / "wavs"
    metadata_lines: list[str] = []

    for split_name in splits:
        dataset = load_dataset(repo_id, split=split_name)
        dataset = dataset.cast_column("audio", Audio(decode=False))

        for index, example in enumerate(tqdm(dataset, desc=f"exporting {split_name}")):
            payload = example["audio"]
            text = str(example[text_key]).strip()
            utterance_id = f"{split_name}_{index:06d}"
            wav_path = wavs_dir / f"{utterance_id}.wav"

            if payload.get("bytes") is not None:
                wav_path.write_bytes(payload["bytes"])
            elif payload.get("path") is not None and Path(payload["path"]).exists():
                shutil.copy2(payload["path"], wav_path)
            else:
                raise FileNotFoundError(
                    f"Could not materialize audio for {utterance_id}. "
                    f"Expected embedded bytes or an accessible local audio path."
                )

            metadata_lines.append(f"{utterance_id}|{text}|{text}")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for line in metadata_lines:
            handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    source = SOURCES[args.source]
    repo_id = args.repo_id or source["repo_id"]
    output_dir = Path(args.output_dir).resolve()
    metadata_path = output_dir / "metadata.csv"
    wavs_dir = output_dir / "wavs"

    if metadata_path.exists() and any(wavs_dir.glob("*.wav")) and not args.force:
        print(f"dataset already exists at {output_dir}. Use --force to rebuild it.")
        return

    ensure_clean_output(output_dir, force=args.force)

    if args.source == "ljspeech":
        prepare_standard_wav_repo(repo_id=repo_id, output_dir=output_dir)
    elif args.source == "blog_timit":
        prepare_parquet_audio_repo(
            repo_id=repo_id,
            output_dir=output_dir,
            splits=source["splits"],
            text_key=source["text_key"],
        )
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Download completed but metadata.csv is missing under {output_dir}")
    wav_count = len(list(wavs_dir.glob("*.wav")))
    if wav_count == 0:
        raise FileNotFoundError(f"Download completed but no wav files were found under {wavs_dir}")

    cache_dir = output_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"source: {args.source}")
    print(f"source repo: {repo_id}")
    print(f"saved dataset to {output_dir}")
    print(f"wav files: {wav_count}")
    print(f"metadata: {metadata_path}")
    print(f"wavs: {wavs_dir}")


if __name__ == "__main__":
    main()
