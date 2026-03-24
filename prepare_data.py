from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from text import normalize_text


SOURCES = {
    "ljspeech": {
        "repo_id": "flexthink/ljspeech",
    },
    "blog_timit": {
        "repo_id": "m-aliabbas/idrak_timit_subsample1",
        "splits": ["train", "test"],
        "text_key": "transcription",
    },
    "librispeech": {
        "archives": {
            "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
            "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
            "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        },
        "default_splits": ["train-clean-100", "dev-clean", "test-clean"],
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
    parser.add_argument(
        "--librispeech-splits",
        type=str,
        default=None,
        help="Comma-separated LibriSpeech splits to download, for example train-clean-100,dev-clean,test-clean.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output dir if it already exists.")
    return parser.parse_args()


def ensure_clean_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wavs").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


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
    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []

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

            normalized = normalize_text(text)
            metadata_lines.append(f"{utterance_id}|{text}|{normalized}")
            record = {
                "id": utterance_id,
                "audio_path": str(wav_path.resolve()),
                "text": normalized,
            }
            if split_name == "train":
                train_records.append(record)
            elif split_name in {"validation", "val"}:
                val_records.append(record)
            else:
                test_records.append(record)

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for line in metadata_lines:
            handle.write(line + "\n")

    if train_records:
        write_jsonl(train_records, output_dir / "manifests" / "train.jsonl")
    if val_records:
        write_jsonl(val_records, output_dir / "manifests" / "val.jsonl")
    elif test_records:
        write_jsonl(test_records, output_dir / "manifests" / "val.jsonl")
    if test_records:
        write_jsonl(test_records, output_dir / "manifests" / "test.jsonl")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_tar_gz(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)


def collect_librispeech_records(corpus_root: Path, split_name: str) -> list[dict]:
    split_root = corpus_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"Expected LibriSpeech split directory at {split_root}")

    records: list[dict] = []
    for transcript_path in sorted(split_root.rglob("*.trans.txt")):
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                utterance_id, text = line.rstrip("\n").split(" ", maxsplit=1)
                audio_path = transcript_path.parent / f"{utterance_id}.flac"
                normalized = normalize_text(text)
                if not normalized or not audio_path.exists():
                    continue
                records.append(
                    {
                        "id": utterance_id,
                        "audio_path": str(audio_path.resolve()),
                        "text": normalized,
                    }
                )
    return records


def prepare_librispeech(output_dir: Path, selected_splits: list[str]) -> None:
    archives = SOURCES["librispeech"]["archives"]
    downloads_dir = output_dir / "downloads"

    for split_name in selected_splits:
        if split_name not in archives:
            raise ValueError(f"Unsupported LibriSpeech split: {split_name}")
        archive_url = archives[split_name]
        archive_path = downloads_dir / Path(archive_url).name
        if not archive_path.exists():
            print(f"downloading {archive_url}")
            download_file(archive_url, archive_path)
        print(f"extracting {archive_path}")
        extract_tar_gz(archive_path, output_dir)

    corpus_root = output_dir / "LibriSpeech"
    manifests_dir = output_dir / "manifests"
    all_records: list[str] = []
    split_to_manifest_name = {
        "train-clean-100": "train",
        "train-clean-360": "train",
        "train-other-500": "train",
        "dev-clean": "val",
        "dev-other": "val",
        "test-clean": "test",
        "test-other": "test",
    }
    grouped_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for split_name in selected_splits:
        records = collect_librispeech_records(corpus_root, split_name)
        grouped_records[split_to_manifest_name[split_name]].extend(records)
        for record in records:
            all_records.append(f"{record['id']}|{record['text']}|{record['text']}")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for line in all_records:
            handle.write(line + "\n")

    for manifest_name, records in grouped_records.items():
        if records:
            write_jsonl(records, manifests_dir / f"{manifest_name}.jsonl")


def main() -> None:
    args = parse_args()
    source = SOURCES[args.source]
    repo_id = args.repo_id or source.get("repo_id")
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
    elif args.source == "librispeech":
        if args.librispeech_splits is None:
            selected_splits = list(source["default_splits"])
        else:
            selected_splits = [item.strip() for item in args.librispeech_splits.split(",") if item.strip()]
        prepare_librispeech(output_dir=output_dir, selected_splits=selected_splits)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Download completed but metadata.csv is missing under {output_dir}")
    wav_count = len(list(wavs_dir.glob("*.wav")))
    if wav_count == 0:
        train_manifest = output_dir / "manifests" / "train.jsonl"
        if train_manifest.exists():
            with train_manifest.open("r", encoding="utf-8") as handle:
                wav_count = sum(1 for _ in handle)
        if wav_count == 0:
            raise FileNotFoundError(f"Download completed but no audio files were found under {output_dir}")

    cache_dir = output_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"source: {args.source}")
    if repo_id is not None:
        print(f"source repo: {repo_id}")
    print(f"saved dataset to {output_dir}")
    print(f"wav files: {wav_count}")
    print(f"metadata: {metadata_path}")
    print(f"wavs: {wavs_dir}")


if __name__ == "__main__":
    main()
