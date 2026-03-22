from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LJSpeech from Hugging Face and write it to standard LJSpeech folder layout.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="flexthink/ljspeech",
        help="Hugging Face dataset repo id containing metadata.csv and wavs/ directly.",
    )
    parser.add_argument("--output-dir", type=str, default="data/LJSpeech-1.1", help="Destination folder.")
    parser.add_argument("--force", action="store_true", help="Overwrite output dir if it already exists.")
    return parser.parse_args()


def ensure_clean_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wavs").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    metadata_path = output_dir / "metadata.csv"
    wavs_dir = output_dir / "wavs"

    if metadata_path.exists() and any(wavs_dir.glob("*.wav")) and not args.force:
        print(f"dataset already exists at {output_dir}. Use --force to rebuild it.")
        return

    ensure_clean_output(output_dir, force=args.force)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=["metadata.csv", "wavs/*.wav"],
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Download completed but metadata.csv is missing under {output_dir}")
    wav_count = len(list(wavs_dir.glob("*.wav")))
    if wav_count == 0:
        raise FileNotFoundError(f"Download completed but no wav files were found under {wavs_dir}")

    cache_dir = output_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"source repo: {args.repo_id}")
    print(f"saved dataset to {output_dir}")
    print(f"wav files: {wav_count}")
    print(f"metadata: {metadata_path}")
    print(f"wavs: {wavs_dir}")


if __name__ == "__main__":
    main()
