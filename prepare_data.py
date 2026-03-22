from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LJSpeech from Hugging Face and write it to standard LJSpeech folder layout.")
    parser.add_argument("--repo-id", type=str, default="keithito/lj_speech", help="Hugging Face dataset repo id.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to download.")
    parser.add_argument("--output-dir", type=str, default="data/LJSpeech-1.1", help="Destination folder.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for a partial download.")
    parser.add_argument("--force", action="store_true", help="Overwrite output dir if it already exists.")
    return parser.parse_args()


def ensure_clean_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wavs").mkdir(parents=True, exist_ok=True)


def save_audio_from_array(audio: dict, destination: Path) -> None:
    array = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
    torchaudio.save(
        str(destination),
        array,
        sample_rate=int(audio["sampling_rate"]),
        encoding="PCM_S",
        bits_per_sample=16,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    metadata_path = output_dir / "metadata.csv"
    wavs_dir = output_dir / "wavs"

    if metadata_path.exists() and any(wavs_dir.glob("*.wav")) and not args.force:
        print(f"dataset already exists at {output_dir}. Use --force to rebuild it.")
        return

    ensure_clean_output(output_dir, force=args.force)

    dataset = load_dataset(args.repo_id, split=args.split, trust_remote_code=True)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    records = []
    for example in tqdm(dataset, desc="downloading ljspeech"):
        utterance_id = str(example["id"])
        text = str(example.get("text", "")).strip()
        normalized_text = str(example.get("normalized_text", text)).strip()
        source_path = example.get("file")
        destination = wavs_dir / f"{utterance_id}.wav"

        if source_path and Path(source_path).exists():
            shutil.copy2(source_path, destination)
        else:
            audio = example.get("audio")
            if audio is None:
                raise RuntimeError(f"Sample {utterance_id} has neither a local file path nor decoded audio.")
            save_audio_from_array(audio, destination)

        records.append(f"{utterance_id}|{text}|{normalized_text}")

    with metadata_path.open("w", encoding="utf-8") as handle:
        for line in records:
            handle.write(line + "\n")

    print(f"saved {len(records)} samples to {output_dir}")
    print(f"metadata: {metadata_path}")
    print(f"wavs: {wavs_dir}")


if __name__ == "__main__":
    main()
