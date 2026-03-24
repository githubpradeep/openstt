from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from audio import AudioPreprocessor, load_audio
from text import CharTokenizer, normalize_text


def _read_jsonl(path: str | Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def _write_jsonl(records: Sequence[Dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _prepared_manifest_paths(dataset_root: str | Path) -> tuple[Path, Path] | None:
    root = Path(dataset_root)
    train_manifest = root / "manifests" / "train.jsonl"
    val_manifest = root / "manifests" / "val.jsonl"
    if train_manifest.exists() and val_manifest.exists():
        return train_manifest, val_manifest
    return None


def build_ljspeech_manifest(dataset_root: str | Path, manifest_path: str | Path) -> List[Dict[str, str]]:
    root = Path(dataset_root)
    metadata_path = root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found under {root}. "
            f"Run `python prepare_data.py --output-dir {root}` first, or point --dataset-root to an existing prepared dataset folder."
        )

    records: List[Dict[str, str]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            utterance_id = parts[0]
            text = parts[2] if len(parts) > 2 and parts[2].strip() else parts[1]
            text = normalize_text(text)
            audio_path = root / "wavs" / f"{utterance_id}.wav"
            if not text or not audio_path.exists():
                continue
            records.append(
                {
                    "id": utterance_id,
                    "audio_path": str(audio_path.resolve()),
                    "text": text,
                }
            )

    if not records:
        raise RuntimeError(f"No usable samples found under {root}")

    _write_jsonl(records, manifest_path)
    return records


def _split_records(records: Sequence[Dict[str, str]], val_ratio: float, seed: int) -> tuple[list[Dict[str, str]], list[Dict[str, str]]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    return train_records, val_records


def _select_subset(records: Sequence[Dict[str, str]], subset_size: int | None, seed: int) -> List[Dict[str, str]]:
    selected = list(records)
    if subset_size is None or subset_size >= len(selected):
        return selected
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected[:subset_size]


def prepare_ljspeech_splits(
    dataset_root: str | Path,
    output_dir: str | Path,
    val_ratio: float,
    seed: int,
    subset_size: int | None,
    val_subset_size: int | None,
    overfit_mode: bool,
    overfit_size: int,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    manifest_dir = output_dir / "manifests"
    prepared_paths = _prepared_manifest_paths(dataset_root)
    if prepared_paths is not None:
        source_train_manifest, source_val_manifest = prepared_paths
        train_records = _read_jsonl(source_train_manifest)
        val_records = _read_jsonl(source_val_manifest)
    else:
        all_manifest = manifest_dir / "all.jsonl"
        records = build_ljspeech_manifest(dataset_root, all_manifest)
        train_records, val_records = _split_records(records, val_ratio=val_ratio, seed=seed)

    if overfit_mode:
        overfit_records = _select_subset(train_records, overfit_size, seed)
        train_records = overfit_records
        val_records = overfit_records
    else:
        train_records = _select_subset(train_records, subset_size, seed)
        val_records = _select_subset(val_records, val_subset_size, seed + 1)

    train_manifest = manifest_dir / "train.jsonl"
    val_manifest = manifest_dir / "val.jsonl"
    _write_jsonl(train_records, train_manifest)
    _write_jsonl(val_records, val_manifest)

    return {
        "train_manifest": str(train_manifest.resolve()),
        "val_manifest": str(val_manifest.resolve()),
        "train_records": train_records,
        "val_records": val_records,
    }


class LJSpeechDataset(Dataset):
    def __init__(self, manifest_path: str | Path, tokenizer: CharTokenizer, sample_rate: int) -> None:
        self.records = _read_jsonl(manifest_path)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        text_ids = self.tokenizer.text_to_ids(record["text"])
        if not text_ids:
            raise ValueError(f"Empty token sequence for sample {record['id']}")
        waveform = load_audio(record["audio_path"], self.sample_rate)
        return {
            "id": record["id"],
            "audio_path": record["audio_path"],
            "waveform": waveform,
            "text": record["text"],
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
        }


class SpeechCollator:
    def __init__(self, preprocessor: AudioPreprocessor) -> None:
        self.preprocessor = preprocessor

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        waveforms = [sample["waveform"] for sample in batch]
        waveform_lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
        padded_waveforms = pad_sequence(waveforms, batch_first=True)
        features, feature_lengths = self.preprocessor(padded_waveforms, waveform_lengths)

        targets = [sample["text_ids"] for sample in batch]
        target_lengths = torch.tensor([target.numel() for target in targets], dtype=torch.long)
        flat_targets = torch.cat(targets, dim=0)

        return {
            "ids": [sample["id"] for sample in batch],
            "audio_paths": [sample["audio_path"] for sample in batch],
            "texts": [sample["text"] for sample in batch],
            "waveforms": padded_waveforms,
            "waveform_lengths": waveform_lengths,
            "features": features,
            "feature_lengths": feature_lengths,
            "targets": flat_targets,
            "target_lengths": target_lengths,
        }
