from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def save_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def find_vocab_path(checkpoint_path: str | Path, explicit_vocab_path: str | Path | None = None) -> Path | None:
    if explicit_vocab_path is not None:
        return Path(explicit_vocab_path)
    checkpoint_dir = Path(checkpoint_path).resolve().parent
    vocab_path = checkpoint_dir / "vocab.json"
    return vocab_path if vocab_path.exists() else None


def list_audio_files(audio_dir: str | Path) -> List[Path]:
    audio_dir = Path(audio_dir)
    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files found under {audio_dir}")
    return [path.resolve() for path in files]


def format_learning_rate(optimizer: torch.optim.Optimizer) -> str:
    return f"{optimizer.param_groups[0]['lr']:.6f}"


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)
