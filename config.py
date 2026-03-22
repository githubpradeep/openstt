from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "ljspeech_stt",
        "output_dir": "runs/ljspeech_stt",
    },
    "dataset": {
        "root": "data/LJSpeech-1.1",
        "val_ratio": 0.05,
        "seed": 1337,
        "subset_size": None,
        "val_subset_size": 128,
        "overfit_mode": False,
        "overfit_size": 20,
    },
    "audio": {
        "sample_rate": 16000,
        "n_fft": 400,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 80,
        "f_min": 0,
        "f_max": 8000,
        "log_epsilon": 1.0e-5,
        "normalize": True,
        "time_mask_param": 0,
        "freq_mask_param": 0,
    },
    "model": {
        "cnn_channels": [32, 64],
        "cnn_time_strides": [2, 2],
        "cnn_dropout": 0.1,
        "lstm_input_size": 256,
        "lstm_hidden_size": 256,
        "lstm_layers": 3,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 16,
        "num_workers": 2,
        "epochs": 15,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "grad_clip": 5.0,
        "amp": True,
        "device": "auto",
        "log_interval": 20,
        "val_interval": 1,
        "save_every_epoch": True,
        "resume_from": None,
        "cudnn_benchmark": True,
        "scheduler": {
            "name": None,
            "factor": 0.5,
            "patience": 3,
            "min_lr": 1.0e-5,
        },
    },
    "debug": {
        "print_batch_shapes": True,
        "print_spectrogram_stats": True,
        "verify_output_lengths": True,
        "run_forward_pass": True,
        "run_backward_pass": True,
        "print_predictions_before_training": True,
        "sample_predictions": 3,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = deep_update(DEFAULT_CONFIG, loaded)
    if overrides:
        config = deep_update(config, overrides)
    return config


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
