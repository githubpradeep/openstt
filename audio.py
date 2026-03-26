from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(audio_path: str | Path, sample_rate: int) -> torch.Tensor:
    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except RuntimeError:
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(np.asarray(audio).T)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0)


class AudioPreprocessor:
    def __init__(self, config: dict, augment: bool = False) -> None:
        self.sample_rate = int(config["sample_rate"])
        self.hop_length = int(config["hop_length"])
        self.n_mels = int(config["n_mels"])
        self.log_epsilon = float(config["log_epsilon"])
        self.normalize = bool(config["normalize"])
        self.augment = augment
        self.time_mask_param = int(config.get("time_mask_param", 0) or 0)
        self.freq_mask_param = int(config.get("freq_mask_param", 0) or 0)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(config["n_fft"]),
            win_length=int(config["win_length"]),
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=float(config.get("f_min", 0)),
            f_max=float(config.get("f_max", self.sample_rate // 2)),
            center=True,
            power=2.0,
        )
        self.time_mask = (
            torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask_param)
            if self.time_mask_param > 0
            else None
        )
        self.freq_mask = (
            torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param)
            if self.freq_mask_param > 0
            else None
        )

    def samples_to_frames(self, lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(lengths, self.hop_length, rounding_mode="floor") + 1

    def _normalize(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> torch.Tensor:
        normalized = features.clone()
        for index in range(features.size(0)):
            valid_frames = int(feature_lengths[index].item())
            sample = normalized[index, :, :valid_frames]
            mean = sample.mean()
            std = sample.std(unbiased=False).clamp_min(1.0e-5)
            normalized[index] = (normalized[index] - mean) / std
        return normalized

    def _augment(self, features: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return features
        output = features.clone()
        for index in range(output.size(0)):
            sample = output[index].unsqueeze(0)
            if self.freq_mask is not None:
                sample = self.freq_mask(sample)
            if self.time_mask is not None:
                sample = self.time_mask(sample)
            output[index] = sample.squeeze(0)
        return output

    def __call__(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.mel(waveforms)
        features = torch.log(features.clamp_min(self.log_epsilon))
        feature_lengths = self.samples_to_frames(waveform_lengths)
        if self.normalize:
            features = self._normalize(features, feature_lengths)
        features = self._augment(features)
        return features, feature_lengths


def describe_feature_batch(features: torch.Tensor, lengths: torch.Tensor) -> str:
    return (
        f"features={tuple(features.shape)} "
        f"feature_lengths[min={int(lengths.min())}, max={int(lengths.max())}] "
        f"mean={features.mean().item():.4f} std={features.std(unbiased=False).item():.4f}"
    )
