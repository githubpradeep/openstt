from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _conv_out_length(lengths: torch.Tensor, stride: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1) -> torch.Tensor:
    return ((lengths + (2 * padding) - (dilation * (kernel_size - 1)) - 1) // stride) + 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_stride: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, time_stride),
            padding=(1, 1),
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.time_stride = time_stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class SpeechToTextModel(nn.Module):
    def __init__(
        self,
        n_mels: int,
        vocab_size: int,
        cnn_channels: Iterable[int],
        cnn_time_strides: Iterable[int],
        cnn_dropout: float,
        lstm_input_size: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        cnn_channels = list(cnn_channels)
        cnn_time_strides = list(cnn_time_strides)
        if len(cnn_channels) != len(cnn_time_strides):
            raise ValueError("cnn_channels and cnn_time_strides must have the same length")

        blocks: List[nn.Module] = []
        in_channels = 1
        for out_channels, stride in zip(cnn_channels, cnn_time_strides):
            blocks.append(ConvBlock(in_channels, out_channels, stride, cnn_dropout))
            in_channels = out_channels
        self.conv = nn.Sequential(*blocks)
        self.cnn_time_strides = cnn_time_strides
        self.n_mels = n_mels

        conv_out_size = cnn_channels[-1] * n_mels
        self.pre_lstm = nn.Linear(conv_out_size, lstm_input_size)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(lstm_hidden_size * 2, vocab_size)

    def output_lengths(self, feature_lengths: torch.Tensor) -> torch.Tensor:
        lengths = feature_lengths.clone()
        for stride in self.cnn_time_strides:
            lengths = _conv_out_length(lengths, stride=stride)
        return lengths

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = features.unsqueeze(1)
        outputs = self.conv(outputs)
        output_lengths = self.output_lengths(feature_lengths)

        batch_size, channels, freqs, time_steps = outputs.shape
        outputs = outputs.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, channels * freqs)
        outputs = self.pre_lstm(outputs)

        packed = pack_padded_sequence(
            outputs,
            lengths=output_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        logits = self.classifier(outputs)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1), output_lengths
