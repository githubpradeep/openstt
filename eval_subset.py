from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from audio import AudioPreprocessor, load_audio
from metrics import cer_stats, error_rate, wer_stats
from model import SpeechToTextModel
from text import CharTokenizer
from utils import choose_device, find_vocab_path, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained STT checkpoint on a small subset of manifest entries.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to JSONL manifest.")
    parser.add_argument("--num-samples", type=int, default=25, help="Number of samples to evaluate.")
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--vocab", type=str, default=None, help="Optional vocab.json path.")
    return parser.parse_args()


def load_manifest(path: str | Path) -> List[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def load_tokenizer(checkpoint: dict, checkpoint_path: str, explicit_vocab_path: str | None) -> CharTokenizer:
    vocab_path = find_vocab_path(checkpoint_path, explicit_vocab_path)
    if vocab_path is not None and vocab_path.exists():
        return CharTokenizer.from_file(vocab_path)
    if checkpoint.get("token_to_id") is None:
        raise FileNotFoundError("Could not find vocab.json and checkpoint has no token_to_id.")
    return CharTokenizer(token_to_id=checkpoint["token_to_id"])


def build_model(checkpoint: dict, tokenizer: CharTokenizer, device: torch.device) -> SpeechToTextModel:
    config = checkpoint["config"]
    model = SpeechToTextModel(
        n_mels=int(config["audio"]["n_mels"]),
        vocab_size=tokenizer.vocab_size,
        cnn_channels=config["model"]["cnn_channels"],
        cnn_time_strides=config["model"]["cnn_time_strides"],
        cnn_dropout=float(config["model"]["cnn_dropout"]),
        lstm_input_size=int(config["model"]["lstm_input_size"]),
        lstm_hidden_size=int(config["model"]["lstm_hidden_size"]),
        lstm_layers=int(config["model"]["lstm_layers"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def decode_batch(log_probs: torch.Tensor, output_lengths: torch.Tensor, tokenizer: CharTokenizer) -> List[str]:
    token_ids = log_probs.argmax(dim=-1).transpose(0, 1)
    return [
        tokenizer.decode_ctc(token_ids[index, : int(output_lengths[index].item())].tolist())
        for index in range(token_ids.size(0))
    ]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer = load_tokenizer(checkpoint, args.checkpoint, args.vocab)
    model = build_model(checkpoint, tokenizer, device)
    preprocessor = AudioPreprocessor(checkpoint["config"]["audio"], augment=False)

    records = load_manifest(args.manifest)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    records = records[: min(args.num_samples, len(records))]

    predictions: List[str] = []
    references: List[str] = []
    sample_rows: List[tuple[str, str, str]] = []

    for start in range(0, len(records), args.batch_size):
        chunk = records[start : start + args.batch_size]
        waveforms = [load_audio(record["audio_path"], preprocessor.sample_rate) for record in chunk]
        lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
        padded = pad_sequence(waveforms, batch_first=True)
        features, feature_lengths = preprocessor(padded, lengths)
        log_probs, output_lengths = model(features.to(device), feature_lengths.to(device))
        chunk_predictions = decode_batch(log_probs.cpu(), output_lengths.cpu(), tokenizer)

        for record, prediction in zip(chunk, chunk_predictions):
            predictions.append(prediction)
            references.append(record["text"])
            sample_rows.append((record["audio_path"], record["text"], prediction))

    cer_edits, cer_total = cer_stats(predictions, references)
    wer_edits, wer_total = wer_stats(predictions, references)

    print(f"samples={len(references)}")
    print(f"cer={error_rate(cer_edits, cer_total):.4f}")
    print(f"wer={error_rate(wer_edits, wer_total):.4f}")
    print("")
    for audio_path, reference, prediction in sample_rows[:10]:
        print(f"sample: {audio_path}")
        print(f"  ref : {reference}")
        print(f"  pred: {prediction}")


if __name__ == "__main__":
    main()
