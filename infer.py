from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from audio import AudioPreprocessor, load_audio
from model import SpeechToTextModel
from text import CharTokenizer
from utils import choose_device, find_vocab_path, list_audio_files, load_checkpoint, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy CTC inference for a trained CNN + BiLSTM STT model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint file.")
    parser.add_argument("--audio", type=str, default=None, help="Path to one wav file.")
    parser.add_argument("--audio-dir", type=str, default=None, help="Folder containing wav files.")
    parser.add_argument("--vocab", type=str, default=None, help="Optional path to vocab.json.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for folder inference.")
    parser.add_argument("--output", type=str, default=None, help="Optional output jsonl path.")
    return parser.parse_args()


def load_tokenizer(args: argparse.Namespace, checkpoint: Dict[str, object]) -> CharTokenizer:
    vocab_path = find_vocab_path(args.checkpoint, args.vocab)
    if vocab_path is not None and vocab_path.exists():
        return CharTokenizer.from_file(vocab_path)
    token_to_id = checkpoint.get("token_to_id")
    if token_to_id is None:
        raise FileNotFoundError("Could not find vocab.json and checkpoint does not store token_to_id.")
    return CharTokenizer(token_to_id=token_to_id)


def build_model(checkpoint: Dict[str, object], tokenizer: CharTokenizer, device: torch.device) -> SpeechToTextModel:
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
    transcripts: List[str] = []
    for index in range(token_ids.size(0)):
        transcripts.append(tokenizer.decode_ctc(token_ids[index, : int(output_lengths[index].item())].tolist()))
    return transcripts


def build_inputs(audio_paths: List[Path], preprocessor: AudioPreprocessor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    waveforms = [load_audio(audio_path, preprocessor.sample_rate) for audio_path in audio_paths]
    lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
    padded = pad_sequence(waveforms, batch_first=True)
    features, feature_lengths = preprocessor(padded, lengths)
    return features.to(device), feature_lengths.to(device)


@torch.no_grad()
def run_inference(
    model: SpeechToTextModel,
    tokenizer: CharTokenizer,
    preprocessor: AudioPreprocessor,
    audio_paths: List[Path],
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, str]]:
    outputs: List[Dict[str, str]] = []
    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start : start + batch_size]
        features, feature_lengths = build_inputs(batch_paths, preprocessor, device)
        log_probs, output_lengths = model(features, feature_lengths)
        transcripts = decode_batch(log_probs.cpu(), output_lengths.cpu(), tokenizer)
        for audio_path, transcript in zip(batch_paths, transcripts):
            outputs.append({"audio_path": str(audio_path), "transcript": transcript})
    return outputs


def main() -> None:
    args = parse_args()
    if bool(args.audio) == bool(args.audio_dir):
        raise ValueError("Provide exactly one of --audio or --audio-dir.")

    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer = load_tokenizer(args, checkpoint)
    model = build_model(checkpoint, tokenizer, device)
    preprocessor = AudioPreprocessor(checkpoint["config"]["audio"], augment=False)

    if args.audio is not None:
        audio_paths = [Path(args.audio).resolve()]
    else:
        audio_paths = list_audio_files(args.audio_dir)

    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        audio_paths=audio_paths,
        batch_size=max(1, args.batch_size),
        device=device,
    )

    if args.output is not None:
        save_jsonl(results, args.output)

    if len(results) == 1:
        print(results[0]["transcript"])
    else:
        for item in results:
            print(f"{item['audio_path']}\t{item['transcript']}")


if __name__ == "__main__":
    main()
