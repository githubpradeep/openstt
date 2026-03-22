from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio import AudioPreprocessor, describe_feature_batch
from config import load_config, save_config
from dataset import LJSpeechDataset, SpeechCollator, prepare_ljspeech_splits
from metrics import cer_stats, error_rate, wer_stats
from model import SpeechToTextModel
from text import CharTokenizer
from utils import AverageMeter, choose_device, ensure_dir, format_learning_rate, load_checkpoint, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fast CNN + BiLSTM + CTC STT model on LJSpeech.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset.root.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override experiment.output_dir.")
    parser.add_argument("--subset-size", type=int, default=None, help="Override dataset.subset_size.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--device", type=str, default=None, help="Override training.device.")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    overrides: Dict[str, Dict[str, object]] = {}
    if args.dataset_root is not None:
        overrides.setdefault("dataset", {})["root"] = args.dataset_root
    if args.output_dir is not None:
        overrides.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.subset_size is not None:
        overrides.setdefault("dataset", {})["subset_size"] = args.subset_size
    if args.resume is not None:
        overrides.setdefault("training", {})["resume_from"] = args.resume
    if args.device is not None:
        overrides.setdefault("training", {})["device"] = args.device
    return overrides


def move_tensor_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    for key in ("features", "feature_lengths", "targets", "target_lengths"):
        moved[key] = moved[key].to(device)
    return moved


def decode_predictions(log_probs: torch.Tensor, output_lengths: torch.Tensor, tokenizer: CharTokenizer) -> List[str]:
    token_ids = log_probs.argmax(dim=-1).transpose(0, 1)
    decoded: List[str] = []
    for index in range(token_ids.size(0)):
        decoded.append(tokenizer.decode_ctc(token_ids[index, : int(output_lengths[index].item())].tolist()))
    return decoded


def validate_output_lengths(output_lengths: torch.Tensor, target_lengths: torch.Tensor) -> None:
    invalid = output_lengths < target_lengths
    if invalid.any():
        pairs = list(zip(output_lengths[invalid].tolist(), target_lengths[invalid].tolist()))
        raise ValueError(f"CTC invalid lengths found. output_lengths < target_lengths for samples: {pairs[:5]}")


def run_sanity_checks(
    model: SpeechToTextModel,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    tokenizer: CharTokenizer,
    device: torch.device,
    config: Dict[str, object],
) -> None:
    debug_config = config["debug"]
    batch = next(iter(loader))
    if debug_config["print_batch_shapes"]:
        print(f"[sanity] waveforms={tuple(batch['waveforms'].shape)} waveform_lengths={tuple(batch['waveform_lengths'].shape)}")
        print(f"[sanity] features={tuple(batch['features'].shape)} targets={tuple(batch['targets'].shape)}")
    if debug_config["print_spectrogram_stats"]:
        print(f"[sanity] {describe_feature_batch(batch['features'], batch['feature_lengths'])}")

    batch = move_tensor_batch_to_device(batch, device)
    model.train()
    log_probs, output_lengths = model(batch["features"], batch["feature_lengths"])
    loss = criterion(log_probs, batch["targets"], output_lengths, batch["target_lengths"])
    if debug_config["verify_output_lengths"]:
        validate_output_lengths(output_lengths.cpu(), batch["target_lengths"].cpu())
    if debug_config["run_forward_pass"]:
        print(
            f"[sanity] log_probs={tuple(log_probs.shape)} "
            f"output_lengths[min={int(output_lengths.min())}, max={int(output_lengths.max())}] "
            f"loss={loss.item():.4f}"
        )
    if debug_config["run_backward_pass"]:
        model.zero_grad(set_to_none=True)
        loss.backward()
        model.zero_grad(set_to_none=True)
        print("[sanity] backward pass completed")
    if debug_config["print_predictions_before_training"]:
        predictions = decode_predictions(log_probs.detach().cpu(), output_lengths.detach().cpu(), tokenizer)
        print("[sanity] sample greedy decodes before training:")
        for reference, prediction in zip(batch["texts"][: debug_config["sample_predictions"]], predictions[: debug_config["sample_predictions"]]):
            print(f"  ref:  {reference}")
            print(f"  pred: {prediction}")


def train_one_epoch(
    model: SpeechToTextModel,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    tokenizer: CharTokenizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, object],
) -> Dict[str, float]:
    training_config = config["training"]
    debug_config = config["debug"]
    use_amp = bool(training_config["amp"]) and device.type == "cuda"

    model.train()
    loss_meter = AverageMeter()
    cer_edits = 0
    cer_total = 0
    wer_edits = 0
    wer_total = 0

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_tensor_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with (torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()):
            log_probs, output_lengths = model(batch["features"], batch["feature_lengths"])
            if debug_config["verify_output_lengths"]:
                validate_output_lengths(output_lengths.detach().cpu(), batch["target_lengths"].detach().cpu())
            loss = criterion(log_probs, batch["targets"], output_lengths, batch["target_lengths"])

        scaler.scale(loss).backward()
        if training_config["grad_clip"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()

        predictions = decode_predictions(log_probs.detach().cpu(), output_lengths.detach().cpu(), tokenizer)
        batch_cer_edits, batch_cer_total = cer_stats(predictions, batch["texts"])
        batch_wer_edits, batch_wer_total = wer_stats(predictions, batch["texts"])
        cer_edits += batch_cer_edits
        cer_total += batch_cer_total
        wer_edits += batch_wer_edits
        wer_total += batch_wer_total
        loss_meter.update(loss.item(), n=len(batch["texts"]))

        if step % int(training_config["log_interval"]) == 0 or step == len(loader):
            progress.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                cer=f"{error_rate(cer_edits, cer_total):.4f}",
                wer=f"{error_rate(wer_edits, wer_total):.4f}",
                lr=format_learning_rate(optimizer),
            )

    return {
        "loss": loss_meter.avg,
        "cer": error_rate(cer_edits, cer_total),
        "wer": error_rate(wer_edits, wer_total),
    }


@torch.no_grad()
def evaluate(
    model: SpeechToTextModel,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    tokenizer: CharTokenizer,
    device: torch.device,
    config: Dict[str, object],
) -> Dict[str, object]:
    debug_config = config["debug"]
    model.eval()
    loss_meter = AverageMeter()
    cer_edits = 0
    cer_total = 0
    wer_edits = 0
    wer_total = 0
    samples: List[Dict[str, str]] = []

    for batch_index, batch in enumerate(tqdm(loader, desc="validate", leave=False), start=1):
        batch = move_tensor_batch_to_device(batch, device)
        log_probs, output_lengths = model(batch["features"], batch["feature_lengths"])
        if debug_config["verify_output_lengths"]:
            validate_output_lengths(output_lengths.detach().cpu(), batch["target_lengths"].detach().cpu())
        loss = criterion(log_probs, batch["targets"], output_lengths, batch["target_lengths"])
        predictions = decode_predictions(log_probs.detach().cpu(), output_lengths.detach().cpu(), tokenizer)
        batch_cer_edits, batch_cer_total = cer_stats(predictions, batch["texts"])
        batch_wer_edits, batch_wer_total = wer_stats(predictions, batch["texts"])
        cer_edits += batch_cer_edits
        cer_total += batch_cer_total
        wer_edits += batch_wer_edits
        wer_total += batch_wer_total
        loss_meter.update(loss.item(), n=len(batch["texts"]))

        if batch_index == 1:
            limit = int(debug_config["sample_predictions"])
            for reference, prediction, audio_path in zip(batch["texts"][:limit], predictions[:limit], batch["audio_paths"][:limit]):
                samples.append(
                    {
                        "audio_path": audio_path,
                        "reference": reference,
                        "prediction": prediction,
                    }
                )

    return {
        "loss": loss_meter.avg,
        "cer": error_rate(cer_edits, cer_total),
        "wer": error_rate(wer_edits, wer_total),
        "samples": samples,
    }


def create_dataloader(dataset: LJSpeechDataset, collator: SpeechCollator, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=build_overrides(args))

    seed = int(config["dataset"]["seed"])
    seed_everything(seed)

    device = choose_device(config["training"]["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["training"]["cudnn_benchmark"])

    output_dir = ensure_dir(config["experiment"]["output_dir"])
    save_config(config, output_dir / "resolved_config.yaml")

    split_info = prepare_ljspeech_splits(
        dataset_root=config["dataset"]["root"],
        output_dir=output_dir,
        val_ratio=float(config["dataset"]["val_ratio"]),
        seed=seed,
        subset_size=config["dataset"]["subset_size"],
        val_subset_size=config["dataset"]["val_subset_size"],
        overfit_mode=bool(config["dataset"]["overfit_mode"]),
        overfit_size=int(config["dataset"]["overfit_size"]),
    )

    resume_path = config["training"]["resume_from"]
    resume_checkpoint = load_checkpoint(resume_path, torch.device("cpu")) if resume_path else None

    vocab_path = output_dir / "vocab.json"
    if vocab_path.exists():
        tokenizer = CharTokenizer.from_file(vocab_path)
    elif resume_checkpoint is not None and resume_checkpoint.get("token_to_id") is not None:
        tokenizer = CharTokenizer(token_to_id=resume_checkpoint["token_to_id"])
        tokenizer.save(vocab_path)
    else:
        tokenizer = CharTokenizer.build(record["text"] for record in split_info["train_records"])
        tokenizer.save(vocab_path)

    train_dataset = LJSpeechDataset(
        manifest_path=split_info["train_manifest"],
        tokenizer=tokenizer,
        sample_rate=int(config["audio"]["sample_rate"]),
    )
    val_dataset = LJSpeechDataset(
        manifest_path=split_info["val_manifest"],
        tokenizer=tokenizer,
        sample_rate=int(config["audio"]["sample_rate"]),
    )

    train_loader = create_dataloader(
        dataset=train_dataset,
        collator=SpeechCollator(AudioPreprocessor(config["audio"], augment=True)),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
    )
    val_loader = create_dataloader(
        dataset=val_dataset,
        collator=SpeechCollator(AudioPreprocessor(config["audio"], augment=False)),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = None
    scheduler_config = config["training"].get("scheduler", {})
    if scheduler_config.get("name") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 3)),
            min_lr=float(scheduler_config.get("min_lr", 1.0e-5)),
        )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"]["amp"]) and device.type == "cuda")
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    start_epoch = 1
    best_cer = float("inf")
    if resume_path:
        checkpoint = resume_checkpoint if resume_checkpoint is not None else load_checkpoint(resume_path, device=device)
        if device.type != "cpu":
            checkpoint = load_checkpoint(resume_path, device=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_cer = float(checkpoint.get("best_cer", best_cer))
        print(f"resumed from {resume_path} at epoch {start_epoch}")

    run_sanity_checks(
        model=model,
        loader=train_loader,
        criterion=criterion,
        tokenizer=tokenizer,
        device=device,
        config=config,
    )

    max_epochs = int(config["training"]["epochs"])
    for epoch in range(start_epoch, max_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            tokenizer=tokenizer,
            device=device,
            epoch=epoch,
            config=config,
        )
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} "
            f"train_cer={train_metrics['cer']:.4f} train_wer={train_metrics['wer']:.4f} "
            f"lr={format_learning_rate(optimizer)}"
        )

        val_metrics = None
        if epoch % int(config["training"]["val_interval"]) == 0:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                tokenizer=tokenizer,
                device=device,
                config=config,
            )
            print(
                f"[epoch {epoch}] val_loss={val_metrics['loss']:.4f} "
                f"val_cer={val_metrics['cer']:.4f} val_wer={val_metrics['wer']:.4f}"
            )
            for sample in val_metrics["samples"]:
                print(f"  sample: {sample['audio_path']}")
                print(f"    ref : {sample['reference']}")
                print(f"    pred: {sample['prediction']}")
            if scheduler is not None:
                scheduler.step(val_metrics["loss"])

        if val_metrics is not None and val_metrics["cer"] < best_cer:
            best_cer = float(val_metrics["cer"])
            print(f"[epoch {epoch}] saved new best checkpoint with cer={best_cer:.4f}")

        checkpoint_state = {
            "epoch": epoch,
            "best_cer": best_cer,
            "config": config,
            "token_to_id": tokenizer.token_to_id,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        }
        save_checkpoint(checkpoint_state, output_dir / "last.pt")

        if val_metrics is not None and val_metrics["cer"] == best_cer:
            save_checkpoint(checkpoint_state, output_dir / "best.pt")

        if bool(config["training"]["save_every_epoch"]):
            save_checkpoint(checkpoint_state, output_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
