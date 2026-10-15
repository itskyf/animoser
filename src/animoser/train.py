"""Reproducible Stratified 5-Fold fine-tuning pipeline and model refit."""

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from animoser.audio import AudioClassificationDataset, collate_audio_batch
from animoser.evaluate import (
    evaluate_predictions,
    format_evaluation_report,
    save_evaluation_results,
)
from animoser.model import (
    MultiTaskViPVLModel,
    load_encoder_config,
    load_pretrained_encoder,
)
from animoser.schema import Sample, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(  # noqa: PLR0913, PLR0917
    model: MultiTaskViPVLModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float = 5.0,
) -> float:
    """Execute one training epoch with mixed precision and zero-supervision safety."""
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for batch in dataloader:
        feats = batch["feats"].to(device)
        feats_lengths = batch["feats_lengths"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            logits = model(feats, feats_lengths)
            loss, _ = model.compute_loss(logits, batch, device)

        if loss is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            valid_batches += 1
        else:
            # Batch had zero valid supervision: safely skip backward and optimizer step
            optimizer.zero_grad()

    return total_loss / max(1, valid_batches)


@torch.no_grad()
def evaluate_fold(
    model: MultiTaskViPVLModel,
    dataloader: DataLoader,
    device: torch.device,
    fold_idx: int,
) -> list[dict[str, Any]]:
    """Generate predictions for outer validation fold."""
    model.eval()
    records: list[dict[str, Any]] = []

    for batch in dataloader:
        feats = batch["feats"].to(device)
        feats_lengths = batch["feats_lengths"].to(device)
        sample_ids = batch["sample_ids"]

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            logits = model(feats, feats_lengths)

        for i, s_id in enumerate(sample_ids):
            rec: dict[str, Any] = {
                "sample_id": s_id,
                "fold": fold_idx,
                "ground_truth": {},
                "predicted": {},
                "probabilities": {},
            }
            for task in model.task_names:
                true_val = batch[f"{task}_label"][i].item()
                rec["ground_truth"][task] = true_val

                task_probs = torch.softmax(logits[task][i], dim=-1).cpu().tolist()
                pred_val = int(torch.argmax(logits[task][i]).item())

                rec["predicted"][task] = pred_val
                rec["probabilities"][task] = task_probs

            records.append(rec)

    return records


def run_kfold_cv(  # noqa: PLR0913, PLR0917
    samples: list[Sample],
    output_dir: Path,
    k_folds: int = 5,
    epochs: int = 20,
    batch_size: int = 8,
    lr_encoder: float = 5e-5,
    lr_heads: float = 2e-4,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run Stratified K-Fold cross-validation and return OOF predictions."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_conf = load_encoder_config()

    # Stratify strictly on primary_emotion (all 11 classes represented)
    strat_labels = [s.target_indices["primary_emotion"] for s in samples]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    oof_predictions: list[dict[str, Any]] = []
    fold_splits: list[dict[str, list[str]]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(samples, strat_labels)):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        fold_splits.append(
            {
                "train": [s.sample_id for s in train_samples],
                "val": [s.sample_id for s in val_samples],
            }
        )

        logger.info(
            "=== Starting Fold %d/%d (train: %d, val: %d) ===",
            fold_idx + 1,
            k_folds,
            len(train_samples),
            len(val_samples),
        )

        train_ds = AudioClassificationDataset(train_samples, dither=1.0)
        val_ds = AudioClassificationDataset(val_samples, dither=0.0)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_audio_batch,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_audio_batch,
        )

        # Fresh model and pretrained encoder reinitialization for each fold
        model = MultiTaskViPVLModel(encoder_conf=encoder_conf).to(device)
        load_pretrained_encoder(model, checkpoint_path=checkpoint_path)

        optimizer = AdamW(
            [
                {"params": model.encoder.parameters(), "lr": lr_encoder},
                {"params": model.heads.parameters(), "lr": lr_heads},
            ],
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = create_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=(device.type == "cuda"),
        )

        # Train for predetermined fixed epochs without outer-fold early stopping
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, device
            )
            if epoch % 5 == 0 or epoch == epochs:
                logger.info(
                    "Fold %d Epoch %d/%d - loss: %.4f",
                    fold_idx + 1,
                    epoch,
                    epochs,
                    train_loss,
                )

        # Generate unbiased OOF predictions on outer validation fold
        fold_records = evaluate_fold(model, val_loader, device, fold_idx)
        oof_predictions.extend(fold_records)

        # Save fold checkpoint
        fold_ckpt_path = output_dir / f"model_fold_{fold_idx + 1}.pt"
        torch.save(model.state_dict(), fold_ckpt_path)

    # Save fold assignments
    with (output_dir / "fold_splits.json").open("w", encoding="utf-8") as f:
        json.dump(fold_splits, f, indent=2)

    return oof_predictions


def train_final_model(  # noqa: PLR0913, PLR0917
    samples: list[Sample],
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 8,
    lr_encoder: float = 5e-5,
    lr_heads: float = 2e-4,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Train one final refit model on all eligible labeled data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    logger.info("Training final model on all %d samples...", len(samples))

    encoder_conf = load_encoder_config()
    model = MultiTaskViPVLModel(encoder_conf=encoder_conf).to(device)
    load_pretrained_encoder(model, checkpoint_path=checkpoint_path)

    dataset = AudioClassificationDataset(samples, dither=1.0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_audio_batch,
    )

    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr_encoder},
            {"params": model.heads.parameters(), "lr": lr_heads},
        ],
        weight_decay=weight_decay,
    )

    total_steps = len(dataloader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = create_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(device.type == "cuda"),
    )

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device)
        if epoch % 5 == 0 or epoch == epochs:
            logger.info("Final Refit Epoch %d/%d - loss: %.4f", epoch, epochs, loss)

    final_ckpt = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_ckpt)
    logger.info("Final model saved to %s", final_ckpt)

    # Export reproducibility metadata
    metadata = {
        "num_samples": len(samples),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_encoder": lr_encoder,
        "lr_heads": lr_heads,
        "weight_decay": weight_decay,
        "seed": seed,
        "encoder_conf": encoder_conf,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return final_ckpt


def main() -> None:
    """Run full fine-tuning pipeline CLI."""
    parser = argparse.ArgumentParser(description="ViP-VL Multi-Task Audio Fine-Tuning")
    parser.add_argument("--data-dir", type=str, default="data/bronze")
    parser.add_argument("--output-dir", type=str, default="data/runs")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-encoder", type=float, default=5e-5)
    parser.add_argument("--lr-heads", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    samples = load_dataset(args.data_dir)
    if not samples:
        msg = "No valid samples found in dataset directory!"
        raise RuntimeError(msg)

    logger.info("=== Starting %d-Fold Cross Validation ===", args.k_folds)
    oof_predictions = run_kfold_cv(
        samples=samples,
        output_dir=out_dir,
        k_folds=args.k_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_encoder=args.lr_encoder,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    # Save OOF predictions
    oof_path = out_dir / "oof_predictions.json"
    with oof_path.open("w", encoding="utf-8") as f:
        json.dump(oof_predictions, f, indent=2)
    logger.info("OOF predictions saved to %s", oof_path)

    # Compute and format OOF metrics
    metrics = evaluate_predictions(oof_predictions)
    save_evaluation_results(metrics, out_dir / "metrics.json")
    report = format_evaluation_report(metrics)
    with (out_dir / "evaluation_report.md").open("w", encoding="utf-8") as f:
        f.write(report)

    logger.info("OOF Evaluation Report:\n%s", report)

    logger.info("=== Refitting Final Model on All Data ===")
    train_final_model(
        samples=samples,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_encoder=args.lr_encoder,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
