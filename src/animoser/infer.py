"""Minimal single-recording audio inference interface and Typer CLI."""

import json
from pathlib import Path
from typing import Annotated, Any

import torch
import typer

from animoser.audio import decode_audio, extract_fbank
from animoser.model import MultiTaskViPVLModel, load_encoder_config
from animoser.schema import TASK_CLASSES, TASK_IDX_TO_CLASS

app = typer.Typer(help="ViP-VL Multi-Task Speech Classification Inference")


def load_inference_model(
    checkpoint_path: str | Path,
    encoder_conf: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> tuple[MultiTaskViPVLModel, torch.device]:
    """Load trained multi-task model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = encoder_conf or load_encoder_config()
    model = MultiTaskViPVLModel(encoder_conf=conf).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@torch.no_grad()
def predict_audio(
    audio_path: str | Path,
    model: MultiTaskViPVLModel,
    device: torch.device,
) -> dict[str, Any]:
    """Extract features and predict all 6 speech attributes for a single audio file."""
    waveform = decode_audio(audio_path, target_sr=16000)
    feat = extract_fbank(waveform, sample_rate=16000, num_mel_bins=80, dither=0.0)

    feats = feat.unsqueeze(0).to(device)
    lengths = torch.tensor([feat.size(0)], dtype=torch.long, device=device)

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = model(feats, lengths)

    predictions: dict[str, Any] = {}
    for task, classes in TASK_CLASSES.items():
        task_logits = logits[task][0]
        probs = torch.softmax(task_logits, dim=-1)
        pred_idx = int(torch.argmax(probs).item())
        pred_label = TASK_IDX_TO_CLASS[task][pred_idx]
        confidence = float(probs[pred_idx].item())

        class_distribution = {
            cls_name: round(float(probs[i].item()), 4)
            for i, cls_name in enumerate(classes)
        }

        predictions[task] = {
            "label": pred_label,
            "class_id": pred_idx,
            "confidence": round(confidence, 4),
            "distribution": class_distribution,
        }

    return predictions


@app.command()
def predict(
    audio_path: Annotated[
        Path,
        typer.Argument(help="Path to input audio file (.wav, etc.)"),
    ],
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            "-c",
            help="Path to trained model checkpoint",
        ),
    ] = Path("data/runs/final_model.pt"),
) -> None:
    """Run multi-task inference on an audio file and print predictions as JSON."""
    if not audio_path.is_file():
        typer.secho(
            f"Error: Audio file not found at {audio_path}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    if not checkpoint.is_file():
        typer.secho(
            f"Error: Checkpoint file not found at {checkpoint}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    model, device = load_inference_model(checkpoint)
    results = predict_audio(audio_path, model, device)
    typer.echo(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
