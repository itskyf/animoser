"""Animoser: ViP-VL Multi-Task Speech Attribute Fine-Tuning Pipeline."""

from animoser.audio import (
    AudioClassificationDataset,
    collate_audio_batch,
    decode_audio,
    extract_fbank,
)
from animoser.evaluate import evaluate_predictions, format_evaluation_report
from animoser.infer import load_inference_model, predict_audio
from animoser.model import (
    MultiTaskViPVLModel,
    load_encoder_config,
    load_pretrained_encoder,
)
from animoser.schema import (
    TASK_CLASSES,
    Sample,
    load_dataset,
    parse_annotation,
)
from animoser.train import run_kfold_cv, train_final_model

__all__ = [
    "TASK_CLASSES",
    "AudioClassificationDataset",
    "MultiTaskViPVLModel",
    "Sample",
    "collate_audio_batch",
    "decode_audio",
    "evaluate_predictions",
    "extract_fbank",
    "format_evaluation_report",
    "load_dataset",
    "load_encoder_config",
    "load_inference_model",
    "load_pretrained_encoder",
    "parse_annotation",
    "predict_audio",
    "run_kfold_cv",
    "train_final_model",
]
