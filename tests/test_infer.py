"""Fast offline unit tests for inference pipeline on synthetic audio."""

import math
import tempfile
import wave
from pathlib import Path

import torch

from animoser.infer import predict_audio
from animoser.model import MultiTaskViPVLModel
from animoser.schema import TASK_CLASSES

TINY_ENCODER_CONF = {
    "output_size": 64,
    "attention_heads": 2,
    "linear_units": 128,
    "num_blocks": 2,
    "dropout_rate": 0.0,
    "positional_dropout_rate": 0.0,
    "attention_dropout_rate": 0.0,
    "input_layer": "dw_striding",
    "pos_enc_layer_type": "chunk_rel_pos",
    "normalize_before": True,
    "selfattention_layer_type": "chunk_rel_seflattn",
    "activation_type": "swish",
    "use_cnn_module": True,
    "cnn_module_kernel": 15,
    "cnn_module_norm": "layer_norm",
    "dynamic_conv": True,
    "encoder_layerdrop": 0.0,
}


def test_predict_audio_synthetic() -> None:
    """Verify predict_audio produces valid structured output for all 6 tasks."""
    model = MultiTaskViPVLModel(encoder_conf=TINY_ENCODER_CONF)
    device = torch.device("cpu")

    # Generate synthetic 16kHz WAV file
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    t = [i / sample_rate for i in range(num_samples)]
    raw_pcm = bytearray()
    for val in t:
        sample_val = int(32767.0 * math.sin(2 * math.pi * 440.0 * val))
        raw_pcm.extend(sample_val.to_bytes(2, byteorder="little", signed=True))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = Path(f.name)
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_pcm)

    try:
        preds = predict_audio(tmp_path, model, device)
        assert len(preds) == 6

        for task, classes in TASK_CLASSES.items():
            assert task in preds
            res = preds[task]
            assert res["label"] in classes
            assert 0 <= res["class_id"] < len(classes)
            assert 0.0 <= res["confidence"] <= 1.0
            assert len(res["distribution"]) == len(classes)
    finally:
        tmp_path.unlink(missing_ok=True)
