"""Fast offline unit tests for MultiTaskViPVLModel architecture and loss safety."""

import torch

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


def test_model_forward_output_shapes() -> None:
    """Verify forward pass output shapes for all 6 tasks."""
    model = MultiTaskViPVLModel(encoder_conf=TINY_ENCODER_CONF)
    batch_size = 2
    time_frames = 64  # Needs to be > subsampling factor
    feat_dim = 80

    speech = torch.randn(batch_size, time_frames, feat_dim)
    lengths = torch.tensor([50, 64], dtype=torch.long)

    logits = model(speech, lengths)
    assert len(logits) == 6

    for task, classes in TASK_CLASSES.items():
        assert task in logits
        assert logits[task].shape == (batch_size, len(classes))


def test_model_masked_loss() -> None:
    """Verify masked cross-entropy computation with partial labels."""
    model = MultiTaskViPVLModel(encoder_conf=TINY_ENCODER_CONF)
    device = torch.device("cpu")

    speech = torch.randn(2, 64, 80)
    lengths = torch.tensor([64, 64], dtype=torch.long)
    logits = model(speech, lengths)

    batch = {
        "primary_emotion_label": torch.tensor([0, 1], dtype=torch.long),
        # 1 label masked (-100)
        "secondary_emotion_label": torch.tensor([0, -100], dtype=torch.long),
        # all labels masked
        "speech_style_label": torch.tensor([-100, -100], dtype=torch.long),
    }

    total_loss, task_losses = model.compute_loss(logits, batch, device)
    assert total_loss is not None
    assert total_loss.item() > 0.0
    assert "primary_emotion" in task_losses
    assert "secondary_emotion" in task_losses
    assert "speech_style" not in task_losses


def test_model_zero_supervision_safety() -> None:
    """Verify safe behavior when a batch has zero valid labels."""
    model = MultiTaskViPVLModel(encoder_conf=TINY_ENCODER_CONF)
    device = torch.device("cpu")

    speech = torch.randn(2, 64, 80)
    lengths = torch.tensor([64, 64], dtype=torch.long)
    logits = model(speech, lengths)

    # All tasks completely unannotated/masked (-100)
    batch = {
        f"{task}_label": torch.tensor([-100, -100], dtype=torch.long)
        for task in TASK_CLASSES
    }

    total_loss, task_losses = model.compute_loss(logits, batch, device)
    assert total_loss is None
    assert task_losses == {}
