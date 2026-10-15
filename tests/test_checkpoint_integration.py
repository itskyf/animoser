"""Integration test for loading real pretrained ViP-VL checkpoint."""

import pytest
import torch

from animoser.model import (
    MultiTaskViPVLModel,
    load_encoder_config,
    load_pretrained_encoder,
)


@pytest.mark.integration
def test_load_pretrained_checkpoint_coverage() -> None:
    """Verify loading real khanhld/vip-vl-base-vie weights into MultiTaskViPVLModel."""
    config = load_encoder_config()
    model = MultiTaskViPVLModel(encoder_conf=config)

    # Loads and dynamically asserts 100% parameter coverage
    load_pretrained_encoder(model)

    # Smoke test forward pass on full model
    speech = torch.randn(1, 100, 80)
    lengths = torch.tensor([100], dtype=torch.long)
    with torch.no_grad():
        logits = model(speech, lengths)

    assert len(logits) == 6
    for task_logits in logits.values():
        assert task_logits.shape[0] == 1
        assert not torch.isnan(task_logits).any()
