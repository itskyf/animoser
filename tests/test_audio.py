"""Fast offline unit tests for audio feature extraction and batch collation."""

import math

import torch

from animoser.audio import collate_audio_batch, extract_fbank
from animoser.schema import TASK_CLASSES


def test_extract_fbank_synthetic() -> None:
    """Verify 80-bin Kaldi fbank extraction on a synthetic 1-second sine wave."""
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    # 440 Hz tone scaled to 16-bit PCM integer range
    t = torch.linspace(0, duration_sec, num_samples)
    waveform = torch.sin(2 * math.pi * 440.0 * t).unsqueeze(0) * 30000.0

    feat = extract_fbank(waveform, sample_rate=sample_rate, num_mel_bins=80)
    assert feat.dim() == 2
    assert feat.size(1) == 80
    # For 1s with 25ms window and 10ms shift: num_frames ~ (1000 - 25)/10 + 1 = 98-100
    assert 95 <= feat.size(0) <= 105
    assert not torch.isnan(feat).any()
    assert not torch.isinf(feat).any()


def test_collate_audio_batch() -> None:
    """Verify dynamic padding along time dimension and label collation."""
    batch = [
        {
            "sample_id": "s1",
            "feat": torch.randn(50, 80),
            "targets": dict.fromkeys(TASK_CLASSES, 0),
        },
        {
            "sample_id": "s2",
            "feat": torch.randn(80, 80),
            "targets": dict.fromkeys(TASK_CLASSES, 1),
        },
    ]

    collated = collate_audio_batch(batch)
    assert collated["feats"].shape == (2, 80, 80)
    assert collated["feats_lengths"].tolist() == [50, 80]
    assert collated["sample_ids"] == ["s1", "s2"]

    for task in TASK_CLASSES:
        label_key = f"{task}_label"
        assert label_key in collated
        assert collated[label_key].tolist() == [0, 1]
