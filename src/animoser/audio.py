"""Audio decoding, log-mel filterbank feature extraction, and PyTorch dataset."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchaudio.compliance import kaldi
from torchcodec.decoders import AudioDecoder

from animoser.schema import TASK_CLASSES, Sample


def decode_audio(
    audio_path: str | Path,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Decode audio file to 16kHz mono PCM-scaled waveform tensor.

    Scale normalized [-1.0, 1.0] float samples by 32768 (1 << 15) to match Kaldi
    fbank expectations. Output shape: (1, num_samples).
    """
    path_str = str(audio_path)
    decoder = AudioDecoder(path_str, sample_rate=target_sr, num_channels=1)
    samples = decoder.get_all_samples()
    # samples.data is (1, num_samples) float32 in range [-1.0, 1.0]
    return samples.data * (1 << 15)


def extract_fbank(  # noqa: PLR0913
    waveform: torch.Tensor,
    *,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
) -> torch.Tensor:
    """Extract 80-bin Kaldi log-mel filterbank features.

    Returns tensor of shape (time_frames, num_mel_bins).
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # kaldi.fbank expects (1, num_samples) float tensor
    feat = kaldi.fbank(
        waveform.float(),
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        energy_floor=0.0,
        sample_frequency=sample_rate,
        window_type="povey",
    )
    return feat


class AudioClassificationDataset(Dataset):
    """Dataset producing 80-bin log-mel filterbanks and task target labels."""

    def __init__(
        self,
        samples: list[Sample],
        target_sr: int = 16000,
        dither: float = 0.0,
    ) -> None:
        self.samples = samples
        self.target_sr = target_sr
        self.dither = dither

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        waveform = decode_audio(item.audio_path, target_sr=self.target_sr)
        feat = extract_fbank(
            waveform,
            sample_rate=self.target_sr,
            num_mel_bins=80,
            dither=self.dither,
        )
        return {
            "sample_id": item.sample_id,
            "feat": feat,  # (time_frames, 80)
            "targets": item.target_indices,
        }


def collate_audio_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length audio features and collate task targets.

    Returns:
        feats: (B, max_T, 80) padded float tensor
        feats_lengths: (B,) int64 tensor of valid frame counts
        {task}_label: (B,) int64 tensor with task indices (-100 for masked)
        sample_ids: list of string IDs
    """
    feats = [item["feat"] for item in batch]
    lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = feats[0].size(1)

    padded_feats = torch.zeros(len(feats), max_len, feat_dim, dtype=torch.float32)
    for i, f in enumerate(feats):
        padded_feats[i, : f.size(0), :] = f

    collated: dict[str, Any] = {
        "feats": padded_feats,
        "feats_lengths": lengths,
        "sample_ids": [item["sample_id"] for item in batch],
    }

    # Collate categorical targets for each task
    for task in TASK_CLASSES:
        labels = [item["targets"].get(task, -100) for item in batch]
        collated[f"{task}_label"] = torch.tensor(labels, dtype=torch.long)

    return collated
