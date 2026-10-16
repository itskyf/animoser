"""Multi-task speech attribute classification model using ViP-VL encoder."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from chunkformer.modules.cmvn import GlobalCMVN
from chunkformer.modules.encoder import ChunkFormerEncoder
from huggingface_hub import hf_hub_download
from torch import nn

from animoser.schema import TASK_CLASSES

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Default ViP-VL base configuration matching khanhld/vip-vl-base-vie config.yaml
DEFAULT_VIPVL_ENCODER_CONF: dict[str, Any] = {
    "output_size": 512,
    "attention_heads": 8,
    "linear_units": 2048,
    "num_blocks": 12,
    "dropout_rate": 0.1,
    "positional_dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,
    "input_layer": "dw_striding",
    "pos_enc_layer_type": "chunk_rel_pos",
    "normalize_before": True,
    "selfattention_layer_type": "chunk_rel_seflattn",
    "activation_type": "swish",
    "use_cnn_module": True,
    "cnn_module_kernel": 15,
    "cnn_module_norm": "layer_norm",
    "dynamic_conv": True,
    "encoder_layerdrop": 0.05,
}


def load_encoder_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load ViP-VL encoder config from local file, Hugging Face Hub, or defaults."""
    if config_path is not None:
        p = Path(config_path)
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("encoder_conf", DEFAULT_VIPVL_ENCODER_CONF)

    try:
        downloaded = Path(
            hf_hub_download(
                repo_id="khanhld/vip-vl-base-vie",
                filename="config.yaml",
            )
        )
        with downloaded.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("encoder_conf", DEFAULT_VIPVL_ENCODER_CONF)
    except (OSError, yaml.YAMLError) as err:
        logger.warning(
            "Could not fetch config.yaml from Hugging Face Hub (%s); "
            "using default config.",
            err,
        )
        return dict(DEFAULT_VIPVL_ENCODER_CONF)


class ClassificationHead(nn.Module):
    """Simple 256-hidden-unit classification head with dropout."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, input_dim). Returns: logits: (B, num_classes)."""
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.out_proj(x)


class MultiTaskViPVLModel(nn.Module):
    """Multi-task speech classification model wrapping ChunkFormer encoder."""

    def __init__(
        self,
        encoder_conf: dict[str, Any] | None = None,
        tasks: dict[str, list[str]] | None = None,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        conf = dict(encoder_conf or DEFAULT_VIPVL_ENCODER_CONF)
        self.tasks = tasks or TASK_CLASSES
        self.task_names = list(self.tasks.keys())

        # Initialize GlobalCMVN buffer (will be populated from pretrained weights)
        global_cmvn = GlobalCMVN(torch.zeros(80), torch.ones(80))

        # ChunkFormerEncoder constructor requires input_size=80
        self.encoder = ChunkFormerEncoder(
            input_size=80,
            global_cmvn=global_cmvn,
            **conf,
        )

        encoder_dim = self.encoder.output_size()

        # Six lightweight categorical classification heads
        self.heads = nn.ModuleDict(
            {
                task: ClassificationHead(
                    input_dim=encoder_dim,
                    hidden_dim=256,
                    num_classes=len(classes),
                    dropout_rate=dropout_rate,
                )
                for task, classes in self.tasks.items()
            }
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute logits for all classification tasks.

        Args:
            speech: (B, max_T, 80)
            speech_lengths: (B,)

        Returns:
            dict mapping task_name -> logits (B, num_classes)
        """
        # Encoder forward pass applies global CMVN internally
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        # encoder_out: (B, T_sub, 512), encoder_mask: (B, 1, T_sub)

        # Padding-aware temporal mean pooling
        mask = encoder_mask.transpose(1, 2).float()  # (B, T_sub, 1)
        valid_counts = mask.sum(dim=1).clamp_min(1e-10)  # (B, 1)
        pooled = (encoder_out * mask).sum(dim=1) / valid_counts  # (B, 512)

        return {task: self.heads[task](pooled) for task in self.task_names}

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Compute task-wise masked cross-entropy loss.

        Returns:
            total_loss: scalar loss Tensor (or None if no task has valid supervision)
            task_losses: dict mapping task_name -> loss float
        """
        active_losses: list[torch.Tensor] = []
        task_losses: dict[str, float] = {}

        for task in self.task_names:
            label_key = f"{task}_label"
            if label_key not in batch:
                continue

            labels = batch[label_key].to(device)
            valid_mask = labels != IGNORE_INDEX

            if valid_mask.sum() == 0:
                continue

            task_logits = logits[task]
            loss_t = F.cross_entropy(task_logits[valid_mask], labels[valid_mask])
            active_losses.append(loss_t)
            task_losses[task] = loss_t.item()

        total_loss = sum(active_losses) / len(active_losses) if active_losses else None
        return total_loss, task_losses


def load_pretrained_encoder(
    model: MultiTaskViPVLModel,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Load pretrained ViP-VL encoder weights and verify parameter coverage."""
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        logger.info(
            "Downloading checkpoint from Hugging Face Hub (khanhld/vip-vl-base-vie)..."
        )
        checkpoint_path = hf_hub_download(
            repo_id="khanhld/vip-vl-base-vie",
            filename="pytorch_model.pt",
        )

    logger.info("Loading pretrained checkpoint from %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # ViP-VL checkpoint keys are prefixed with 'encoder.'
    encoder_state = {
        k[len("encoder.") :]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }

    model_state = model.encoder.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(encoder_state.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        msg = f"Missing encoder parameters in checkpoint: {sorted(missing)}"
        raise RuntimeError(msg)
    if unexpected:
        msg = f"Unexpected encoder parameters in checkpoint: {sorted(unexpected)}"
        raise RuntimeError(msg)

    # Verify tensor shapes match before loading
    for k in model_keys:
        expected_shape = model_state[k].shape
        loaded_shape = encoder_state[k].shape
        if expected_shape != loaded_shape:
            msg = (
                f"Shape mismatch for encoder parameter '{k}': "
                f"expected {expected_shape}, got {loaded_shape}"
            )
            raise RuntimeError(msg)

    # Load with strict=True
    model.encoder.load_state_dict(encoder_state, strict=True)
    logger.info("Pretrained ViP-VL encoder loaded with verified parameter coverage.")
