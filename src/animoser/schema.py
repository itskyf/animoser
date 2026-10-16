"""Annotation schema, label contracts, and dataset loading for ViP-VL fine-tuning."""

import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label contracts (preserving Vietnamese diacritics and exact contract names)
# ---------------------------------------------------------------------------

PRIMARY_EMOTION_CLASSES: list[str] = [
    "Hạnh phúc",
    "Hào hứng",
    "Ngạc nhiên",
    "Sợ hãi",
    "Lo lắng",
    "Tức giận",
    "Khó chịu",
    "Đau khổ",
    "Buồn bã",
    "Chán nản",
    "Trung tính",
]

SECONDARY_EMOTION_CLASSES: list[str] = list(PRIMARY_EMOTION_CLASSES)

SPEECH_STYLE_CLASSES: list[str] = [
    "Quảng cáo",
    "Kể chuyện/Nội dung sáng tạo",
    "Hội thoại/Đời thường",
    "Chính thống/Trang trọng/Phát biểu",
]

GENDER_CLASSES: list[str] = ["Male", "Female", "Unknown"]

ACCENT_CLASSES: list[str] = ["Northern", "Central", "Southern", "Unknown"]

AGE_CLASSES: list[str] = ["Child", "Adult", "Elderly", "Unknown"]

TASK_CLASSES: dict[str, list[str]] = {
    "primary_emotion": PRIMARY_EMOTION_CLASSES,
    "secondary_emotion": SECONDARY_EMOTION_CLASSES,
    "speech_style": SPEECH_STYLE_CLASSES,
    "gender": GENDER_CLASSES,
    "accent": ACCENT_CLASSES,
    "age": AGE_CLASSES,
}

TASK_CLASS_TO_IDX: dict[str, dict[str, int]] = {
    task: {cls_name: idx for idx, cls_name in enumerate(classes)}
    for task, classes in TASK_CLASSES.items()
}

TASK_IDX_TO_CLASS: dict[str, dict[int, str]] = {
    task: dict(enumerate(classes)) for task, classes in TASK_CLASSES.items()
}


def _norm_str(s: str | None) -> str:
    """Normalize string: NFC unicode, strip whitespace, lowercase, normalize dashes."""
    if s is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(s)).strip().lower()
    # Replace en-dash (\u2013) and em-dash (\u2014) with standard hyphen
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    # Normalize internal spaces around hyphens
    return " ".join(normalized.split())


# Internal normalized lookup tables for resilient schema parsing
_NORM_EMOTION_MAP: dict[str, str] = {_norm_str(c): c for c in PRIMARY_EMOTION_CLASSES}

_NORM_SPEECH_STYLE_MAP: dict[str, str] = {
    "quang cao": "Quảng cáo",
    "quảng cáo": "Quảng cáo",
    "ke chuyen - noi dung sang tao": "Kể chuyện/Nội dung sáng tạo",
    "kể chuyện - nội dung sáng tạo": "Kể chuyện/Nội dung sáng tạo",
    "hoi thoai - doi thuong": "Hội thoại/Đời thường",
    "hội thoại - đời thường": "Hội thoại/Đời thường",
    "chinh thong - trang trong - phat bieu": "Chính thống/Trang trọng/Phát biểu",
    "chính thống - trang trọng - phát biểu": "Chính thống/Trang trọng/Phát biểu",
}

_NORM_GENDER_MAP: dict[str, str] = {
    "nam": "Male",
    "male": "Male",
    "nữ": "Female",
    "nu": "Female",
    "female": "Female",
    "unknown": "Unknown",
    "khong ro": "Unknown",
    "không rõ": "Unknown",
}

_NORM_ACCENT_MAP: dict[str, str] = {
    "bac": "Northern",
    "bắc": "Northern",
    "northern": "Northern",
    "trung": "Central",
    "central": "Central",
    "nam": "Southern",
    "southern": "Southern",
    "unknown": "Unknown",
    "khong ro": "Unknown",
    "không rõ": "Unknown",
}

_NORM_AGE_MAP: dict[str, str] = {
    "tre em": "Child",
    "trẻ em": "Child",
    "child": "Child",
    "nguoi truong thanh": "Adult",
    "người trưởng thành": "Adult",
    "adult": "Adult",
    "nguoi cao tuoi": "Elderly",
    "người cao tuổi": "Elderly",
    "elderly": "Elderly",
    "unknown": "Unknown",
    "khong ro": "Unknown",
    "không rõ": "Unknown",
}


# ---------------------------------------------------------------------------
# Pydantic boundary schema for raw annotation.json
# ---------------------------------------------------------------------------


class RawAnnotation(BaseModel):
    """Raw review annotation schema matching crawled bronze format."""

    gioi_tinh: str = Field(alias="giới tính")
    do_tuoi: str = Field(alias="độ tuổi")
    vung_mien: str = Field(alias="vùng miền giọng nói")
    phong_cach: str = Field(alias="phân loại phong cách")
    cam_xuc_chinh: str = Field(alias="cảm xúc chính")
    cam_xuc_phu: str = Field(alias="cảm xúc phụ")
    note: str | None = Field(default="")

    model_config = {"populate_by_name": True, "extra": "ignore"}


@dataclass(frozen=True)
class Sample:
    """Validated sample ready for audio feature extraction and modeling."""

    sample_id: str
    audio_path: Path
    primary_emotion: str
    secondary_emotion: str
    speech_style: str
    gender: str
    accent: str
    age: str
    target_indices: dict[str, int]


def parse_annotation(raw_dict: dict[str, Any], sample_id: str = "") -> dict[str, str]:
    """Parse and normalize raw annotation dictionary into canonical contract labels."""
    validated = RawAnnotation.model_validate(raw_dict)

    # 1. Primary emotion
    norm_primary = _norm_str(validated.cam_xuc_chinh)
    primary_emotion = _NORM_EMOTION_MAP.get(norm_primary)
    if not primary_emotion:
        msg = (
            f"Unknown primary emotion '{validated.cam_xuc_chinh}' in sample {sample_id}"
        )
        raise ValueError(msg)

    # 2. Secondary emotion: "không có" falls back to primary emotion per contract rule
    norm_secondary = _norm_str(validated.cam_xuc_phu)
    if norm_secondary in ("không có", "khong co", "none", "", "no"):
        secondary_emotion = primary_emotion
    else:
        secondary_emotion = _NORM_EMOTION_MAP.get(norm_secondary)
        if not secondary_emotion:
            msg = (
                f"Unknown secondary emotion '{validated.cam_xuc_phu}' in "
                f"sample {sample_id}"
            )
            raise ValueError(msg)

    # 3. Speech style
    norm_style = _norm_str(validated.phong_cach)
    speech_style = _NORM_SPEECH_STYLE_MAP.get(norm_style)
    if not speech_style:
        msg = f"Unknown speech style '{validated.phong_cach}' in sample {sample_id}"
        raise ValueError(msg)

    # 4. Gender
    norm_gender = _norm_str(validated.gioi_tinh)
    gender = _NORM_GENDER_MAP.get(norm_gender)
    if not gender:
        msg = f"Unknown gender '{validated.gioi_tinh}' in sample {sample_id}"
        raise ValueError(msg)

    # 5. Accent
    norm_accent = _norm_str(validated.vung_mien)
    accent = _NORM_ACCENT_MAP.get(norm_accent)
    if not accent:
        msg = f"Unknown accent '{validated.vung_mien}' in sample {sample_id}"
        raise ValueError(msg)

    # 6. Age
    norm_age = _norm_str(validated.do_tuoi)
    age = _NORM_AGE_MAP.get(norm_age)
    if not age:
        msg = f"Unknown age '{validated.do_tuoi}' in sample {sample_id}"
        raise ValueError(msg)

    return {
        "primary_emotion": primary_emotion,
        "secondary_emotion": secondary_emotion,
        "speech_style": speech_style,
        "gender": gender,
        "accent": accent,
        "age": age,
    }


def load_dataset(data_dir: Path | str = "data/bronze") -> list[Sample]:
    """Scan and validate local audio dataset directories."""
    base_dir = Path(data_dir)
    if not base_dir.is_dir():
        msg = f"Dataset directory not found: {base_dir}"
        raise FileNotFoundError(msg)

    samples: list[Sample] = []
    skipped = 0

    # Scan sorted directories for deterministic ordering
    for item_dir in sorted(base_dir.iterdir()):
        if not item_dir.is_dir():
            continue

        sample_id = item_dir.name
        anno_path = item_dir / "annotation.json"
        audio_files = list(item_dir.glob("audio.*"))

        if not anno_path.is_file() or not audio_files:
            logger.warning("Incomplete sample directory: %s", item_dir)
            skipped += 1
            continue

        audio_path = audio_files[0]
        try:
            with anno_path.open("r", encoding="utf-8") as f:
                raw_anno = json.load(f)
            labels = parse_annotation(raw_anno, sample_id=sample_id)
        except (json.JSONDecodeError, ValueError) as err:
            logger.error("Failed to parse annotation in %s: %s", item_dir, err)
            skipped += 1
            continue

        target_indices = {
            task: TASK_CLASS_TO_IDX[task][lbl] for task, lbl in labels.items()
        }

        samples.append(
            Sample(
                sample_id=sample_id,
                audio_path=audio_path,
                primary_emotion=labels["primary_emotion"],
                secondary_emotion=labels["secondary_emotion"],
                speech_style=labels["speech_style"],
                gender=labels["gender"],
                accent=labels["accent"],
                age=labels["age"],
                target_indices=target_indices,
            )
        )

    logger.info(
        "Loaded %d valid samples from %s (skipped: %d)",
        len(samples),
        base_dir,
        skipped,
    )
    return samples
