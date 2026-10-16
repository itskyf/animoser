"""Fast offline unit tests for schema parsing and label contracts."""

import pytest

from animoser.schema import (
    ACCENT_CLASSES,
    AGE_CLASSES,
    GENDER_CLASSES,
    PRIMARY_EMOTION_CLASSES,
    SPEECH_STYLE_CLASSES,
    TASK_CLASS_TO_IDX,
    TASK_CLASSES,
    TASK_IDX_TO_CLASS,
    parse_annotation,
)


def test_label_contracts_integrity() -> None:
    """Verify class vocabularies match contract requirements."""
    assert len(PRIMARY_EMOTION_CLASSES) == 11
    assert len(SPEECH_STYLE_CLASSES) == 4
    assert len(GENDER_CLASSES) == 3
    assert len(ACCENT_CLASSES) == 4
    assert len(AGE_CLASSES) == 4

    for task, classes in TASK_CLASSES.items():
        assert len(classes) == len(set(classes)), f"Duplicate classes in {task}"
        assert len(TASK_CLASS_TO_IDX[task]) == len(classes)
        assert len(TASK_IDX_TO_CLASS[task]) == len(classes)


def test_parse_annotation_standard() -> None:
    """Verify parsing of standard Vietnamese diacritic annotations."""
    raw = {
        "giới tính": "nam",
        "độ tuổi": "người trưởng thành",
        "vùng miền giọng nói": "nam",
        "phân loại phong cách": "hội thoại \u2013 đời thường",
        "cảm xúc chính": "tức giận",
        "cảm xúc phụ": "tức giận",
        "note": "",
    }
    parsed = parse_annotation(raw, sample_id="test1")
    assert parsed["primary_emotion"] == "Tức giận"
    assert parsed["secondary_emotion"] == "Tức giận"
    assert parsed["speech_style"] == "Hội thoại/Đời thường"
    assert parsed["gender"] == "Male"
    assert parsed["accent"] == "Southern"
    assert parsed["age"] == "Adult"


def test_parse_annotation_casing_and_dashes() -> None:
    """Verify casing variations and standard hyphens are handled."""
    raw = {
        "giới tính": "NỮ",
        "độ tuổi": "Người Cao Tuổi",
        "vùng miền giọng nói": "BẮC",
        "phân loại phong cách": "kể chuyện - nội dung sáng tạo",
        "cảm xúc chính": "Hào hứng",
        "cảm xúc phụ": "Trung tính",
        "note": "",
    }
    parsed = parse_annotation(raw, sample_id="test2")
    assert parsed["primary_emotion"] == "Hào hứng"
    assert parsed["secondary_emotion"] == "Trung tính"
    assert parsed["speech_style"] == "Kể chuyện/Nội dung sáng tạo"
    assert parsed["gender"] == "Female"
    assert parsed["accent"] == "Northern"
    assert parsed["age"] == "Elderly"


def test_parse_annotation_khong_co_secondary_fallback() -> None:
    """Verify 'không có' in secondary emotion falls back to primary emotion."""
    raw = {
        "giới tính": "nam",
        "độ tuổi": "người trưởng thành",
        "vùng miền giọng nói": "nam",
        "phân loại phong cách": "hội thoại \u2013 đời thường",
        "cảm xúc chính": "hào hứng",
        "cảm xúc phụ": "không có",
        "note": "",
    }
    parsed = parse_annotation(raw, sample_id="13703240")
    assert parsed["primary_emotion"] == "Hào hứng"
    assert parsed["secondary_emotion"] == "Hào hứng"


def test_parse_annotation_invalid() -> None:
    """Verify validation errors on unknown or missing labels."""
    raw_invalid = {
        "giới tính": "nam",
        "độ tuổi": "người trưởng thành",
        "vùng miền giọng nói": "nam",
        "phân loại phong cách": "hội thoại \u2013 đời thường",
        "cảm xúc chính": "không rõ cảm xúc này",
        "cảm xúc phụ": "trung tính",
        "note": "",
    }
    with pytest.raises(ValueError, match="Unknown primary emotion"):
        parse_annotation(raw_invalid, sample_id="err1")
