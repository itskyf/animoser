"""Behavior-focused test suite for the animoser crawler.

Verifies end-to-end observable outcomes across listing pagination,
data integrity, idempotency/recovery, failure handling, session safety,
confidentiality, and CLI invocation without connecting to real browsers.
"""

import asyncio
import base64
import urllib.parse
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from animoser import crawler
from animoser.crawler import (
    APPROVED_STATUS,
    CompletionManifest,
    ListingItem,
    ListingResponse,
    compute_source_fingerprint,
    fetch_all_listing_items,
    is_item_complete,
    is_safe_audio_url,
    process_item,
    run_crawler,
    run_subprocess_safe,
    save_item_files,
    validate_wav_bytes,
    verify_audio_file,
    verify_session,
)

# Synthetic fixtures for confidentiality testing
SYNTHETIC_CDP_ID = "dummy-cdp-browser-id-12345"
SYNTHETIC_CDP_URL = f"ws://127.0.0.1:0/devtools/browser/{SYNTHETIC_CDP_ID}"
SYNTHETIC_USER = "synthetic_reviewer_user"
SYNTHETIC_DOMAIN = "https://confidential-client.example.com"
SYNTHETIC_ROUND = "42"
SYNTHETIC_USER_TRANS = "99887"

# Valid 76-byte WAV audio with 16 samples (fully decodable, positive samples)
VALID_WAV_BYTES = (
    b"RIFFD\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00"
    b"\x00}\x00\x00\x02\x00\x10\x00data \x00\x00\x00" + b"\x00" * 32
)
VALID_WAV_B64 = base64.b64encode(VALID_WAV_BYTES).decode()

# 44-byte WAV header with 0 samples (zero sample data)
ZERO_SAMPLE_WAV_BYTES = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00"
    b"@\x1f\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00"
)
ZERO_SAMPLE_WAV_B64 = base64.b64encode(ZERO_SAMPLE_WAV_BYTES).decode()

# Corrupt WAV bytes with invalid stream data
CORRUPT_WAV_BYTES = (
    b"RIFFD\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00"
    b"\x00}\x00\x00\x02\x00\x10\x00data \x00\x00\x00\xff\xff"
)
CORRUPT_WAV_B64 = base64.b64encode(CORRUPT_WAV_BYTES).decode()


@pytest.fixture(autouse=True)
def _setup_synthetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject required synthetic secrets into test environment."""
    monkeypatch.setenv("CDP_WS_URL", SYNTHETIC_CDP_URL)
    monkeypatch.setenv("BASE_URL", SYNTHETIC_DOMAIN)
    monkeypatch.setenv("REVIEW_USER", SYNTHETIC_USER)


# ==============================================================================
# 1. Pydantic Model Boundary Validation
# ==============================================================================


def test_listing_item_validation_success() -> None:
    """Valid item parses successfully and preserves content verbatim."""
    raw_content = '{\n  "custom_key": "custom_val",  "status": 2\n}'
    item = ListingItem.model_validate(
        {
            "id": 12345,
            "path": "audio/sample.wav",
            "content": raw_content,
            "status": "2",
        }
    )
    assert item.id == "12345"
    assert item.path == "audio/sample.wav"
    assert item.content == raw_content
    assert item.status == "2"


@pytest.mark.parametrize(
    ("bad_item", "expected_err_substr"),
    [
        (
            {"id": "", "path": "a.wav", "content": "{}"},
            "Invalid item ID format",
        ),
        (
            {"id": "../../etc/passwd", "path": "a.wav", "content": "{}"},
            "Invalid item ID format",
        ),
        (
            {"id": "bad id spaces", "path": "a.wav", "content": "{}"},
            "Invalid item ID format",
        ),
        (
            {"id": "1", "path": "", "content": "{}"},
            "Audio path must be non-empty",
        ),
        (
            {"id": "1", "path": "no_extension", "content": "{}"},
            "contain a file extension",
        ),
        (
            {"id": "1", "path": "a.wav", "content": ""},
            "Empty annotation content",
        ),
        (
            {"id": "1", "path": "a.wav", "content": "not-valid-json{"},
            "Invalid JSON in annotation content",
        ),
        (
            {"id": "1", "path": "a.wav", "content": "{}", "status": "1"},
            "does not match approved status",
        ),
    ],
)
def test_listing_item_validation_failures(
    bad_item: dict, expected_err_substr: str
) -> None:
    """Model boundary rejects invalid item fields."""
    with pytest.raises(ValidationError, match=expected_err_substr):
        ListingItem.model_validate(bad_item)


def test_listing_response_strict_types() -> None:
    """ListingResponse enforces strict integer types and non-negative counts."""
    # Valid response
    valid_data = {
        "recordsFiltered": 1,
        "recordsTotal": 1,
        "data": [{"id": "10", "path": "a.wav", "content": "{}"}],
    }
    resp = ListingResponse.model_validate(valid_data)
    assert resp.recordsFiltered == 1
    assert len(resp.data) == 1

    # String coercion for recordsFiltered rejected by StrictInt
    with pytest.raises(ValidationError):
        ListingResponse.model_validate({"recordsFiltered": "1", "data": []})

    # Negative count rejected
    with pytest.raises(ValidationError):
        ListingResponse.model_validate({"recordsFiltered": -1, "data": []})


# ==============================================================================
# 2. Audio Verification with FFmpeg / ffprobe
# ==============================================================================


@pytest.mark.parametrize(
    ("header", "is_valid"),
    [
        (VALID_WAV_BYTES, True),
        (ZERO_SAMPLE_WAV_BYTES, True),  # Magic bytes valid, but zero samples
        (b"", False),
        (b"RIFF123", False),
        (b"NOPE$\x00\x00\x00WAVEfmt ", False),
    ],
)
def test_validate_wav_bytes(header: bytes, is_valid: bool) -> None:  # noqa: FBT001
    """Validate preliminary WAV magic header bytes."""
    assert validate_wav_bytes(header) is is_valid


@pytest.mark.asyncio
async def test_verify_audio_file_valid_audio(tmp_path: Path) -> None:
    """Valid audio with decoded PCM samples passes full verification."""
    audio_file = tmp_path / "valid.wav"
    audio_file.write_bytes(VALID_WAV_BYTES)
    await verify_audio_file(audio_file, "wav")


@pytest.mark.asyncio
async def test_verify_audio_file_zero_samples_rejected(tmp_path: Path) -> None:
    """Zero-sample WAV is rejected by ffmpeg decoded sample check."""
    audio_file = tmp_path / "zero.wav"
    audio_file.write_bytes(ZERO_SAMPLE_WAV_BYTES)
    with pytest.raises(ValueError, match="zero decoded samples"):
        await verify_audio_file(audio_file, "wav")


@pytest.mark.asyncio
async def test_verify_audio_file_corrupt_rejected(tmp_path: Path) -> None:
    """Corrupted audio stream is rejected by ffmpeg full-stream decode."""
    audio_file = tmp_path / "corrupt.wav"
    audio_file.write_bytes(CORRUPT_WAV_BYTES)
    with pytest.raises(ValueError, match="stream corrupt or undecodable"):
        await verify_audio_file(audio_file, "wav")


@pytest.mark.asyncio
async def test_verify_audio_file_container_format_mismatch(tmp_path: Path) -> None:
    """Extension mismatch with container format is rejected."""
    audio_file = tmp_path / "test.flac"
    audio_file.write_bytes(VALID_WAV_BYTES)  # WAV content with .flac extension
    with pytest.raises(ValueError, match="container format does not match extension"):
        await verify_audio_file(audio_file, "flac")


@pytest.mark.asyncio
async def test_run_subprocess_safe_timeout_reaps_process() -> None:
    """Subprocess lifecycle terminates and reaps child process on timeout."""
    # Run sleep command with very small timeout
    with pytest.raises(TimeoutError):
        await run_subprocess_safe(["sleep", "5"], timeout_sec=0.1)


# ==============================================================================
# 3. Data Integrity & Verbatim JSON
# ==============================================================================


@pytest.mark.asyncio
async def test_data_integrity_exact_audio_and_verbatim_json(
    tmp_path: Path,
) -> None:
    """Verify exact audio bytes and verbatim annotation JSON written to disk."""
    raw_annotation = (
        '{\n  "verbatim_key":  "exact_formatting_preserved",\n  "status": 2\n}'
    )
    item = {
        "id": "item_101",
        "path": "audio/clip_101.wav",
        "content": raw_annotation,
    }

    mock_page = MagicMock()
    mock_page.evaluate = AsyncMock(return_value=VALID_WAV_B64)
    semaphore = asyncio.Semaphore(1)

    ma_cau, status, err = await process_item(item, mock_page, tmp_path, semaphore)

    assert status == "downloaded"
    assert err is None
    assert ma_cau == "item_101"

    item_dir = tmp_path / "item_101"
    saved_audio = item_dir / "audio.wav"
    saved_anno = item_dir / "annotation.json"
    saved_manifest = item_dir / "manifest.json"

    assert saved_audio.is_file()
    assert saved_anno.is_file()
    assert saved_manifest.is_file()

    # Exact byte-for-byte audio match
    assert saved_audio.read_bytes() == VALID_WAV_BYTES

    # Verbatim annotation JSON preservation without re-formatting
    assert saved_anno.read_text(encoding="utf-8") == raw_annotation

    # Manifest verification
    manifest = CompletionManifest.model_validate_json(
        saved_manifest.read_text(encoding="utf-8")
    )
    listing_item = ListingItem.model_validate(item)
    assert manifest.source_fingerprint == compute_source_fingerprint(listing_item)


# ==============================================================================
# 4. Idempotency, Resume & Recovery
# ==============================================================================


@pytest.mark.asyncio
async def test_idempotency_skips_complete_and_verified_item(
    tmp_path: Path,
) -> None:
    """Re-running skips valid files without invoking browser or modifying files."""
    item_dir = tmp_path / "item_201"
    item_dir.mkdir(parents=True)

    item = ListingItem(
        id="item_201",
        path="item_201.wav",
        content='{"id": 201}',
    )

    # Save item files initially
    await save_item_files(item_dir, "wav", VALID_WAV_BYTES, item.content, item)

    audio_path = item_dir / "audio.wav"
    anno_path = item_dir / "annotation.json"
    initial_audio_mtime = audio_path.stat().st_mtime_ns
    initial_anno_mtime = anno_path.stat().st_mtime_ns

    mock_page = MagicMock()
    mock_page.evaluate = AsyncMock()
    semaphore = asyncio.Semaphore(1)

    ma_cau, status, err = await process_item(item, mock_page, tmp_path, semaphore)

    assert status == "skipped"
    assert err is None
    assert ma_cau == "item_201"

    mock_page.evaluate.assert_not_called()
    assert audio_path.stat().st_mtime_ns == initial_audio_mtime
    assert anno_path.stat().st_mtime_ns == initial_anno_mtime


@pytest.mark.asyncio
async def test_stale_source_fingerprint_forces_redownload(
    tmp_path: Path,
) -> None:
    """If server item metadata changes, cached item is re-downloaded."""
    item_dir = tmp_path / "item_stale"
    item_dir.mkdir(parents=True)

    old_item = ListingItem(
        id="item_stale",
        path="audio/v1.wav",
        content='{"version": 1}',
    )
    await save_item_files(item_dir, "wav", VALID_WAV_BYTES, old_item.content, old_item)

    new_item = ListingItem(
        id="item_stale",
        path="audio/v2.wav",  # Server changed audio path
        content='{"version": 2}',  # Server changed annotation
    )

    mock_page = MagicMock()
    mock_page.evaluate = AsyncMock(return_value=VALID_WAV_B64)
    semaphore = asyncio.Semaphore(1)

    ma_cau, status, err = await process_item(new_item, mock_page, tmp_path, semaphore)

    assert status == "downloaded"
    assert err is None
    assert ma_cau == "item_stale"

    # Manifest updated with new source fingerprint
    manifest = CompletionManifest.model_validate_json(
        (item_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.source_fingerprint == compute_source_fingerprint(new_item)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("corrupt_target", "corrupt_data"),
    [
        ("audio", b""),
        ("audio", CORRUPT_WAV_BYTES),
        ("annotation", b""),
        ("annotation", b"not-valid-json{"),
        ("annotation", b"\xff\xfe\x00\x00"),  # Invalid UTF-8
        ("manifest", b"invalid-manifest-json{"),
        ("manifest", b"\xff\xfe"),  # Invalid UTF-8 manifest
    ],
)
async def test_recovery_of_corrupted_files(
    tmp_path: Path,
    corrupt_target: str,
    corrupt_data: bytes,
) -> None:
    """Corrupted, truncated, or invalid UTF-8 files recover without neighbor damage."""
    valid_neighbor = tmp_path / "valid_neighbor"
    valid_neighbor.mkdir(parents=True)
    neighbor_item = ListingItem(
        id="valid_neighbor",
        path="neighbor.wav",
        content='{"neighbor": true}',
    )
    await save_item_files(
        valid_neighbor,
        "wav",
        VALID_WAV_BYTES,
        neighbor_item.content,
        neighbor_item,
    )
    neighbor_mtime = (valid_neighbor / "audio.wav").stat().st_mtime_ns

    target_dir = tmp_path / "target_item"
    target_dir.mkdir(parents=True)
    target_item = ListingItem(
        id="target_item",
        path="target.wav",
        content='{"recovered": true}',
    )
    await save_item_files(
        target_dir, "wav", VALID_WAV_BYTES, target_item.content, target_item
    )

    # Corrupt target file
    if corrupt_target == "audio":
        (target_dir / "audio.wav").write_bytes(corrupt_data)
    elif corrupt_target == "annotation":
        (target_dir / "annotation.json").write_bytes(corrupt_data)
    elif corrupt_target == "manifest":
        (target_dir / "manifest.json").write_bytes(corrupt_data)

    assert is_item_complete(target_dir, target_item, "wav") is False

    mock_page = MagicMock()
    mock_page.evaluate = AsyncMock(return_value=VALID_WAV_B64)
    semaphore = asyncio.Semaphore(1)

    ma_cau, status, err = await process_item(
        target_item, mock_page, tmp_path, semaphore
    )

    assert status == "downloaded"
    assert err is None
    assert ma_cau == "target_item"

    assert is_item_complete(target_dir, target_item, "wav") is True
    # Neighbor untouched
    assert (valid_neighbor / "audio.wav").stat().st_mtime_ns == neighbor_mtime


@pytest.mark.asyncio
async def test_no_leftover_temporary_files_on_failure(tmp_path: Path) -> None:
    """Failures during processing do not leave .tmp_* files on disk."""
    item = ListingItem(
        id="item_failure",
        path="audio.wav",
        content='{"ok": true}',
    )
    mock_page = MagicMock()
    # Return zero-sample audio to trigger verification failure
    mock_page.evaluate = AsyncMock(return_value=ZERO_SAMPLE_WAV_B64)
    semaphore = asyncio.Semaphore(1)

    _ma_cau, status, err = await process_item(item, mock_page, tmp_path, semaphore)
    assert status == "failed"
    assert err is not None

    item_dir = tmp_path / "item_failure"
    if item_dir.exists():
        tmp_files = list(item_dir.glob(".tmp_*"))
        assert len(tmp_files) == 0


# ==============================================================================
# 5. Safety & URL Validation
# ==============================================================================


@pytest.mark.parametrize(
    ("path", "is_safe"),
    [
        ("audio/clip.wav", True),
        (f"{SYNTHETIC_DOMAIN}/audio/clip.wav", True),
        ("//evil.com/audio/clip.wav", False),
        ("https://evil.com/audio/clip.wav", False),
        ("http://evil.com/audio/clip.wav", False),
        ("audio/../secret.wav", False),
        ("audio/%2e%2e/secret.wav", False),
        ("javascript:alert(1)", False),
        ("data:audio/wav;base64,...", False),
        (
            f"https://user:pass@{urllib.parse.urlsplit(SYNTHETIC_DOMAIN).netloc}/clip.wav",
            False,
        ),
    ],
)
def test_is_safe_audio_url(path: str, is_safe: bool) -> None:  # noqa: FBT001
    """Validate audio URL against origin, traversal, and injection."""
    safe, _ = is_safe_audio_url(path, SYNTHETIC_DOMAIN)
    assert safe is is_safe


@pytest.mark.asyncio
async def test_symlinked_item_dir_rejected(tmp_path: Path) -> None:
    """Symlinked output item directory is rejected immediately."""
    real_dir = tmp_path / "real_target"
    real_dir.mkdir(parents=True)
    symlink_dir = tmp_path / "symlink_item"
    symlink_dir.symlink_to(real_dir, target_is_directory=True)

    item = ListingItem(id="symlink_item", path="a.wav", content='{"ok": 1}')
    mock_page = MagicMock()
    semaphore = asyncio.Semaphore(1)

    ma_cau, status, err = await process_item(item, mock_page, tmp_path, semaphore)
    assert status == "failed"
    assert err == "Unsafe item directory path rejected"
    assert ma_cau == "symlink_item"


# ==============================================================================
# 6. Session Safety & Reviewer Identity Limitation
# ==============================================================================


@pytest.mark.asyncio
async def test_session_safety_wrong_origin_aborts_immediately() -> None:
    """Mismatched page origin aborts immediately before issuing API requests."""
    mock_page = MagicMock()
    mock_page.url = "https://unauthorized-domain.com/review"
    mock_page.evaluate = AsyncMock()

    with pytest.raises(RuntimeError, match="Browser page origin does not match"):
        await verify_session(mock_page, base_url=SYNTHETIC_DOMAIN)

    mock_page.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_session_safety_exact_review_route_logs_unverified_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exact route passes and records unverified identity limitation."""
    mock_page = MagicMock()
    mock_page.url = f"{SYNTHETIC_DOMAIN}/review"
    mock_page.evaluate = AsyncMock()

    caplog.clear()
    caplog.set_level("WARNING")
    await verify_session(mock_page, base_url=SYNTHETIC_DOMAIN)

    assert "Reviewer identity verification deferred" in caplog.text
    mock_page.evaluate.assert_not_called()


# ==============================================================================
# 7. Listing & Pagination Backend Contract
# ==============================================================================


@pytest.mark.asyncio
async def test_listing_pagination_and_status_contract() -> None:
    """Verify DataTables query enforces approved status and handles pagination."""
    mock_page = MagicMock()
    page_1 = [
        {"id": "1", "path": "1.wav", "status": "2", "content": '{"n": 1}'},
        {"id": "2", "path": "2.wav", "status": "2", "content": '{"n": 2}'},
    ]
    page_2 = [
        {"id": "3", "path": "3.wav", "content": '{"n": 3}'},  # Status omitted
    ]

    mock_page.evaluate = AsyncMock(
        side_effect=[
            {"recordsTotal": 3, "recordsFiltered": 3, "data": page_1},
            {"recordsTotal": 3, "recordsFiltered": 3, "data": page_2},
        ]
    )

    items = await fetch_all_listing_items(
        mock_page,
        round_id=SYNTHETIC_ROUND,
        user_trans=SYNTHETIC_USER_TRANS,
        page_size=2,
    )

    assert len(items) == 3
    assert [item.id for item in items] == ["1", "2", "3"]

    # Verify status=2 enforced in query payload
    call_args = mock_page.evaluate.call_args_list[0]
    payload = call_args[0][1]
    assert payload["status"] == APPROVED_STATUS


@pytest.mark.asyncio
async def test_listing_pagination_short_page_continues_until_total() -> None:
    """Pagination does not stop on short pages when recordsFiltered not reached."""
    mock_page = MagicMock()
    # Backend returns short page (1 item when page_size=2), but total is 2
    page_1 = [{"id": "1", "path": "1.wav", "content": '{"n": 1}'}]
    page_2 = [{"id": "2", "path": "2.wav", "content": '{"n": 2}'}]

    mock_page.evaluate = AsyncMock(
        side_effect=[
            {"recordsTotal": 2, "recordsFiltered": 2, "data": page_1},
            {"recordsTotal": 2, "recordsFiltered": 2, "data": page_2},
        ]
    )

    items = await fetch_all_listing_items(
        mock_page,
        round_id=SYNTHETIC_ROUND,
        user_trans=SYNTHETIC_USER_TRANS,
        page_size=2,
    )

    assert len(items) == 2
    assert mock_page.evaluate.await_count == 2


@pytest.mark.asyncio
async def test_listing_pagination_duplicate_fails_fast() -> None:
    """Duplicate record ID across pages raises safe RuntimeError."""
    mock_page = MagicMock()
    page_1 = [{"id": "101", "path": "a.wav", "content": '{"n": 1}'}]
    page_2 = [{"id": "101", "path": "a.wav", "content": '{"n": 1}'}]

    mock_page.evaluate = AsyncMock(
        side_effect=[
            {"recordsTotal": 2, "recordsFiltered": 2, "data": page_1},
            {"recordsTotal": 2, "recordsFiltered": 2, "data": page_2},
        ]
    )

    with pytest.raises(
        RuntimeError, match="Duplicate record ID encountered in pagination"
    ):
        await fetch_all_listing_items(
            mock_page,
            round_id=SYNTHETIC_ROUND,
            user_trans=SYNTHETIC_USER_TRANS,
            page_size=1,
        )


@pytest.mark.asyncio
async def test_listing_pagination_missing_discrepancy_fails_fast() -> None:
    """Discrepancy between collected items and recordsFiltered raises error."""
    mock_page = MagicMock()
    page_1 = [{"id": "201", "path": "a.wav", "content": '{"n": 1}'}]

    # Server claims recordsFiltered=5, returns 1 item then empty list
    mock_page.evaluate = AsyncMock(
        side_effect=[
            {"recordsTotal": 5, "recordsFiltered": 5, "data": page_1},
            {"recordsTotal": 5, "recordsFiltered": 5, "data": []},
        ]
    )

    with pytest.raises(RuntimeError, match="Incomplete listing: record count mismatch"):
        await fetch_all_listing_items(
            mock_page,
            round_id=SYNTHETIC_ROUND,
            user_trans=SYNTHETIC_USER_TRANS,
            page_size=5,
        )


@pytest.mark.asyncio
async def test_listing_api_schema_failure_raises_safe_error() -> None:
    """Malformed DataTables schema raises safe generic error without leaking body."""
    mock_page = MagicMock()
    mock_page.evaluate = AsyncMock(return_value={"corrupted": "no recordsFiltered"})

    with pytest.raises(RuntimeError, match="Invalid listing response schema"):
        await fetch_all_listing_items(
            mock_page,
            round_id=SYNTHETIC_ROUND,
            user_trans=SYNTHETIC_USER_TRANS,
        )


# ==============================================================================
# 8. Confidentiality & Safe Errors
# ==============================================================================


@pytest.mark.asyncio
async def test_confidentiality_cdp_connection_failure_masks_secrets() -> None:
    """CDP connection failure raises safe message without leaking CDP URL."""
    mock_chromium = MagicMock()
    mock_chromium.connect_over_cdp = AsyncMock(
        side_effect=RuntimeError(f"Cannot connect to {SYNTHETIC_CDP_URL}")
    )
    mock_pw_instance = MagicMock()
    mock_pw_instance.chromium = mock_chromium
    mock_pw_context = AsyncMock()
    mock_pw_context.__aenter__.return_value = mock_pw_instance
    mock_pw_context.__aexit__.return_value = None

    with (
        patch("animoser.crawler.async_playwright", return_value=mock_pw_context),
        pytest.raises(RuntimeError) as exc_info,
    ):
        await run_crawler(
            round_id=SYNTHETIC_ROUND,
            user_trans=SYNTHETIC_USER_TRANS,
        )

    err_msg = str(exc_info.value)
    assert SYNTHETIC_CDP_ID not in err_msg
    assert SYNTHETIC_CDP_URL not in err_msg
    assert err_msg == "Failed to connect to browser session over CDP"


@pytest.mark.asyncio
async def test_confidentiality_no_secrets_in_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CDP URLs, client domain, and credentials never appear in log output."""
    mock_page = MagicMock()
    mock_page.url = f"{SYNTHETIC_DOMAIN}/review"
    mock_page.evaluate = AsyncMock(
        return_value={"recordsTotal": 0, "recordsFiltered": 0, "data": []}
    )

    mock_context = MagicMock()
    mock_context.pages = [mock_page]

    mock_browser = MagicMock()
    mock_browser.contexts = [mock_context]

    mock_chromium = MagicMock()
    mock_chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)

    mock_pw_instance = MagicMock()
    mock_pw_instance.chromium = mock_chromium

    mock_pw_context = AsyncMock()
    mock_pw_context.__aenter__.return_value = mock_pw_instance
    mock_pw_context.__aexit__.return_value = None

    caplog.clear()
    caplog.set_level("INFO")
    with patch("animoser.crawler.async_playwright", return_value=mock_pw_context):
        await run_crawler(
            output_dir=tmp_path,
            round_id=SYNTHETIC_ROUND,
            user_trans=SYNTHETIC_USER_TRANS,
        )

    log_output = caplog.text
    assert SYNTHETIC_CDP_ID not in log_output
    assert SYNTHETIC_CDP_URL not in log_output
    assert SYNTHETIC_DOMAIN not in log_output
    assert SYNTHETIC_USER not in log_output


# ==============================================================================
# 9. CLI Runner & Argument Validation
# ==============================================================================


def test_cli_smoke_runner_with_required_args(
    tmp_path: Path,
) -> None:
    """CLI runner succeeds with required long-form arguments."""
    runner = CliRunner()
    with patch("animoser.crawler.run_crawler", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (0, 0, 0)
        result = runner.invoke(
            crawler.app,
            [
                "--round",
                SYNTHETIC_ROUND,
                "--user-trans",
                SYNTHETIC_USER_TRANS,
                "--output-dir",
                str(tmp_path),
                "--concurrency",
                "2",
                "--limit",
                "1",
            ],
        )

    assert result.exit_code == 0
    mock_run.assert_awaited_once_with(
        output_dir=tmp_path,
        concurrency=2,
        limit=1,
        round_id=SYNTHETIC_ROUND,
        user_trans=SYNTHETIC_USER_TRANS,
    )


def test_cli_rejects_removed_status_option() -> None:
    """CLI fails when removed --status option is passed."""
    runner = CliRunner()
    result = runner.invoke(
        crawler.app,
        [
            "--round",
            SYNTHETIC_ROUND,
            "--user-trans",
            SYNTHETIC_USER_TRANS,
            "--status",
            "2",
        ],
    )
    assert result.exit_code != 0
    assert "No such option: --status" in result.output
