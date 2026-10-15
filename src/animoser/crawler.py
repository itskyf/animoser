"""Crawler for downloading reviewed audio dataset using Rustwright."""

import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import urllib.parse
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field, StrictInt, ValidationError, field_validator
from rustwright import Error as RustwrightError
from rustwright.async_api import Page, async_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APPROVED_STATUS = "2"  # "Đạt"
DEFAULT_CONCURRENCY = 4
DEFAULT_OUTPUT_DIR = Path("data/bronze")
MIN_WAV_HEADER_LEN = 12
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


class ListingItem(BaseModel):
    """Review item in backend listing response."""

    id: str
    path: str
    content: str
    status: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, v: object) -> str:
        """Validate that item ID is non-empty and contains only safe characters."""
        s = str(v).strip()
        if not s or not SAFE_ID_PATTERN.match(s) or ".." in s:
            msg = "Invalid item ID format"
            raise ValueError(msg)
        return s

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate non-empty audio path with file extension."""
        s = v.strip()
        if not s or not Path(s).suffix:
            msg = "Audio path must be non-empty and contain a file extension"
            raise ValueError(msg)
        return s

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure annotation content is valid JSON while preserving formatting."""
        s = v.strip()
        if not s:
            msg = "Empty annotation content"
            raise ValueError(msg)
        try:
            json.loads(s)
        except Exception as err:
            msg = "Invalid JSON in annotation content"
            raise ValueError(msg) from err
        return v

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: object) -> str | None:
        """Validate status matches approved filter when present."""
        if v is None:
            return None
        s = str(v).strip()
        if s != APPROVED_STATUS:
            msg = f"Status '{s}' does not match approved status '{APPROVED_STATUS}'"
            raise ValueError(msg)
        return s


class ListingResponse(BaseModel):
    """DataTables paginated listing response."""

    recordsFiltered: StrictInt = Field(ge=0)
    recordsTotal: StrictInt | None = None
    data: list[ListingItem]


class CompletionManifest(BaseModel):
    """Minimal completion manifest tracking item hashes and source fingerprint."""

    source_fingerprint: str
    audio_sha256: str
    annotation_sha256: str


def compute_source_fingerprint(item: ListingItem) -> str:
    """Generate deterministic SHA-256 fingerprint from structured source metadata."""
    payload = [item.id, item.path, item.content]
    canonical_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def is_safe_audio_url(audio_path: str, base_url: str) -> tuple[bool, str]:
    """Validate audio path against base URL using standard URL parsing."""
    unquoted_raw = urllib.parse.unquote(audio_path)
    raw_segments = [seg for seg in unquoted_raw.split("/") if seg and seg != "."]
    if ".." in raw_segments:
        return False, ""

    parsed_base = urllib.parse.urlsplit(base_url)
    expected_scheme = parsed_base.scheme.lower()
    expected_netloc = parsed_base.netloc.lower()

    if not expected_scheme or not expected_netloc:
        return False, ""

    parsed_target = urllib.parse.urlsplit(audio_path)
    if not parsed_target.scheme:
        if audio_path.startswith("//"):
            return False, ""
        joined_url = urllib.parse.urljoin(
            f"{base_url.rstrip('/')}/",
            audio_path.lstrip("/"),
        )
        parsed_target = urllib.parse.urlsplit(joined_url)

    target_scheme = parsed_target.scheme.lower()
    target_netloc = parsed_target.netloc.lower()

    if (
        target_scheme not in ("http", "https")
        or target_scheme != expected_scheme
        or target_netloc != expected_netloc
        or parsed_target.username
        or parsed_target.password
    ):
        return False, ""

    unquoted_path = urllib.parse.unquote(parsed_target.path)
    segments = [seg for seg in unquoted_path.split("/") if seg and seg != "."]
    if ".." in segments:
        return False, ""

    return True, parsed_target.geturl()


async def run_subprocess_safe(
    cmd: list[str],
    timeout_sec: float = 10.0,
) -> tuple[int, bytes, bytes]:
    """Execute subprocess nonblocking and guarantee reaping on timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(timeout_sec):
            stdout, stderr = await proc.communicate()
            return proc.returncode, stdout, stderr
    except (TimeoutError, asyncio.CancelledError):
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            async with asyncio.timeout(2.0):
                await proc.wait()
        except (TimeoutError, ProcessLookupError):
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
        raise


async def verify_audio_file(
    audio_path: Path,
    expected_ext: str,
    timeout_sec: float = 10.0,
) -> None:
    """Verify audio container format, decodability, and presence of decoded samples."""
    # 1. ffprobe inspect container format and audio streams
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=format_name",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(audio_path),
    ]
    try:
        retcode, stdout, _ = await run_subprocess_safe(probe_cmd, timeout_sec)
    except TimeoutError as err:
        msg = "Audio inspection timed out"
        raise ValueError(msg) from err

    if retcode != 0:
        msg = "Audio inspection failed: unreadable audio container"
        raise ValueError(msg)

    try:
        info = json.loads(stdout.decode("utf-8", errors="replace"))
    except Exception as err:
        msg = "Audio inspection failed: invalid ffprobe output"
        raise ValueError(msg) from err

    format_name = info.get("format", {}).get("format_name", "")
    norm_ext = expected_ext.lower().lstrip(".")
    if norm_ext and norm_ext not in format_name.split(","):
        msg = "Audio inspection failed: container format does not match extension"
        raise ValueError(msg)

    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        msg = "Audio inspection failed: no audio stream found"
        raise ValueError(msg)

    # 2. ffmpeg full stream decode test requiring actual decoded PCM bytes
    decode_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(audio_path),
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        retcode, pcm_data, _ = await run_subprocess_safe(decode_cmd, timeout_sec)
    except TimeoutError as err:
        msg = "Audio decoding timed out"
        raise ValueError(msg) from err

    if retcode != 0:
        msg = "Audio verification failed: stream corrupt or undecodable"
        raise ValueError(msg)

    if len(pcm_data) == 0:
        msg = "Audio verification failed: zero decoded samples"
        raise ValueError(msg)


def sanitize_text(text: str, secrets: list[str] | None = None) -> str:
    """Scrub sensitive endpoints, domains, and tokens from error messages and logs."""
    if not text:
        return ""
    sanitized = text
    if secrets:
        for secret in secrets:
            if secret and secret in sanitized:
                sanitized = sanitized.replace(secret, "[REDACTED]")
    return sanitized


def get_required_env(key: str) -> str:
    """Retrieve a required environment variable or fail fast."""
    val = os.environ.get(key)
    if not val:
        msg = f"{key} environment variable is required"
        raise RuntimeError(msg)
    return val


async def verify_session(
    page: Page,
    base_url: str | None = None,
    expected_user: str | None = None,  # noqa: ARG001
) -> None:
    """Verify that the browser page is on the expected origin and exact review route.

    Note on authentication:
    URL and route matching validates target origin and navigation state, but does
    not constitute proof of reviewer authentication. Offline inspection cannot
    reliably verify reviewer identity selectors or window attributes without live
    inspection. We rely on the attached browser session already being authenticated,
    and explicitly record that reviewer identity remains unverified.
    """
    if base_url is None:
        base_url = get_required_env("BASE_URL")

    parsed_page = urllib.parse.urlsplit(page.url)
    parsed_base = urllib.parse.urlsplit(base_url)

    page_origin = f"{parsed_page.scheme}://{parsed_page.netloc}"
    expected_origin = f"{parsed_base.scheme}://{parsed_base.netloc}"

    if (
        not parsed_page.scheme
        or not parsed_page.netloc
        or page_origin != expected_origin
    ):
        msg = "Browser page origin does not match expected review origin"
        raise RuntimeError(msg)

    # Validate exact review route (/review or /review/)
    page_path = parsed_page.path.rstrip("/")
    if page_path != "/review":
        msg = "Browser page path does not match exact review route"
        raise RuntimeError(msg)

    # Document unverified reviewer identity without logging confidential values
    logger.warning(
        "Reviewer identity verification deferred (unverified offline); "
        "relying on attached authenticated browser session"
    )
    logger.info("Session verified on exact review route successfully")


async def fetch_all_listing_items(
    page: Page,
    round_id: str,
    user_trans: str,
    page_size: int = 100,
) -> list[ListingItem]:
    """Fetch matching review items via DataTables backend API with pagination."""
    all_items: list[ListingItem] = []
    seen_ids: set[str] = set()
    start = 0
    total_records = None
    draw = 1

    logger.info("Fetching listing items via backend DataTables API...")

    while True:
        try:
            data = await page.evaluate(
                """async (paramsObj) => {
                    const params = new URLSearchParams(paramsObj);
                    const res = await fetch('/review/listing', {
                        method: 'POST',
                        headers: {
                            'Content-Type':
                                'application/x-www-form-urlencoded; charset=UTF-8',
                            'X-Requested-With': 'XMLHttpRequest'
                        },
                        body: params.toString()
                    });
                    if (!res.ok) {
                        throw new Error(
                            `Listing API returned HTTP ${res.status}: ${res.statusText}`
                        );
                    }
                    return await res.json();
                }""",
                {
                    "draw": str(draw),
                    "start": str(start),
                    "length": str(page_size),
                    "userTrans": user_trans,
                    "round": round_id,
                    "status": APPROVED_STATUS,
                    "keyword": "",
                },
            )
        except Exception:  # noqa: BLE001
            msg = "Listing API request failed"
            raise RuntimeError(msg) from None

        draw += 1

        try:
            parsed = ListingResponse.model_validate(data)
        except ValidationError:
            msg = "Invalid listing response schema"
            raise RuntimeError(msg) from None

        if total_records is None:
            total_records = parsed.recordsFiltered
            logger.info(
                "Total matching records reported by backend: %d",
                total_records,
            )
        elif parsed.recordsFiltered != total_records:
            msg = "Listing response recordsFiltered changed during pagination"
            raise RuntimeError(msg)

        page_items = parsed.data
        if not page_items:
            break

        for item in page_items:
            if item.id in seen_ids:
                msg = "Duplicate record ID encountered in pagination"
                raise RuntimeError(msg)

            seen_ids.add(item.id)
            all_items.append(item)

        logger.info(
            "Retrieved %d unique items (offset=%d, page_items=%d)",
            len(all_items),
            start,
            len(page_items),
        )

        start += len(page_items)
        if (total_records is not None and start >= total_records) or len(
            page_items
        ) == 0:
            break

    if total_records is not None and len(all_items) != total_records:
        msg = "Incomplete listing: record count mismatch"
        raise RuntimeError(msg)

    logger.info(
        "Finished listing: %d items ready for processing",
        len(all_items),
    )
    return all_items


async def fetch_audio_base64(
    page: Page,
    audio_url: str,
    max_retries: int = 3,
) -> str:
    """Fetch audio bytes via in-browser context and return as base64 string."""
    js_code = """async (url) => {
        const res = await fetch(url);
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        const blob = await res.blob();
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const dataUrl = reader.result;
                const base64 = dataUrl.split(',')[1];
                resolve(base64);
            };
            reader.onerror = () => reject(new Error('FileReader failed'));
            reader.readAsDataURL(blob);
        });
    }"""

    for attempt in range(1, max_retries + 1):
        try:
            return await page.evaluate(js_code, audio_url)
        except Exception:
            if attempt == max_retries:
                raise
            wait_sec = 2**attempt
            logger.warning(
                "Audio fetch failed (attempt %d/%d). Retrying in %ds...",
                attempt,
                max_retries,
                wait_sec,
            )
            await asyncio.sleep(wait_sec)

    msg = f"Failed to fetch audio after {max_retries} attempts"
    raise RuntimeError(msg)


def validate_wav_bytes(data: bytes) -> bool:
    """Validate WAV magic header bytes."""
    return (
        len(data) >= MIN_WAV_HEADER_LEN
        and data[:4] == b"RIFF"
        and data[8:MIN_WAV_HEADER_LEN] == b"WAVE"
    )


def is_item_complete(
    item_dir: Path,
    item: ListingItem | dict,
    ext: str = "wav",
) -> bool:
    """Check if verified, unchanged audio, annotation, and manifest exist."""
    audio_path = item_dir / f"audio.{ext}"
    anno_path = item_dir / "annotation.json"
    manifest_path = item_dir / "manifest.json"

    if (
        not item_dir.is_dir()
        or item_dir.is_symlink()
        or not audio_path.is_file()
        or audio_path.is_symlink()
        or not anno_path.is_file()
        or anno_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        return False

    try:
        listing_item = (
            item if isinstance(item, ListingItem) else ListingItem.model_validate(item)
        )
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = CompletionManifest.model_validate_json(manifest_text)

        if manifest.source_fingerprint != compute_source_fingerprint(listing_item):
            return False

        audio_bytes = audio_path.read_bytes()
        if hashlib.sha256(audio_bytes).hexdigest() != manifest.audio_sha256:
            return False

        anno_text = anno_path.read_text(encoding="utf-8")
        anno_hash = hashlib.sha256(anno_text.encode("utf-8")).hexdigest()
        if anno_hash != manifest.annotation_sha256:
            return False

        json.loads(anno_text)
        return True
    except (UnicodeDecodeError, ValidationError, json.JSONDecodeError, OSError):
        return False


def _write_temp_audio(
    item_dir: Path,
    ext: str,
    audio_bytes: bytes,
) -> Path:
    """Create secure temp audio file synchronously."""
    if item_dir.is_symlink():
        msg = "Symlinked item directory rejected"
        raise RuntimeError(msg)

    item_dir.mkdir(parents=True, exist_ok=True)
    if item_dir.is_symlink():
        msg = "Symlinked item directory rejected"
        raise RuntimeError(msg)

    with tempfile.NamedTemporaryFile(
        dir=item_dir, prefix=".tmp_audio_", suffix=f".{ext}", delete=False
    ) as f:
        f.write(audio_bytes)
        f.flush()
        os.fsync(f.fileno())
        return Path(f.name)


def _commit_item_files(
    item_dir: Path,
    ext: str,
    tmp_audio: Path,
    audio_bytes: bytes,
    item: ListingItem,
) -> None:
    """Atomically commit annotation, audio, and manifest files."""
    anno_text = item.content
    tmp_files: list[Path] = [tmp_audio]
    try:
        # Write secure temp annotation
        with tempfile.NamedTemporaryFile(
            dir=item_dir, prefix=".tmp_anno_", suffix=".json", delete=False
        ) as f_anno:
            f_anno.write(anno_text.encode("utf-8"))
            f_anno.flush()
            os.fsync(f_anno.fileno())
            tmp_anno = Path(f_anno.name)
            tmp_files.append(tmp_anno)

        # Write secure temp manifest
        manifest = CompletionManifest(
            source_fingerprint=compute_source_fingerprint(item),
            audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
            annotation_sha256=hashlib.sha256(anno_text.encode("utf-8")).hexdigest(),
        )
        manifest_bytes = manifest.model_dump_json(indent=2).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            dir=item_dir, prefix=".tmp_manifest_", suffix=".json", delete=False
        ) as f_manifest:
            f_manifest.write(manifest_bytes)
            f_manifest.flush()
            os.fsync(f_manifest.fileno())
            tmp_manifest = Path(f_manifest.name)
            tmp_files.append(tmp_manifest)

        # Invalidate previous marker before replacing files
        manifest_path = item_dir / "manifest.json"
        manifest_path.unlink(missing_ok=True)

        # Per-file atomic replace
        final_audio = item_dir / f"audio.{ext}"
        final_anno = item_dir / "annotation.json"
        tmp_audio.replace(final_audio)
        tmp_anno.replace(final_anno)

        # Commit new marker last
        tmp_manifest.replace(manifest_path)
    finally:
        for p in tmp_files:
            if p.exists():
                p.unlink(missing_ok=True)


async def save_item_files(
    item_dir: Path,
    ext: str,
    audio_bytes: bytes,
    anno_text: str,  # noqa: ARG001
    item: ListingItem,
) -> None:
    """Save audio, annotation, and manifest files atomically with verification."""
    tmp_audio = await asyncio.to_thread(_write_temp_audio, item_dir, ext, audio_bytes)
    try:
        await verify_audio_file(tmp_audio, ext)
        await asyncio.to_thread(
            _commit_item_files,
            item_dir,
            ext,
            tmp_audio,
            audio_bytes,
            item,
        )
    finally:
        if tmp_audio.exists():
            tmp_audio.unlink(missing_ok=True)


def _is_safe_item_dir(item_dir: Path, output_dir: Path) -> bool:
    """Check directory safety synchronously."""
    if item_dir.is_symlink():
        return False
    try:
        return item_dir.resolve().is_relative_to(output_dir.resolve())
    except OSError:
        return False


async def process_item(  # noqa: PLR0911, PLR0912, PLR0913
    item_data: dict | ListingItem,
    page: Page,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    *,
    base_url: str | None = None,
    secret_tokens: list[str] | None = None,  # noqa: ARG001
) -> tuple[str, str, str | None]:
    """Download audio and annotation for a single item."""
    if base_url is None:
        base_url = get_required_env("BASE_URL")
    base_url = base_url.rstrip("/")

    # Validate item schema via Pydantic at the boundary
    try:
        if isinstance(item_data, ListingItem):
            item = item_data
        else:
            item = ListingItem.model_validate(item_data)
    except ValidationError:
        raw_id = (
            str(item_data.get("id", "")).strip() if isinstance(item_data, dict) else ""
        )
        safe_id = raw_id if SAFE_ID_PATTERN.match(raw_id) else ""
        return (safe_id, "failed", "Invalid listing record schema")

    ma_cau = item.id
    item_dir = output_dir / ma_cau

    # Validate path safety
    is_safe_dir = await asyncio.to_thread(_is_safe_item_dir, item_dir, output_dir)
    if not is_safe_dir:
        return (ma_cau, "failed", "Unsafe item directory path rejected")

    # Audio extension
    ext = Path(item.path).suffix.lstrip(".").lower()
    if not ext:
        return (ma_cau, "failed", "Audio path missing extension")

    # Idempotency check: skip if complete and verified
    is_complete = await asyncio.to_thread(is_item_complete, item_dir, item, ext)
    if is_complete:
        return (ma_cau, "skipped", None)

    # Validate audio URL
    is_safe, audio_url = is_safe_audio_url(item.path, base_url)
    if not is_safe:
        return (ma_cau, "failed", "Unsafe audio URL rejected")

    async with semaphore:
        try:
            b64_data = await fetch_audio_base64(page, audio_url)
            try:
                audio_bytes = base64.b64decode(b64_data)
            except binascii.Error:
                return (ma_cau, "failed", "Audio decoding failed: invalid base64")

            if len(audio_bytes) == 0:
                return (ma_cau, "failed", "Audio verification failed: empty download")

            await save_item_files(
                item_dir,
                ext,
                audio_bytes,
                item.content,
                item,
            )
            return (ma_cau, "downloaded", None)
        except ValueError:
            return (ma_cau, "failed", "Audio verification failed")
        except (RustwrightError, TimeoutError, OSError):
            return (ma_cau, "failed", "Failed to download or persist item files")
        except Exception:  # noqa: BLE001
            return (ma_cau, "failed", "Unexpected failure processing item")


async def run_crawler(  # noqa: PLR0913
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    concurrency: int = DEFAULT_CONCURRENCY,
    *,
    round_id: str,
    user_trans: str,
    limit: int | None = None,
    cdp_url: str | None = None,
    base_url: str | None = None,
    expected_user: str | None = None,
) -> tuple[int, int, int]:
    """Execute the data collection pipeline."""
    if cdp_url is None:
        cdp_url = get_required_env("CDP_WS_URL")
    if base_url is None:
        base_url = get_required_env("BASE_URL")
    if expected_user is None:
        expected_user = get_required_env("REVIEW_USER")

    # Build secret tokens for redaction
    parsed_base = urllib.parse.urlsplit(base_url)
    secret_tokens = [base_url, parsed_base.netloc, cdp_url, expected_user]
    if parsed_base.hostname:
        secret_tokens.append(parsed_base.hostname)

    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    logger.info("Connecting to browser over CDP...")
    async with async_playwright() as p:
        try:
            browser = await p.chromium.connect_over_cdp(cdp_url)
        except Exception:  # noqa: BLE001
            msg = "Failed to connect to browser session over CDP"
            raise RuntimeError(msg) from None

        context = browser.contexts[0]

        # Locate review page on exact /review route
        target_page = None
        for page in context.pages:
            parsed = urllib.parse.urlsplit(page.url)
            if parsed.path.rstrip("/") == "/review":
                target_page = page
                break

        if not target_page:
            # Fall back to first page in attached context
            target_page = context.pages[0]

        await verify_session(
            target_page,
            base_url=base_url,
            expected_user=expected_user,
        )

        # Retrieve items
        items = await fetch_all_listing_items(
            target_page,
            round_id=round_id,
            user_trans=user_trans,
        )
        if limit is not None:
            logger.info("Applying limit: %d items", limit)
            items = items[:limit]

        logger.info(
            "Processing %d items with concurrency=%d...",
            len(items),
            concurrency,
        )

        tasks = [
            process_item(
                item,
                target_page,
                output_dir,
                semaphore,
                base_url=base_url,
                secret_tokens=secret_tokens,
            )
            for item in items
        ]
        results = await asyncio.gather(*tasks)

        downloaded = sum(1 for _, s, _ in results if s == "downloaded")
        skipped = sum(1 for _, s, _ in results if s == "skipped")
        failed = sum(1 for _, s, _ in results if s == "failed")

        logger.info("=" * 50)
        logger.info("COLLECTION COMPLETED")
        logger.info("Total items:      %d", len(items))
        logger.info("Newly downloaded: %d", downloaded)
        logger.info("Skipped (cached): %d", skipped)
        logger.info("Failed:           %d", failed)
        logger.info("=" * 50)

        if failed > 0:
            logger.warning("Failed items:")
            for ma_cau, s, err in results:
                if s == "failed":
                    logger.warning("  Item %s: %s", ma_cau, err)

        return (downloaded, skipped, failed)


app = typer.Typer(
    add_completion=False,
    help="Download reviewed audio dataset via Rustwright",
)


@app.command()
def crawl(
    round_id: Annotated[
        str,
        typer.Option(
            "--round",
            help="Review round ID (required)",
        ),
    ],
    user_trans: Annotated[
        str,
        typer.Option(
            "--user-trans",
            help="Review user trans ID (required)",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory to save downloaded dataset (default: data/bronze)",
        ),
    ] = DEFAULT_OUTPUT_DIR,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            help="Number of concurrent audio downloads (default: 4)",
        ),
    ] = DEFAULT_CONCURRENCY,
    *,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            help="Optional limit on number of items to download",
        ),
    ] = None,
) -> None:
    """Download dataset from review session using CDP connection."""
    clean_round = round_id.strip()
    if not clean_round:
        msg = "round must be a non-empty string"
        raise typer.BadParameter(
            msg,
            param_hint="--round",
        )

    clean_user_trans = user_trans.strip()
    if not clean_user_trans:
        msg = "user-trans must be a non-empty string"
        raise typer.BadParameter(
            msg,
            param_hint="--user-trans",
        )

    if concurrency < 1:
        msg = "concurrency must be at least 1"
        raise typer.BadParameter(
            msg,
            param_hint="--concurrency",
        )

    if limit is not None and limit < 1:
        msg = "limit must be at least 1"
        raise typer.BadParameter(
            msg,
            param_hint="--limit",
        )

    try:
        _downloaded, _skipped, failed = asyncio.run(
            run_crawler(
                output_dir=output_dir,
                concurrency=concurrency,
                limit=limit,
                round_id=clean_round,
                user_trans=clean_user_trans,
            )
        )
    except RuntimeError as e:
        logger.error("Crawler execution failed: %s", str(e))
        raise typer.Exit(code=1) from None

    if failed > 0:
        raise typer.Exit(code=1)


def main() -> None:
    """Entrypoint for crawler CLI."""
    app()


if __name__ == "__main__":
    main()
