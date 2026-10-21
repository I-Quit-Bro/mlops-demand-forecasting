"""Download the raw dataset to data.raw_path on a fresh machine.

CI/CD does not need this: pipelines train on the small committed daily file.
Use it for teammates or a new laptop. The file is checked against a SHA-256 checksum,
so a wrong or truncated download (for example an HTML warning page from a file-sharing
site) is rejected instead of silently breaking training.

CLI:  forecast-fetch-data [--url URL] [--sha256 HEX] [--force]
Only https:// URLs are accepted (file:// is allowed for local tests).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from forecast.config import load_config
from forecast.exception import ConfigError, DataIngestionError, DataValidationError, ForecastError
from forecast.logger import get_logger

log = get_logger(__name__)

CHUNK_BYTES = 1024 * 1024
ALLOWED_SCHEMES = {"https", "file"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_data(
    url: str, dest: Path, expected_sha256: str | None = None, force: bool = False
) -> Path:
    scheme = urlparse(url).scheme
    if scheme not in ALLOWED_SCHEMES:
        raise DataIngestionError("Only https URLs are supported", url=url)

    dest = Path(dest)
    expected = expected_sha256.lower() if expected_sha256 else None
    if dest.exists() and not force:
        if expected is None or sha256_of(dest) == expected:
            log.info("Dataset already present, skipping download: %s", dest)
            return dest
        log.warning("Existing file has a different checksum, downloading again: %s", dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
            while chunk := response.read(CHUNK_BYTES):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except (OSError, ValueError) as exc:  # URLError and timeouts are OSErrors
        partial.unlink(missing_ok=True)
        raise DataIngestionError("Download failed", url=url) from exc

    actual = digest.hexdigest()
    if expected and actual != expected:
        partial.unlink(missing_ok=True)
        raise DataValidationError(
            "Checksum mismatch: the download is not the expected file",
            expected=expected,
            actual=actual,
            hint="large files on some sharing sites return an HTML warning page instead of data",
        )
    if expected is None:
        log.warning("No sha256 configured, integrity not verified. Add sha256: %s to the config", actual)

    os.replace(partial, dest)  # atomic: dest is either missing or complete
    log.info("Downloaded %s (%.1f MB, sha256=%s)", dest, size / 1e6, actual)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the raw dataset to data.raw_path")
    parser.add_argument("--config", default=None)
    parser.add_argument("--url", default=None, help="Overrides data.source_url")
    parser.add_argument("--sha256", default=None, help="Overrides data.sha256")
    parser.add_argument("--force", action="store_true", help="Download even if the file exists")
    args = parser.parse_args(argv)
    try:
        data = load_config(args.config).data
        data = replace(data, source_url=args.url or data.source_url, sha256=args.sha256 or data.sha256)
        if not data.source_url:
            raise ConfigError("No source_url configured; set data.source_url or pass --url")
        fetch_data(data.source_url, data.raw_path, data.sha256, args.force)
    except ForecastError as exc:
        log.error("Could not fetch data: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
