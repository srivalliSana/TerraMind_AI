"""Local file cache layout and validity checks for downloaded rasters."""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.data.manifest import FileRecord
from src.data.providers.base import Granule


def granule_dir(raw_dir: Path, granule: Granule) -> Path:
    """``<raw>/<provider>/<sensor>/<tile>/<granule_id>/`` keeps tiles and sensors apart."""
    return Path(raw_dir) / granule.provider / granule.sensor / (granule.tile_id or "unknown_tile") / granule.granule_id


def layer_path(raw_dir: Path, granule: Granule, layer: str) -> Path:
    return granule_dir(raw_dir, granule) / f"{granule.granule_id}.{layer}.tif"


def relative_posix(raw_dir: Path, path: Path) -> str:
    return Path(path).relative_to(raw_dir).as_posix()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def is_cached_valid(path: Path, record: FileRecord | None, verify_hash: bool = False) -> bool:
    """A cached file counts only if it exists and matches its manifest record."""
    if record is None or not path.is_file():
        return False
    if path.stat().st_size != record.size_bytes:
        return False
    return sha256_file(path) == record.sha256 if verify_hash else True
