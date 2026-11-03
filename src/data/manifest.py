"""Download manifests: provenance records for every file fetched.

One JSON file per granule in ``data/manifests/``. A manifest answers "which exact
files, from which URL, with which checksum and catalog metadata produced this
result?", which is what makes outputs reproducible and auditable.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1


@dataclass
class FileRecord:
    layer: str
    url: str
    path: str                 # relative to the raw data directory, POSIX separators
    size_bytes: int
    sha256: str
    downloaded_at: str        # ISO-8601 UTC


@dataclass
class GranuleManifest:
    granule_id: str
    provider: str
    sensor: str
    tile_id: str | None
    acquired: str                                   # ISO-8601 UTC
    catalog_metadata: dict[str, Any] = field(default_factory=dict)  # scale, offset, fill, cloud cover ...
    files: dict[str, FileRecord] = field(default_factory=dict)      # layer -> record
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GranuleManifest":
        files = {k: FileRecord(**v) for k, v in data.get("files", {}).items()}
        return cls(**{**data, "files": files})


def manifest_path(manifest_dir: Path, granule_id: str) -> Path:
    return Path(manifest_dir) / f"{granule_id}.json"


def load_manifest(manifest_dir: Path, granule_id: str) -> GranuleManifest | None:
    """Return the stored manifest, or None if absent or unreadable (a corrupt manifest is ignored, not trusted)."""
    path = manifest_path(manifest_dir, granule_id)
    if not path.is_file():
        return None
    try:
        return GranuleManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        log.warning("Ignoring unreadable manifest %s: %s", path.name, exc)
        return None


def save_manifest(manifest_dir: Path, manifest: GranuleManifest) -> Path:
    """Atomically write the manifest (temp file + rename)."""
    path = manifest_path(manifest_dir, manifest.granule_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(manifest.to_json(), encoding="utf-8")
    os.replace(tmp, path)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
