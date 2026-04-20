"""Data manifest writer.

Records the exact list of raw MIMIC-IV files observed on disk (path,
size, SHA-256 prefix, mtime) so any pipeline output can be traced back
to a specific data snapshot. Written to ``reports/artifacts/data_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from aki.utils.paths import paths


def _file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def write_manifest(raw_dir: Path | None = None) -> Path:
    """Scan ``raw_dir`` recursively and write a manifest JSON."""
    raw_dir = raw_dir or paths.raw
    entries: list[dict] = []
    for p in sorted(raw_dir.rglob("*.csv.gz")):
        stat = p.stat()
        entries.append(
            {
                "path": str(p.relative_to(raw_dir)),
                "size_bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256_prefix": _file_hash(p),
            }
        )

    out_path = paths.artifacts / "data_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "raw_dir": str(raw_dir),
                "n_files": len(entries),
                "files": entries,
            },
            indent=2,
        )
    )
    logger.info(f"Data manifest → {out_path}  ({len(entries)} files)")
    return out_path
