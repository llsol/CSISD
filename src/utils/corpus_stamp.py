"""
Corpus version stamping for derived artefacts and model runs.

Usage — write stamp alongside a parquet:
    from src.utils.corpus_stamp import write_stamp, check_stamp
    write_stamp(out_path)          # writes out_path.stem + ".meta.json"
    check_stamp(parquet_path)      # warns if corpus_id mismatch

Usage — embed in model run config dict:
    from src.utils.corpus_stamp import corpus_meta
    config = {**dataclasses.asdict(model_cfg), **corpus_meta()}
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import settings as S

def _gt_patterns() -> list[str]:
    return [
        "annotations/*_ann_pitch_svara.tsv",
        f"raw/*_pitch_{S.PITCH_SOURCE}.tsv",
    ]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _corpus_hash() -> str:
    """MD5 (12 hex chars) of all GT annotation + pitch source files."""
    h = hashlib.md5()
    paths: list[Path] = []
    for rec_id in sorted(S.RECORDING_SELECTION):
        rec_dir = S.DATA_CORPUS / rec_id
        for pat in _gt_patterns():
            paths.extend(sorted(rec_dir.glob(pat)))
    for p in paths:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def corpus_meta() -> dict:
    """Return dict with corpus provenance fields."""
    return {
        "corpus_id":      S.CORPUS_ID,
        "corpus_hash":    _corpus_hash(),
        "corpus_git_sha": _git_sha(),
        "corpus_date":    datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def write_stamp(parquet_path: Path) -> Path:
    """Write <parquet_path>.meta.json next to the parquet."""
    meta_path = parquet_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(corpus_meta(), indent=2))
    return meta_path


def check_stamp(parquet_path: Path, abort: bool = False) -> bool:
    """
    Warn (or raise) if the parquet's stamped corpus_id != settings.CORPUS_ID.
    Returns True if OK, False if mismatch.
    """
    meta_path = parquet_path.with_suffix(".meta.json")
    if not meta_path.exists():
        warnings.warn(
            f"[corpus_stamp] no stamp for {parquet_path.name} — "
            "run the analysis script to regenerate.",
            stacklevel=2,
        )
        return False

    meta         = json.loads(meta_path.read_text())
    stamped_id   = meta.get("corpus_id", "")
    stamped_hash = meta.get("corpus_hash", "")

    id_ok   = stamped_id == S.CORPUS_ID
    hash_ok = (not stamped_hash) or (stamped_hash == _corpus_hash())

    if not id_ok or not hash_ok:
        parts = []
        if not id_ok:
            parts.append(f"corpus_id '{stamped_id}' != '{S.CORPUS_ID}'")
        if not hash_ok:
            parts.append(f"corpus_hash mismatch (source files changed?)")
        msg = (
            f"[corpus_stamp] MISMATCH in {parquet_path.name}: "
            + "; ".join(parts)
            + ". Regenerate derived artefacts before training."
        )
        if abort:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)
        return False
    return True
