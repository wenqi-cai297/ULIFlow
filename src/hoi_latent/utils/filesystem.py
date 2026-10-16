from __future__ import annotations

from pathlib import Path
from typing import Optional


def ensure_dir(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def read_text(path: str | Path, default: Optional[str] = None) -> Optional[str]:
    text_path = Path(path)
    if not text_path.exists():
        return default
    return text_path.read_text(encoding="utf-8", errors="replace")


def safe_relpath(path: str | Path, root: str | Path) -> str:
    path_obj = Path(path).resolve()
    root_obj = Path(root).resolve()
    try:
        return str(path_obj.relative_to(root_obj))
    except ValueError:
        return str(path_obj)
