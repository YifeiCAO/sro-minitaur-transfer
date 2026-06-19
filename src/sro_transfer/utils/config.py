"""Tiny YAML config loader with ``${a.b}`` interpolation.

Avoids a hard dependency on OmegaConf so the diagnostics half of the repo runs
with nothing but pandas/numpy/scikit-learn/pyyaml.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTERP = re.compile(r"\$\{([^}]+)\}")


def _get(d: dict, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _interp(obj: Any, root: dict) -> Any:
    """Resolve ${a.b.c} references against the fully-parsed config tree."""
    if isinstance(obj, str):
        # repeat until no references remain (handles nested ${} chains)
        for _ in range(10):
            m = _INTERP.search(obj)
            if not m:
                break
            obj = obj[: m.start()] + str(_get(root, m.group(1))) + obj[m.end():]
        return obj
    if isinstance(obj, dict):
        return {k: _interp(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interp(v, root) for v in obj]
    return obj


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _interp(cfg, cfg)


def load_tasks(path: str | Path = "configs/tasks.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
