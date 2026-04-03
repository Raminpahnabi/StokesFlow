#!/usr/bin/env python3
"""Utilities for locating the local sweeps Python extension build."""

from __future__ import annotations

import importlib.machinery
import os
import sys
from pathlib import Path


def _candidate_api_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent
    env_candidates = [
        os.environ.get("SWEEPS_API_PATH"),
        os.environ.get("SWEEPSPATH"),
    ]

    relative_candidates = [
        here / "../sweeps/sweeps/build/src/api",
        here / "../sweeps/build/src/api",
        here / "sweepspath",
    ]

    candidates: list[Path] = []
    for raw_path in env_candidates:
        if raw_path:
            candidates.append(Path(raw_path).expanduser())
    candidates.extend(relative_candidates)

    sweeps_roots = [
        (here / "../sweeps/sweeps").resolve(),
        (here / "../sweeps").resolve(),
    ]
    for sweeps_root in sweeps_roots:
        if not sweeps_root.is_dir():
            continue
        for api_dir in sorted(sweeps_root.glob("build*/src/api")):
            candidates.append(api_dir)

    return candidates


def _contains_usable_extension(api_dir: Path) -> bool:
    return _extension_score(api_dir) >= 0


def _extension_score(api_dir: Path) -> int:
    if not api_dir.is_dir():
        return -1

    py_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    abi_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    module_prefixes = ("splines", "sweeps")

    best_score = -1
    for entry in api_dir.iterdir():
        if not entry.is_file() or not entry.name.startswith(module_prefixes):
            continue
        if py_tag in entry.name:
            best_score = max(best_score, 3)
        elif ".abi3" in entry.name and entry.name.endswith(abi_suffixes):
            best_score = max(best_score, 2)
        elif ".so" == entry.suffix and "cpython-" not in entry.name:
            best_score = max(best_score, 1)
    return best_score


def get_sweeps_api_path() -> str:
    """Return the first compatible local sweeps API directory."""
    candidates = []
    for candidate in _candidate_api_dirs():
        resolved = candidate.resolve()
        score = _extension_score(resolved)
        if score >= 0:
            candidates.append((score, resolved))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], str(item[1])))
        return str(candidates[0][1])

    searched = "\n".join(f"  - {candidate.resolve()}" for candidate in _candidate_api_dirs())
    raise ModuleNotFoundError(
        "Could not find a compatible sweeps API build for "
        f"Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "Set SWEEPS_API_PATH to the directory containing "
        "`splines*.so` or `sweeps*.so`, or build sweeps for this interpreter.\n"
        f"Searched:\n{searched}"
    )


def ensure_sweeps_api_on_path() -> str:
    """Insert the local sweeps API directory onto sys.path and return it."""
    api_path = get_sweeps_api_path()
    if api_path not in sys.path:
        sys.path.insert(0, api_path)
    return api_path
