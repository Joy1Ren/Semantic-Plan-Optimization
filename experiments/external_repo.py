"""Resolve the external benchmark checkouts this repo reads from, without vendoring them.

Both benchmark repos are used in place rather than copied, so their evaluators stay the
single source of truth for how a benchmark is scored. Neither is pip-installable for our
purposes:

  - SemBench ships no packaging metadata at all, and its modules import each other as
    top-level packages (``scenario``, ``evaluator``, ``runner``) rooted at ``src/``.
  - docetl *is* packaged, but its build only includes ``docetl/**`` and ``server/**``
    (see its pyproject ``[tool.hatch.build]``), so the CUAD scorer we call --
    ``experiments.reasoning.evaluation.cuad`` -- is not in the installed distribution.

So each benchmark declares, in its ``benchmark.yaml``, where its checkout lives and which
subdirectory belongs on ``sys.path``:

    external_repo:
      env_var: SEMBENCH_ROOT     # optional override of the default location
      default: ../SemBench       # relative to this repository's root (a sibling checkout)
      python_path: src           # subdir to put on sys.path ('.' for the repo root)

This is the only place that puts an *external repository* on ``sys.path``. (The scripts under
``analysis/`` still self-bootstrap the package root so they can be run directly from any
working directory; packaging this repo would remove that.)

External checkouts are read-only inputs: everything this repo writes goes under
``results/`` (see each benchmark's ``results_prefix``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from agent_cost_model.paths import PACKAGE_ROOT


def resolve_repo(config: dict[str, Any] | None) -> Path | None:
    """Return the external checkout's root, or None when a benchmark declares no repo."""
    if not config:
        return None
    default = config.get("default", "")
    env_var = config.get("env_var")
    raw = (os.environ.get(env_var) if env_var else None) or default
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        # Relative to this repository's root, so `../SemBench` means a sibling checkout.
        path = PACKAGE_ROOT / path
    return path.resolve()


def add_repo_to_syspath(root: Path, python_path: str = ".") -> Path:
    """Put ``root/python_path`` on sys.path so the benchmark's own modules import."""
    target = (root / python_path).resolve()
    entry = str(target)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return target


def activate(config: dict[str, Any] | None) -> Path | None:
    """Resolve a benchmark's external checkout and make its modules importable.

    Returns the repo root (not the sys.path entry) so callers can build paths from it.
    """
    root = resolve_repo(config)
    if root is None:
        return None
    add_repo_to_syspath(root, config.get("python_path", "."))
    return root
