"""Wall-clock stamps and the per-run error log, shared by every agent in a run.

The state lives at module level because the failure sites that most need logging
(plan_tools, the exploration checker) hold no reference to the agent, and run_opt.py
runs one query per process. `start_run` resets it; everything else is a no-op until
then, so a sub-agent or a test used on its own still works.
"""

from __future__ import annotations

import pathlib
import time
import traceback
from typing import Any

_start: float = 0.0
_step: int | str = 0
_path: pathlib.Path | None = None


def start_run(path: str | pathlib.Path, label: str) -> float:
    """Begin a run: reset the elapsed origin, truncate the error log, return the start time."""
    global _start, _step, _path
    _start = time.time()
    _step = 0
    _path = pathlib.Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(f"=== run start {time.strftime('%Y-%m-%d %H:%M:%S')} | {label} ===\n")
    return _start


def set_step(step: int | str) -> None:
    global _step
    _step = step


def _elapsed() -> str:
    if not _start:
        return "+??:??"
    secs = int(time.time() - _start)
    hrs, mins, secs = secs // 3600, (secs // 60) % 60, secs % 60
    return f"+{hrs}:{mins:02d}:{secs:02d}" if hrs else f"+{mins:02d}:{secs:02d}"


def stamp() -> str:
    """e.g. `14:22:31 | +07:29`; the clock alone when no run has been started."""
    if not _start:
        return time.strftime("%H:%M:%S")
    return f"{time.strftime('%H:%M:%S')} | {_elapsed()}"


def header(kind: str, label: str) -> str:
    """One step divider, e.g. `--- assistant (step 7) --- [14:22:31 | +07:29]`."""
    return f"\n--- {kind} ({label}) --- [{stamp()}]\n"


def log_error(
    kind: str,
    message: str,
    *,
    content: Any = None,
    exc: BaseException | None = None,
    step: int | str | None = None,
) -> None:
    """Append one error block to the run's error log.

    `content` is whatever caused it -- the code block, the unparseable reply, a plan name.
    Written regardless of `verbose`: that flag gates the console, not this file.
    """
    if _path is None:
        return
    at = _step if step is None else step
    parts = [
        f"\n--- {kind} | step {at} | {time.strftime('%Y-%m-%d %H:%M:%S')} | {_elapsed()} ---",
        message,
    ]
    if content is not None:
        parts.append(f"[action]\n{content}")
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        parts.append(f"[traceback]\n{tb.rstrip()}")
    try:
        with _path.open("a") as f:
            f.write("\n".join(parts) + "\n")
    except OSError as e:
        print(f"[run_log] could not write the error log: {type(e).__name__}: {e}")
