"""Notebook helpers: run a task script as a subprocess and watch it live.

The task notebooks are thin drivers. Every experiment is still a plain `python -m taskX.script ...`
command (the reproducible pipeline listed in the README); `run_live` only launches that command,
streams its console output into the notebook, saves the full output to a log file, and redraws a
progress figure (built from the run's saved JSONL logs) every few seconds. Nothing depends on hidden
notebook state: re-running a cell re-reads everything from disk.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl_safe(path) -> list[dict]:
    """Read a JSONL file that another process may be appending to (ignores a partial last line)."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _fmt_elapsed(s: float) -> str:
    s = int(s)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60:d}m{s % 60:02d}s"


def run_live(module: str, *args, title: str | None = None, plot_fn=None, refresh_s: float = 15.0,
             tail: int = 18, log_file: str | Path | None = None) -> None:
    """Run `python -m module *args` from the repo root and show live progress in the notebook.

    plot_fn: optional zero-argument callable returning a matplotlib Figure (or None); called on every refresh.
    log_file: where to save the complete console output (default results/notebook_logs/<module>.log).
    Raises RuntimeError if the command fails. Interrupting the cell (stop button) kills the subprocess.
    """
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    cmd = [sys.executable, "-u", "-m", module, *map(str, args)]
    title = title or " ".join(cmd[3:])
    log_file = Path(log_file) if log_file else ROOT / "results" / "notebook_logs" / f"{module.replace('.', '_')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}

    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=env)
    lines: queue.Queue = queue.Queue()

    def reader():
        for line in proc.stdout:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    shown = deque(maxlen=tail)
    start, last_draw, done = time.time(), 0.0, False

    def draw(status: str):
        clear_output(wait=True)
        print(f"{title}\n{status}  |  elapsed {_fmt_elapsed(time.time() - start)}  |  full log: {log_file}\n")
        if plot_fn is not None:
            try:
                fig = plot_fn()
                if fig is not None:
                    display(fig)
                    plt.close(fig)
            except Exception as exc:  # a plotting hiccup must never kill the run
                print(f"(progress plot unavailable: {exc})")
        print("\n".join(shown))

    with log_file.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        try:
            while not done:
                try:
                    line = lines.get(timeout=0.5)
                except queue.Empty:
                    line = ""
                if line is None:
                    done = True
                elif line:
                    # drop noisy library warnings from the live view (they stay in the log file)
                    log.write(line + "\n")
                    if "Warning" not in line and not line.startswith("  warnings.warn"):
                        shown.append(line)
                if time.time() - last_draw >= refresh_s:
                    draw("RUNNING")
                    last_draw = time.time()
            proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
            draw("STOPPED by user")
            raise
    draw("FINISHED OK" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})")
    if proc.returncode != 0:
        raise RuntimeError(f"{title} failed with exit code {proc.returncode}; see {log_file}")
