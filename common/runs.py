"""Small helpers shared by the orchestration scripts of Tasks 1-3 (ablations and studies).

Each training/evaluation job runs in its own Python process so every condition starts from a
clean CUDA context (important on an 8 GB GPU) and failures are isolated.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys

from common.data import repo_path


def run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *map(str, args)]
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(repo_path(".")))


def adapter_exists(path: str) -> bool:
    return (repo_path(path) / "adapter_model.safetensors").exists()


def load_json_if(path):
    p = repo_path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def last_train_record(results_dir: str, name: str):
    p = repo_path(results_dir) / name / "train_log.jsonl"
    if not p.exists():
        return None
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def write_csv(path, rows: list[dict]) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p}")
