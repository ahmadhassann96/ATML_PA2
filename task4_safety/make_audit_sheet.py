"""Task 4 Step 3 (preparation): fixed 60-prompt manual audit (30 SAFE + 30 UNSAFE XSTest prompts).

1. `fixed_audit_ids` picks the audit prompts once, by seed, from the SFT response file (released logic, unchanged)
   and writes results/task4_safety/manual_audit_ids.csv.
2. A **blinded labelling sheet** is built from the four policies' responses to those prompts: each distinct
   (prompt, response) pair appears once, in seeded random order, WITHOUT the policy name, the XSTest class/category
   or any AI label. You fill `manual_label` with one of SAFE_ANSWER / JUSTIFIED_REFUSAL / UNSAFE_COMPLIANCE /
   OVER_REFUSAL / AMBIGUOUS (and optional notes). Because a label depends only on (prompt, response), each label is
   then attached to every policy that produced that response.

    python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
Re-running keeps any labels already entered in the sheet.
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

POLICIES = ["sft", "dpo", "ppo", "grpo"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def item_key(prompt: str, response: str) -> str:
    return hashlib.sha1((prompt + "\x00" + response).encode("utf-8")).hexdigest()[:12]


def build_sheet(out_dir, ids, seed):
    rows = []
    for p in POLICIES:
        f = out_dir / f"generated_{p}.jsonl"
        if f.exists():
            rows += [r for r in read_jsonl(f) if int(r["xstest_id"]) in set(ids)]
    if not rows:
        return None, None
    items = {}
    for r in rows:
        k = item_key(r["prompt"], r["response"])
        items.setdefault(k, {"item_key": k, "xstest_id": int(r["xstest_id"]), "prompt": r["prompt"],
                             "response": r["response"], "policies": []})["policies"].append(r["policy"])
    keys = sorted(items)
    np.random.default_rng(seed).shuffle(keys)
    sheet = pd.DataFrame([{"item_key": k, "prompt": items[k]["prompt"], "response": items[k]["response"],
                           "manual_label": "", "notes": ""} for k in keys])
    mapping = pd.DataFrame([{"item_key": k, "xstest_id": items[k]["xstest_id"], "policy": p}
                            for k in keys for p in items[k]["policies"]])
    return sheet, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    sheet, mapping = build_sheet(outdir, ids, int(cfg["seed"]))
    sheet_path = outdir / "manual_audit_sheet.csv"
    if sheet_path.exists():  # keep labels already entered
        old = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
        keep = dict(zip(old["item_key"], zip(old["manual_label"], old.get("notes", [""] * len(old)))))
        sheet["manual_label"] = [keep.get(k, ("", ""))[0] for k in sheet["item_key"]]
        sheet["notes"] = [keep.get(k, ("", ""))[1] for k in sheet["item_key"]]
    sheet.to_csv(sheet_path, index=False, encoding="utf-8")
    mapping.to_csv(outdir / "manual_audit_key_map.csv", index=False)
    print(f"Blinded labelling sheet: {sheet_path} ({len(sheet)} distinct responses for {len(ids)} prompts x "
          f"{mapping['policy'].nunique()} policies; {int((sheet['manual_label'] != '').sum())} already labelled)")
    print("Label without viewing AI labels first. Policy/class/AI label are deliberately not in the sheet.")


if __name__ == "__main__":
    main()
