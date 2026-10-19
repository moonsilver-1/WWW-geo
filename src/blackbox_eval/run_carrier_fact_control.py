"""Matched atomic-fact carrier experiment on two controlled tasks."""
import argparse
import gc
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from p0_data import TASKS, FACTS, states
from rep_variants import build_blocks_rep, build_assets, REPS
from engines import load_engine
from run_p0_2a import parse_scores
from run_absolute_control import NOCTX_SYSTEM, NOCTX_SUFFIX

JOBS = ("t1_solar", "t3_ev")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    args = ap.parse_args()
    build_assets()
    outdir = os.path.join(ROOT, "results", "carrier_fact_control")
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, f"carrier_{args.engine}.jsonl")
    done = set()
    if os.path.exists(dest):
        for line in open(dest, encoding="utf-8"):
            rec = json.loads(line)
            if "A_j" in rec:
                done.add((rec["task"], rec["rep"], tuple(rec["state"])))
    work = [(task, rep, st) for task in JOBS for rep in REPS for st in states()
            if (task, rep, tuple(st)) not in done]
    print(f"[{args.engine}] matched carrier: {len(work)} states remaining", flush=True)
    if not work:
        return
    eng = load_engine(args.engine)
    t0 = time.time()
    for i, (task, rep, st) in enumerate(work, 1):
        query = TASKS[task]["query"]
        statement = FACTS[task][0]["statement"]
        rec = {"engine": args.engine, "checkpoint": eng.repo, "task": task, "rep": rep,
               "state": list(st), "question": query, "statement": statement}
        try:
            blocks = build_blocks_rep(task, rep, list(st))
            gen, dt = eng.generate(blocks, query, max_new_tokens=120,
                                   system=NOCTX_SYSTEM,
                                   suffix=NOCTX_SUFFIX.format(ans=statement))
            p, ok, missing = parse_scores(gen)
            a_j = eng.score(blocks, query, statement)
            rec.update({"gen": gen, "p": p, "parse_ok": ok, "missing": missing,
                        "A_j": a_j, "dt": round(dt, 2)})
        except Exception as exc:
            rec.update({"parse_ok": False, "error": f"{type(exc).__name__}: {exc}"})
        rec["elapsed"] = round(time.time() - t0, 1)
        with open(dest, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 16 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if i == 1 or i % 16 == 0 or i == len(work):
            print(f"[{args.engine}] {i}/{len(work)} {task}/{rep} parsed={rec['parse_ok']} elapsed={rec['elapsed']}s", flush=True)


if __name__ == "__main__":
    main()
