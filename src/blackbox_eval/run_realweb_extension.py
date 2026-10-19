"""Run the fixed twelve-page offline extension with the short absolute prompt."""
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
from p0_data import states
from realweb_extension import SPEC, build_blocks
from engines import load_engine
from run_realweb import neutralize_blocks
from run_p0_2a import parse_scores
from run_absolute_control import NOCTX_SYSTEM, NOCTX_SUFFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    args = ap.parse_args()
    outdir = os.path.join(ROOT, "results", "realweb_extension")
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, f"extension_{args.engine}.jsonl")
    done = set()
    if os.path.exists(dest):
        for line in open(dest, encoding="utf-8"):
            row = json.loads(line)
            if "A_j" in row:
                done.add((row["page"], tuple(row["state"])))
    work = [(page, st) for page in SPEC for st in states() if (page, tuple(st)) not in done]
    print(f"[{args.engine}] extension: {len(work)} states remaining", flush=True)
    if not work:
        return
    eng = load_engine(args.engine)
    t0 = time.time()
    for i, (page, st) in enumerate(work, 1):
        spec = SPEC[page]
        rec = {"engine": args.engine, "checkpoint": eng.repo, "page": page,
               "state": list(st), "question": spec["question"], "statement": spec["statement"]}
        try:
            blocks = neutralize_blocks(build_blocks(page), list(st))
            gen, dt = eng.generate(blocks, spec["question"], max_new_tokens=120,
                                   system=NOCTX_SYSTEM,
                                   suffix=NOCTX_SUFFIX.format(ans=spec["statement"]))
            p, ok, missing = parse_scores(gen)
            a_j = eng.score(blocks, spec["question"], spec["statement"])
            rec.update({"gen": gen, "p": p, "parse_ok": ok, "missing": missing,
                        "A_j": a_j, "dt": round(dt, 2)})
        except Exception as exc:
            rec.update({"parse_ok": False, "error": f"{type(exc).__name__}: {exc}"})
        rec["elapsed"] = round(time.time()-t0, 1)
        with open(dest, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Bound CUDA cache growth without adding a synchronization to every state.
        if i % 16 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if i == 1 or i % 16 == 0 or i == len(work):
            print(f"[{args.engine}] {i}/{len(work)} {page} parsed={rec['parse_ok']} elapsed={rec['elapsed']}s", flush=True)


if __name__ == "__main__":
    main()
