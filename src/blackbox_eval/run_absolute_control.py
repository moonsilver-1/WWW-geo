# -*- coding: utf-8 -*-
"""Explicit absolute-score RealWeb control; see results/absolute_control/protocol.md.

Run from any cwd: python src/blackbox_eval/run_absolute_control.py --engine qwen3b
Writes append-only JSONL in results/absolute_control. Resume validates completed keys.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))

from p0_data import states  # noqa: E402
from realweb_data import SPEC  # noqa: E402
from engines import load_engine  # noqa: E402
from run_realweb import build_blocks_real, neutralize_blocks  # noqa: E402
from run_p0_2a import parse_scores  # noqa: E402

ABS_SYSTEM = (
    "You are a precise research assistant. You will be given a web page as a sequence of "
    "tagged evidence blocks, a question, and a candidate answer. Score the absolute "
    "contribution of each evidence block to supporting the candidate answer in this "
    "current page configuration."
)
ABS_SUFFIX = (
    "Candidate answer: {ans}\n\n"
    "For each block [EV-1] through [EV-6], give an absolute contribution score from 0.0 "
    "to 1.0 for how much that block supports this answer in the current page configuration. "
    "Judge each block on its own scale. Scores are independent: several blocks may all "
    "score highly, and the six scores do not need to sum to 1.0. Do not divide a fixed "
    "pool of credit among blocks. Other blocks may change the context in which a block "
    "is useful; reflect that only when it changes this block's actual support. Output "
    "only one JSON object with exactly EV-1, EV-2, EV-3, EV-4, EV-5, and EV-6 as keys "
    "and numeric values."
)

# A second, deliberately terse ablation removes the contextual-dependence cue
# from the first absolute prompt while retaining its non-normalization rule.
NOCTX_SYSTEM = (
    "You are a precise research assistant. You will be given a web page as a sequence of "
    "tagged evidence blocks, a question, and a candidate answer. Score the absolute "
    "contribution of each evidence block to supporting the candidate answer."
)
NOCTX_SUFFIX = (
    "Candidate answer: {ans}\n\n"
    "For each block [EV-1] through [EV-6], give an absolute contribution score from 0.0 "
    "to 1.0 for how much that block supports this answer. Judge each block on its own "
    "scale. Scores are independent: several blocks may all score highly, and the six "
    "scores do not need to sum to 1.0. Do not divide a fixed pool of credit among "
    "blocks. Output only one JSON object with exactly EV-1, EV-2, EV-3, EV-4, EV-5, "
    "and EV-6 as keys and numeric values."
)


def archived_support(engine):
    path = os.path.join(ROOT, "results", "realweb", f"eaq_{engine}.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    lookup = {(r["page"], tuple(r["state"])): r["A_j"] for r in rows if "A_j" in r}
    assert len(lookup) == len(SPEC) * 64, (path, len(lookup))
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    ap.add_argument("--pages", default=",".join(SPEC))
    ap.add_argument("--limit-states", type=int, default=0)
    ap.add_argument("--variant", choices=["context", "nocontext"], default="context")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "results", "absolute_control")
    os.makedirs(out_dir, exist_ok=True)
    stem = "abs" if args.variant == "context" else "abs_nocontext"
    out_path = os.path.join(out_dir, f"{stem}_{args.engine}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("parse_ok") is True:
                    done.add((r["page"], tuple(r["state"])))
    support = archived_support(args.engine)
    work = [(pid, st) for pid in args.pages.split(",") for st in states()[:args.limit_states or None]
            if (pid, tuple(st)) not in done]
    print(f"[{args.engine}] absolute control: {len(work)} states remaining", flush=True)
    if not work:
        return
    eng = load_engine(args.engine)
    t0 = time.time()
    for k, (pid, st) in enumerate(work, 1):
        key = (pid, tuple(st))
        rec = {"engine": args.engine, "checkpoint": eng.repo, "page": pid,
               "state": list(st), "prompt": stem, "A_j": support[key]}
        try:
            blocks = neutralize_blocks(build_blocks_real(pid, list(st)), list(st))
            system = ABS_SYSTEM if args.variant == "context" else NOCTX_SYSTEM
            suffix = ABS_SUFFIX if args.variant == "context" else NOCTX_SUFFIX
            txt, dt = eng.generate(blocks, SPEC[pid]["query"], max_new_tokens=120,
                                   system=system,
                                   suffix=suffix.format(ans=SPEC[pid]["ref_answer"]))
            p, ok, missing = parse_scores(txt)
            rec.update({"p": p, "parse_ok": ok, "missing": missing,
                        "gen": txt, "dt": round(dt, 2)})
        except Exception as exc:
            rec.update({"parse_ok": False, "error": f"{type(exc).__name__}: {exc}"})
        rec["elapsed"] = round(time.time() - t0, 1)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if k == 1 or k % 16 == 0 or k == len(work):
            print(f"[{args.engine}] {k}/{len(work)} {pid} parsed={rec['parse_ok']} "
                  f"elapsed={rec['elapsed']}s", flush=True)


if __name__ == "__main__":
    main()
