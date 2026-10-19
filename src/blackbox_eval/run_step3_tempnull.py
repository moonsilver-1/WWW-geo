# -*- coding: utf-8 -*-
"""Step 3: Behavioral null (T=0.7, 3 seeds) for sampling variance estimation.
Minimal per user spec: 2 controlled + 2 real pages, 16-state L16 design, 3 seeds, 2 engines.
Goal: measure seed variance of EAQ p_i on the SAME state, to compare with interaction effect size.
Output: results/tempnull/seed_{engine}.jsonl
"""
import argparse, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))

from p0_data import states, neutral_text  # noqa: E402
from run_p0_2a import parse_scores, EAQ_SYSTEM, EAQ_SUFFIX  # noqa: E402
from realweb_data import SPEC, page_units  # noqa: E402
from run_realweb import build_blocks_real, neutralize_blocks  # noqa: E402
from engines import load_engine  # noqa: E402

JOBS = [
    ("ctrl", "t1_solar", None),
    ("ctrl", "t3_ev", None),
    ("real", "rw_jwst", None),
    ("real", "rw_hubble", None),
]
SEEDS = [0, 1, 2]
L16 = [tuple(1 if (v >> (5 - i)) & 1 else -1 for i in range(6))
       for v in range(16)]

def get_meta(job_kind, key):
    if job_kind == "ctrl":
        from p0_data import TASKS, FACTS
        t = TASKS[key]
        return {"query": t["query"], "statement": FACTS[key][0]["statement"],
                "ref_answer": t["refs"][0]}
    else:
        s = SPEC[key]
        return {"query": s["query"], "statement": s["statement"], "ref_answer": s["ref_answer"]}

def build_blocks_ctrl(tk, state):
    from p0_data import build_blocks, TASKS
    return build_blocks(tk, state, version=TASKS[tk].get("version", "A"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "results", "tempnull")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"seed_{args.engine}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["kind"], r["page"], tuple(r["state"]), r["seed"]))
                except Exception:
                    pass
    print(f"[{args.engine}] TEMPNULL resume: {len(done)} present", flush=True)

    eng = load_engine(args.engine)
    t0 = time.time()
    n_new = n_fail = 0
    for kind, key, _ in JOBS:
        meta = get_meta(kind, key)
        for st in L16:
            for seed in SEEDS:
                if (kind, key, tuple(st), seed) in done:
                    continue
                rec = {"engine": args.engine, "kind": kind, "page": key,
                       "state": list(st), "seed": seed}
                try:
                    if kind == "ctrl":
                        blocks = build_blocks_ctrl(key, list(st))
                    else:
                        blocks = neutralize_blocks(build_blocks_real(key, list(st)), list(st))
                    txt, dt = eng.generate(blocks, meta["query"], max_new_tokens=120,
                                           system=EAQ_SYSTEM,
                                           suffix=EAQ_SUFFIX.format(ans=meta["ref_answer"]),
                                           do_sample=True, temperature=0.7, top_p=0.9)
                    p, ok, missing = parse_scores(txt)
                    rec.update({"p": p, "parse_ok": ok, "dt": round(dt, 2),
                                "t": round(time.time() - t0, 1)})
                    if not ok:
                        n_fail += 1
                except Exception as e:  # noqa: BLE001
                    rec.update({"error": f"{type(e).__name__}: {e}", "t": round(time.time() - t0, 1)})
                    n_fail += 1
                    print(f"  ERR {kind}/{key} s{seed} {st}: {rec['error'][:90]}", flush=True)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done.add((kind, key, tuple(st), seed))
                n_new += 1
                if n_new % 20 == 0:
                    print(f"  {kind}/{key} #{n_new} elapsed={rec['t']}s", flush=True)
    print(f"[{args.engine}] TEMPNULL DONE. new={n_new} parse_fail={n_fail} "
          f"wall={time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
