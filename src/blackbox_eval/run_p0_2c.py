# -*- coding: utf-8 -*-
"""Phase 0.2-C runner: Representation Invariance Test.
For each (task, rep) in {t1_solar, t3_ev} x {chart, text, table, info}:
  64 states x [EAQ whole-answer attribution (gen) + A_f1 support (forward)].
Resume-safe JSONL: results/p0_2/rep_{engine}.jsonl
"""
import argparse, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))

from p0_data import TASKS, FACTS, states  # noqa: E402
from rep_variants import build_blocks_rep, REPS, CARRIER_TAG, build_assets  # noqa: E402
from engines import load_engine  # noqa: E402
from run_p0_2a import parse_scores, EAQ_SYSTEM, EAQ_SUFFIX  # noqa: E402

JOBS = [("t1_solar", "f1"), ("t3_ev", "f1")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    ap.add_argument("--reps", default=",".join(REPS))
    ap.add_argument("--jobs", default=",".join(j[0] for j in JOBS))
    args = ap.parse_args()

    build_assets()
    out_dir = os.path.join(ROOT, "results", "p0_2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rep_{args.engine}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["task"], r["rep"], tuple(r["state"])))
                except Exception:
                    pass
    print(f"[{args.engine}] REP resume: {len(done)} present", flush=True)

    eng = load_engine(args.engine)
    t0 = time.time()
    n_new = n_fail = 0
    for tk in args.jobs.split(","):
        t = TASKS[tk]
        fact = FACTS[tk][0]  # f1
        ref = t["refs"][0]
        for rep in args.reps.split(","):
            for st in states():
                if (tk, rep, tuple(st)) in done:
                    continue
                blocks = build_blocks_rep(tk, rep, list(st))
                rec = {"engine": args.engine, "task": tk, "rep": rep, "state": list(st)}
                try:
                    txt, dt = eng.generate(blocks, t["query"], max_new_tokens=120,
                                           system=EAQ_SYSTEM,
                                           suffix=EAQ_SUFFIX.format(ans=ref))
                    p, ok, missing = parse_scores(txt)
                    a_j = eng.score(blocks, t["query"], fact["statement"])
                    rec.update({"p": p, "parse_ok": ok, "A_j": a_j,
                                "dt": round(dt, 2), "t": round(time.time() - t0, 1)})
                    if not ok:
                        n_fail += 1
                except Exception as e:  # noqa: BLE001
                    rec.update({"error": f"{type(e).__name__}: {e}",
                                "t": round(time.time() - t0, 1)})
                    n_fail += 1
                    print(f"  ERR {tk} {rep} {st}: {rec['error'][:90]}", flush=True)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done.add((tk, rep, tuple(st)))
                n_new += 1
                if n_new % 40 == 0:
                    print(f"  {tk} {rep} #{n_new} elapsed={rec['t']}s", flush=True)
    print(f"[{args.engine}] REP DONE. new={n_new} parse_fail={n_fail} "
          f"wall={time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
