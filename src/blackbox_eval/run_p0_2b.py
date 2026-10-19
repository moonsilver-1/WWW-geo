# -*- coding: utf-8 -*-
"""Phase 0.2-B: Answer Attribution Separation Test.

For each atomic fact f_j (2 per task) and each evidence state x, TWO readouts of the
SAME statement from the SAME engine:
  SUPPORT   A_j(x)  = mean token logprob of f_j given (page blocks, question)   [forward]
  ATTRIB    C_A^{(j)}(x) = per-block contribution scores p_i^{(j)} in [0,1]       [EAQ-style gen]
Plus deterministic repeats on 8 states for measurement-noise estimation.
H tests: H1 support genuinely low-order / H2 teacher-forcing artifact / H3 task complexity.
Resume-safe JSONL: results/p0_2/sep_{engine}.jsonl
"""
import argparse, json, os, re, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))

from p0_data import TASKS, FACTS, build_blocks, states, STAT_SYSTEM, STAT_SUFFIX  # noqa: E402
from engines import load_engine  # noqa: E402
from run_p0_2a import parse_scores  # noqa: E402

RERUN_STATES = [tuple([1] * 6)] + [  # all-present + 3 fixed pseudo-random states
    tuple(1 if ((7 * k + 3) >> i) & 1 else -1 for i in range(6)) for k in range(1, 4)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    ap.add_argument("--tasks", default=",".join(TASKS.keys()))
    ap.add_argument("--limit-states", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "results", "p0_2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sep_{args.engine}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["task"], tuple(r["state"]), r["fact"], r.get("rerun", 0)))
                except Exception:
                    pass
    print(f"[{args.engine}] SEP resume: {len(done)} present", flush=True)

    eng = load_engine(args.engine)
    t0 = time.time()
    n_new = n_fail = 0
    for tk in args.tasks.split(","):
        t = TASKS[tk]
        sts = states()
        if args.limit_states:
            sts = sts[:args.limit_states]
        work = [(tuple(s), 0) for s in sts] + [(s, 1) for s in RERUN_STATES]
        for st, rk in work:
            for fact in FACTS[tk]:
                if (tk, st, fact["id"], rk) in done:
                    continue
                blocks = build_blocks(tk, list(st), version=t.get("version", "A"))
                rec = {"engine": args.engine, "task": tk, "state": list(st),
                       "fact": fact["id"], "rerun": rk}
                try:
                    a_j = eng.score(blocks, t["query"], fact["statement"])
                    txt, dt = eng.generate(
                        blocks, t["query"], max_new_tokens=120,
                        system=STAT_SYSTEM, suffix=STAT_SUFFIX.format(stmt=fact["statement"]))
                    p, ok, missing = parse_scores(txt)
                    rec.update({"A_j": a_j, "p": p, "parse_ok": ok, "missing": missing,
                                "gen": txt[:200], "dt": round(dt, 2),
                                "t": round(time.time() - t0, 1)})
                    if not ok:
                        n_fail += 1
                except Exception as e:  # noqa: BLE001
                    rec.update({"error": f"{type(e).__name__}: {e}",
                                "t": round(time.time() - t0, 1)})
                    n_fail += 1
                    print(f"  ERR {tk} {st} {fact['id']}: {rec['error'][:90]}", flush=True)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done.add((tk, st, fact["id"], rk))
                n_new += 1
                if n_new % 40 == 0:
                    print(f"  {tk} {fact['id']} #{n_new} A_j={rec.get('A_j')} "
                          f"elapsed={rec['t']}s", flush=True)
    print(f"[{args.engine}] SEP DONE. new={n_new} fail={n_fail} "
          f"wall={time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
