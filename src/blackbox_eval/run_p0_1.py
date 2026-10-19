# -*- coding: utf-8 -*-
"""P0-1 gate runner: 4 tasks x 64 neutralize states x 1 greedy generation + 2x teacher-forced scoring.
Resume-safe: appends one JSON line per (engine, task, state); existing keys skipped on restart.
Also runs a small rerun protocol (2 states per task, generated twice) for determinism checks."""
import argparse, json, os, re, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))

from p0_data import TASKS, build_blocks, gold_set, states  # noqa: E402
from engines import load_engine  # noqa: E402

CIT = re.compile(r"\[EV-?(\d+)\]", re.IGNORECASE)

def parse_citations(text):
    tags = sorted({int(m) for m in CIT.findall(text)})
    c = [1 if (i + 1) in tags else 0 for i in range(6)]
    return c, tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    ap.add_argument("--tasks", default=",".join(TASKS.keys()))
    ap.add_argument("--limit-states", type=int, default=0, help="smoke test: only first N states")
    ap.add_argument("--skip-reruns", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "results", "p0_1")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.engine}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["task"], tuple(r["state"]), r.get("rerun", 0)))
                except Exception:
                    pass
    print(f"[{args.engine}] resume: {len(done)} records already present", flush=True)

    eng = load_engine(args.engine)
    task_keys = args.tasks.split(",")
    all_states = states()
    if args.limit_states:
        all_states = all_states[:args.limit_states]

    # rerun protocol states: all-present + neutralize both gold units
    def rerun_states(task_key):
        g = gold_set(task_key)
        s2 = [1] * 6
        for gi in g:
            s2[gi - 1] = -1
        return [tuple([1] * 6), tuple(s2)]

    t_start = time.time()
    n_new = 0
    for tk in task_keys:
        t = TASKS[tk]
        work = [(tuple(s), 0) for s in all_states]
        if not args.skip_reruns:
            work += [(s, k) for s in rerun_states(tk) for k in (1, 2)]
        for st, rk in work:
            if (tk, st, rk) in done:
                continue
            state = list(st)
            blocks = build_blocks(tk, state, version=t.get("version", "A"))
            g = gold_set(tk, t.get("version", "A"))
            try:
                gen_txt, dt_gen = eng.generate(blocks, t["query"])
                u_scores = [eng.score(blocks, t["query"], ref) for ref in t["refs"]]
                c, tags = parse_citations(gen_txt)
                n_cited = sum(c)
                inter = sum(1 for i in g if c[i - 1] == 1)
                u_r = inter / len(g)
                u_p = inter / (n_cited + 1e-6)
                rec = {
                    "engine": args.engine, "task": tk, "state": state, "rerun": rk,
                    "gen": gen_txt, "c": c, "tags": tags,
                    "n_cited": n_cited, "insufficient": "INSUFFICIENT EVIDENCE" in gen_txt.upper(),
                    "U_ans": u_scores, "U_R": u_r, "U_P": u_p, "gold": g, "dt_gen": round(dt_gen, 2),
                    "t": round(time.time() - t_start, 1),
                }
            except Exception as e:  # noqa: BLE001 - log and continue; runner must survive single failures
                rec = {"engine": args.engine, "task": tk, "state": state, "rerun": rk,
                       "error": f"{type(e).__name__}: {e}", "t": round(time.time() - t_start, 1)}
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add((tk, st, rk))
            n_new += 1
            if rec.get("error"):
                print(f"  ERR {tk} {st} rerun{rk}: {rec['error'][:100]}", flush=True)
            elif n_new % 10 == 0 or rk:
                print(f"  {tk} state#{all_states.index(state) if rk == 0 else 'R' + str(rk)} "
                      f"cites={rec['tags']} U_ans={rec['U_ans'][0]:.3f} dt={rec['dt_gen']}s "
                      f"elapsed={rec['t']}s", flush=True)
    print(f"[{args.engine}] DONE. new={n_new} total={len(done)} "
          f"wall={time.time() - t_start:.0f}s", flush=True)

if __name__ == "__main__":
    main()
