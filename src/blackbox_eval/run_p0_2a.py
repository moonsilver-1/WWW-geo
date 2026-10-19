# -*- coding: utf-8 -*-
"""Phase 0.2-A: Evidence Attribution Query (EAQ).

Unified attribution probe — NO citation tokens, NO free-text generation to parse.
For each evidence state x, the engine sees (page blocks, question, REFERENCE ANSWER)
and must output ONLY a JSON object scoring EVERY block's contribution to the answer:
  {"EV-1": p_1, ..., "EV-6": p_6},  p_i in [0,1].
C_i(x) = p_i is a continuous function on the cube -> directly Fourier-decomposable.

Resume-safe JSONL: results/p0_2/eaq_{engine}.jsonl
"""
import argparse, json, os, re, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))

from p0_data import TASKS, build_blocks, states  # noqa: E402
from engines import load_engine  # noqa: E402

EAQ_SYSTEM = ("You are a precise research assistant. You will be given a web page as a "
              "sequence of tagged evidence blocks, a question, and a candidate answer. "
              "Your task is to quantify how much each evidence block contributes to "
              "supporting that candidate answer.")

EAQ_SUFFIX = ("Candidate answer: {ans}\n\n"
              "Task: Score the contribution of EVERY evidence block [EV-1] through [EV-6] "
              "to this candidate answer. Give each block a number between 0.0 (no "
              "contribution to the answer) and 1.0 (the answer is fully supported by this "
              "block). Output ONLY a JSON object mapping tags to scores, for example: "
              "{{\"EV-1\": 0.2, \"EV-2\": 0.9, \"EV-3\": 0.0, \"EV-4\": 0.8, \"EV-5\": 0.7, "
              "\"EV-6\": 0.1}} — include all six tags, no other text.")

# EAQ-v2: forces graded scoring by instructing contextual dependency and exemplar spread.
# Used in Step 2 (R1) to test whether RealWeb lookup-mode collapse is fixable by prompt.
EAQ_V2_SYSTEM = ("You are a precise research assistant. You will be given a web page as a "
                  "sequence of tagged evidence blocks, a question, and a candidate answer. "
                  "Your task is to quantify how much each evidence block contributes to "
                  "supporting that candidate answer.")

EAQ_V2_SUFFIX = ("Candidate answer: {ans}\n\n"
                  "Task: Score the contribution of EVERY evidence block [EV-1] through [EV-6] "
                  "to this candidate answer. Use the FULL continuous scale in [0.0, 1.0] — "
                  "do NOT collapse scores to 0.0 or 1.0 unless truly warranted. A score of "
                  "0.7 means 'this block alone supports ~70% of the answer'; 0.3 means "
                  "'this block contributes ~30% context but is not the sole support'. "
                  "Scores must reflect context: removing or adding OTHER blocks may change "
                  "a block's contribution even when its own content is unchanged.\n\n"
                  "Calibration examples (do NOT copy literally — adapt to this query):\n"
                  "  - A block that DIRECTLY states the answer: ~0.6-0.9\n"
                  "  - A block that PROVIDES CONTEXT but not the fact: ~0.2-0.5\n"
                  "  - A block that is irrelevant to the answer: <0.1\n\n"
                  "Output ONLY a JSON object mapping all six tags to scores, e.g.: "
                  "{{\"EV-1\": 0.35, \"EV-2\": 0.85, \"EV-3\": 0.10, \"EV-4\": 0.62, "
                  "\"EV-5\": 0.20, \"EV-6\": 0.05}} — no other text.")

SCORE_RE = re.compile(r"EV-?(\d+)\"\s*:\s*([0-9]*\.?[0-9]+)")

def parse_scores(text):
    scores = {int(m.group(1)): float(m.group(2)) for m in SCORE_RE.finditer(text)}
    missing = [i for i in range(1, 7) if i not in scores]
    p = [min(max(scores.get(i + 1, 0.0), 0.0), 1.0) for i in range(6)]
    ok = len(missing) == 0 and len(scores) == 6
    return p, ok, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    ap.add_argument("--tasks", default=",".join(TASKS.keys()))
    ap.add_argument("--limit-states", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "results", "p0_2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eaq_{args.engine}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["task"], tuple(r["state"])))
                except Exception:
                    pass
    print(f"[{args.engine}] EAQ resume: {len(done)} present", flush=True)

    eng = load_engine(args.engine)
    t_start = time.time()
    n_new, n_fail = 0, 0
    for tk in args.tasks.split(","):
        t = TASKS[tk]
        ref = t["refs"][0]  # fixed estimand: primary reference answer
        sts = states()
        if args.limit_states:
            sts = sts[:args.limit_states]
        for st in sts:
            if (tk, tuple(st)) in done:
                continue
            blocks = build_blocks(tk, list(st), version=t.get("version", "A"))
            suffix = EAQ_SUFFIX.format(ans=ref)
            try:
                txt, dt = eng.generate(blocks, t["query"], max_new_tokens=120,
                                       system=EAQ_SYSTEM, suffix=suffix)
                p, ok, missing = parse_scores(txt)
                rec = {"engine": args.engine, "task": tk, "state": list(st), "p": p,
                       "parse_ok": ok, "missing": missing, "gen": txt[:300],
                       "dt": round(dt, 2), "t": round(time.time() - t_start, 1)}
                if not ok:
                    n_fail += 1
            except Exception as e:  # noqa: BLE001
                rec = {"engine": args.engine, "task": tk, "state": list(st),
                       "error": f"{type(e).__name__}: {e}", "t": round(time.time() - t_start, 1)}
                n_fail += 1
                print(f"  ERR {tk} {st}: {rec['error'][:100]}", flush=True)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add((tk, tuple(st)))
            n_new += 1
            if n_new % 20 == 0:
                print(f"  {tk} #{n_new} p={rec.get('p')} dt={rec.get('dt')} elapsed={rec['t']}s",
                      flush=True)
    print(f"[{args.engine}] EAQ DONE. new={n_new} parse_fail={n_fail} "
          f"wall={time.time() - t_start:.0f}s", flush=True)

if __name__ == "__main__":
    main()
