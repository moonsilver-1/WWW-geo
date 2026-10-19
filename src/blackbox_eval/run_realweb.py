# -*- coding: utf-8 -*-
"""Step 2 runner: RealWeb-6 pilot. 6 pages x 64 states x [EAQ attribution gen + A_j support forward].
Same protocol as 0.2-A/0.2-B. Resume-safe: results/realweb/eaq_{engine}.jsonl"""
import argparse, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))

from p0_data import states, neutral_text  # noqa: E402
from run_p0_2a import parse_scores, EAQ_SYSTEM, EAQ_SUFFIX  # noqa: E402
from realweb_data import SPEC, page_units  # noqa: E402
from engines import load_engine  # noqa: E402

def build_blocks_real(pid, state):
    units = page_units(pid)
    blocks = []
    for i, u in enumerate(units):
        present = state[i] == 1
        b = {"tag": u["tag"], "type": u["type"], "caption": bool(u.get("caption")), "role": u["role"]}
        if u["type"] == "I":
            b["image"] = u["asset"]
        else:
            b["text"] = u["text"]  # neutralize handled below
        blocks.append(b)
    return blocks

def neutralize_blocks(blocks, state):
    """Replace content of neutralized units with info-neutral controls (slot preserved)."""
    from p0_data import neutral_text
    from PIL import Image, ImageFilter
    out = []
    for i, b in enumerate(blocks):
        if state[i] == 1:
            out.append(b)
            continue
        b2 = dict(b)
        if b["type"] == "I":
            im = Image.open(b["image"]).filter(ImageFilter.GaussianBlur(12))
            import os as _os
            p = _os.path.splitext(b["image"])[0] + "_neutral.png"
            im.save(p)
            b2["image"] = p
        else:
            base = b["text"][:400]
            b2["text"] = neutral_text(base if len(base) > 60 else base + " " * (60 - len(base)))
        out.append(b2)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    ap.add_argument("--pages", default=",".join(SPEC.keys()))
    ap.add_argument("--limit-states", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "results", "realweb")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eaq_{args.engine}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["page"], tuple(r["state"])))
                except Exception:
                    pass
    print(f"[{args.engine}] REALWEB resume: {len(done)} present", flush=True)

    eng = load_engine(args.engine)
    t0 = time.time()
    n_new = n_fail = 0
    for pid in args.pages.split(","):
        spec = SPEC[pid]
        sts = states()
        if args.limit_states:
            sts = sts[:args.limit_states]
        for st in sts:
            if (pid, tuple(st)) in done:
                continue
            blocks = neutralize_blocks(build_blocks_real(pid, list(st)), list(st))
            rec = {"engine": args.engine, "page": pid, "state": list(st)}
            try:
                txt, dt = eng.generate(blocks, spec["query"], max_new_tokens=120,
                                       system=EAQ_SYSTEM, suffix=EAQ_SUFFIX.format(ans=spec["ref_answer"]))
                p, ok, missing = parse_scores(txt)
                a_j = eng.score(blocks, spec["query"], spec["statement"])
                rec.update({"p": p, "parse_ok": ok, "A_j": a_j, "dt": round(dt, 2),
                            "t": round(time.time() - t0, 1)})
                if not ok:
                    n_fail += 1
            except Exception as e:  # noqa: BLE001
                rec.update({"error": f"{type(e).__name__}: {e}", "t": round(time.time() - t0, 1)})
                n_fail += 1
                print(f"  ERR {pid} {st}: {rec['error'][:90]}", flush=True)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add((pid, tuple(st)))
            n_new += 1
            if n_new % 30 == 0:
                print(f"  {pid} #{n_new} elapsed={rec['t']}s", flush=True)
    print(f"[{args.engine}] REALWEB DONE. new={n_new} parse_fail={n_fail} "
          f"wall={time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
