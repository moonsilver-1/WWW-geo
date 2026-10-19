"""Prospectively specified second-fact experiment on the six offline pages."""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "evidence_state"))
from p0_data import states
from engines import load_engine
from run_realweb import build_blocks_real, neutralize_blocks
from run_p0_2a import parse_scores
from run_absolute_control import NOCTX_SYSTEM, NOCTX_SUFFIX

FACTS = {
    "rw_jwst": ("Which three agencies partnered in the development of the James Webb Space Telescope?",
                "NASA partnered with the European Space Agency and the Canadian Space Agency to develop the James Webb Space Telescope."),
    "rw_hubble": ("Which servicing mission corrected the Hubble telescope's optics, and in what year?",
                  "Hubble's optics were corrected by servicing mission STS-61 in 1993."),
    "rw_iphone15": ("How much RAM do the iPhone 15 and 15 Plus have?",
                    "The iPhone 15 and iPhone 15 Plus have 6 GB of RAM."),
    "rw_ps5": ("Which PlayStation 5 edition lacks an optical disc drive?",
               "The PlayStation 5 Digital Edition lacks an optical disc drive."),
    "rw_mrna": ("In what year was successful transfection of designed mRNA in a liposomal nanoparticle into a cell published?",
                "Successful transfection of designed mRNA in a liposomal nanoparticle into a cell was published in 1989."),
    "rw_vitaminc": ("Which deficiency disease is vitamin C used to prevent and treat?",
                    "Vitamin C is used to prevent and treat scurvy."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen3b", "internvl3-2b"])
    args = ap.parse_args()
    outdir = os.path.join(ROOT, "results", "altfact")
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, f"alt_{args.engine}.jsonl")
    done = set()
    if os.path.exists(dest):
        for line in open(dest, encoding="utf-8"):
            r = json.loads(line)
            if r.get("parse_ok") and "A_j" in r:
                done.add((r["page"], tuple(r["state"])))
    work = [(page, st) for page in FACTS for st in states() if (page, tuple(st)) not in done]
    print(f"[{args.engine}] second fact: {len(work)} states remaining", flush=True)
    if not work:
        return
    eng = load_engine(args.engine)
    t0 = time.time()
    for i, (page, st) in enumerate(work, 1):
        question, answer = FACTS[page]
        rec = {"engine": args.engine, "checkpoint": eng.repo, "page": page,
               "state": list(st), "question": question, "statement": answer}
        try:
            blocks = neutralize_blocks(build_blocks_real(page, list(st)), list(st))
            gen, dt = eng.generate(blocks, question, max_new_tokens=120,
                                   system=NOCTX_SYSTEM, suffix=NOCTX_SUFFIX.format(ans=answer))
            p, ok, missing = parse_scores(gen)
            support = eng.score(blocks, question, answer)
            rec.update({"gen": gen, "p": p, "parse_ok": ok, "missing": missing,
                        "A_j": support, "dt": round(dt, 2)})
        except Exception as exc:
            rec.update({"parse_ok": False, "error": f"{type(exc).__name__}: {exc}"})
        rec["elapsed"] = round(time.time()-t0, 1)
        with open(dest, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i == 1 or i % 16 == 0 or i == len(work):
            print(f"[{args.engine}] {i}/{len(work)} {page} parsed={rec['parse_ok']} elapsed={rec['elapsed']}s", flush=True)


if __name__ == "__main__":
    main()
