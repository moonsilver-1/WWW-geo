# -*- coding: utf-8 -*-
"""RealWeb-6 pilot spec: 6 real public pages (Wikipedia REST HTML, offline assets).
n=6 evidence units per page; roles: fact / context / image / caption / adjacent / distractor.
Unit texts are pulled VERBATIM from data/realweb/{pid}/blocks.json (no transcription)."""
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RW = os.path.join(ROOT, "data", "realweb")

SPEC = {
    "rw_jwst": {
        "domain": "technology",
        "query": "When was the James Webb Space Telescope launched?",
        "statement": "The James Webb Space Telescope was launched on 25 December 2021.",
        "ref_answer": "The James Webb Space Telescope was launched on 25 December 2021.",
        "units": [
            {"role": "fact", "para": 2, "cut": ","},      # launch sentence
            {"role": "context", "para": 1},               # mirror/infrared
            {"role": "image", "image": 1},
            {"role": "caption", "caption": 1},            # sunshield test unit
            {"role": "adjacent", "para": 3},              # NASA/ESA/CSA
            {"role": "distractor", "para": 4},            # naming
        ],
    },
    "rw_hubble": {
        "domain": "technology",
        "query": "What is the diameter of the Hubble Space Telescope's main mirror?",
        "statement": "The Hubble Space Telescope features a 2.4 m primary mirror.",
        "ref_answer": "The Hubble Space Telescope features a 2.4 m (7 ft 10 in) primary mirror.",
        "units": [
            {"role": "fact", "para": 1},
            {"role": "context", "para": 0},
            {"role": "image", "image": 1},
            {"role": "caption", "caption": 0},            # Lyman Spitzer
            {"role": "adjacent", "para": 2},              # aberration
            {"role": "distractor", "para": 3},            # Great Observatories
        ],
    },
    "rw_iphone15": {
        "domain": "product",
        "query": "Which connector change did the iPhone 15 introduce compared to earlier iPhones?",
        "statement": "The iPhone 15 was the first iPhone to replace the Lightning connector with USB-C.",
        "ref_answer": "The iPhone 15 replaced Apple's proprietary Lightning connector with USB-C.",
        "units": [
            {"role": "fact", "para": 1},
            {"role": "context", "para": 0},
            {"role": "image", "image": 1},
            {"role": "caption", "caption": 0},            # Face and back
            {"role": "adjacent", "para": 2},              # 6GB RAM / switch
            {"role": "distractor", "para": 3},            # discontinued 2025
        ],
    },
    "rw_ps5": {
        "domain": "product",
        "query": "What storage hardware feature distinguishes the PlayStation 5?",
        "statement": "The PlayStation 5 includes a custom solid-state drive designed for high-speed data streaming.",
        "ref_answer": "The PlayStation 5 includes a custom solid-state drive for high-speed data streaming.",
        "units": [
            {"role": "fact", "para": 2},
            {"role": "context", "para": 0},
            {"role": "image", "image": 1},
            {"role": "caption", "caption": 0},            # Mark Cerny
            {"role": "adjacent", "para": 1},              # disc drive / Digital Edition
            {"role": "distractor", "para": 3},            # Cerny feedback cycle
        ],
    },
    "rw_mrna": {
        "domain": "health",
        "query": "When did the first mRNA vaccines obtain authorization, and by which developers?",
        "statement": "In December 2020, Pfizer-BioNTech and Moderna obtained authorization for their mRNA vaccines.",
        "ref_answer": "Pfizer-BioNTech and Moderna obtained authorization for their mRNA vaccines in December 2020.",
        "units": [
            {"role": "fact", "para": 2},                  # Dec 2020 authorization
            {"role": "context", "para": 0},
            {"role": "image", "image": 1},
            {"role": "caption", "caption": 0},            # in vitro transcription diagram
            {"role": "adjacent", "para": 3},              # 1989 transfection
            {"role": "distractor", "para": 4},            # 1993 mice
        ],
    },
    "rw_vitaminc": {
        "domain": "health",
        "query": "When was vitamin C discovered and first chemically produced?",
        "statement": "Vitamin C was discovered in 1912 and was first chemically produced in 1933.",
        "ref_answer": "Vitamin C was discovered in 1912 and first chemically produced in 1933.",
        "units": [
            {"role": "fact", "para": 2},
            {"role": "context", "para": 0},
            {"role": "image", "image": 1},
            {"role": "caption", "caption": 1},            # biosynthesis in plants
            {"role": "adjacent", "para": 1},              # scurvy therapy
            {"role": "distractor", "para": 3},            # enantiomers
        ],
    },
}

MAX_CHARS = 420

def _para(pid, idx, cut=None):
    bl = json.load(open(os.path.join(RW, pid, "blocks.json"), encoding="utf-8"))
    t = bl["paras"][idx]
    if cut:
        t = t.split(cut)[0].rstrip(",; ")
    return t[:MAX_CHARS]

def _caption(pid, idx):
    bl = json.load(open(os.path.join(RW, pid, "blocks.json"), encoding="utf-8"))
    return bl["captions"][idx][:200]

def _image(pid, k):
    fs = json.load(open(os.path.join(RW, "fetch_summary.json"), encoding="utf-8"))
    return os.path.join(RW, pid, fs[pid]["images"][k - 1]["file"])

def page_units(pid):
    """Return 6 unit dicts: {tag, type, text/asset_path, role}."""
    out = []
    for i, u in enumerate(SPEC[pid]["units"], start=1):
        if u["role"] == "image":
            out.append({"tag": i, "type": "I", "asset": _image(pid, u["image"]), "role": u["role"]})
        elif u["role"] == "caption":
            out.append({"tag": i, "type": "T", "caption": True, "text": _caption(pid, u["caption"]), "role": u["role"]})
        else:
            out.append({"tag": i, "type": "T", "text": _para(pid, u["para"], u.get("cut")), "role": u["role"]})
    return out

if __name__ == "__main__":
    for pid in SPEC:
        us = page_units(pid)
        print("=" * 16, pid)
        for u in us:
            print(f"  EV-{u['tag']} [{u['role']}/{u['type']}]", (u.get('text') or u['asset'])[:90])
