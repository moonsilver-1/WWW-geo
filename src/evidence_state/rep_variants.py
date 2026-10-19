# -*- coding: utf-8 -*-
"""0.2-C Representation Invariance Test — data layer.

Same fact z, same query, same evidence quantity (6 units); ONLY the carrier unit's
representation changes: chart (original) / text / table / infographic image.
Each rep has its own information-neutral control (blank axes / filler / empty table / blur).
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from p0_data import TASKS, ASSET_DIR, neutral_text, _font, make_chart, _blank_axes, FILLER  # noqa

def _card(w, h, bg=(246, 244, 238)):
    img = Image.new("RGB", (w, h), bg)
    return img, ImageDraw.Draw(img)

def make_t1_info(path):
    img, d = _card(640, 420)
    d.text((320, 110), "FULL RECHARGE", font=_font(30), fill=(40, 44, 52), anchor="mm")
    d.text((320, 215), "≈ 4 hours", font=_font(64), fill=(183, 65, 14), anchor="mm")
    d.text((320, 305), "with a 60 W panel (full sun)", font=_font(22), fill=(70, 70, 70), anchor="mm")
    d.ellipse([288, 30, 352, 78], outline=(240, 180, 60), width=5)
    img.save(path)

def make_t3_info(path):
    img, d = _card(640, 420)
    d.text((320, 110), "PUBLIC CHARGERS", font=_font(30), fill=(40, 44, 52), anchor="mm")
    d.text((320, 215), "1,100", font=_font(72), fill=(42, 111, 151), anchor="mm")
    d.text((320, 305), "in operation at the end of 2024", font=_font(22), fill=(70, 70, 70), anchor="mm")
    img.save(path)

def _blur(src, dst, radius=12):
    Image.open(src).filter(ImageFilter.GaussianBlur(radius)).save(dst)

T1_TEXT = ("Charging tests under full sun: with a 60 W panel, the SunRay S200 fully "
           "recharges in about 4 hours. Recharge time depends on the panel wattage.")
T1_TABLE = "| Panel wattage | Full recharge time |\n|---|---|\n| 60 W | about 4 hours |"
T3_TEXT = ("Year-end totals from the transport authority: the network operated 1,100 "
           "public chargers at the end of 2024.")
T3_TABLE = "| Milestone | Public chargers |\n|---|---|\n| End of 2024 | 1,100 |"

T1_TABLE_N = "| Panel wattage | Full recharge time |\n|---|---|\n| | |"
T3_TABLE_N = "| Milestone | Public chargers |\n|---|---|\n| | |"

def build_assets():
    make_t1_info(f"{ASSET_DIR}/t1_fact_info.png")
    make_t3_info(f"{ASSET_DIR}/t3_fact_info.png")
    _blur(f"{ASSET_DIR}/t1_fact_info.png", f"{ASSET_DIR}/t1_fact_info_neutral.png")
    _blur(f"{ASSET_DIR}/t3_fact_info.png", f"{ASSET_DIR}/t3_fact_info_neutral.png")

# carrier definitions per (task, rep): replaces the original carrier unit (tag 4)
REPS = ["chart", "text", "table", "info"]
CARRIERS = {
    ("t1_solar", "chart"): {"type": "C", "asset": "t1_chart.png", "neutral_asset": "t1_chart_neutral.png"},
    ("t1_solar", "text"): {"type": "T", "content": T1_TEXT},
    ("t1_solar", "table"): {"type": "B", "content": T1_TABLE, "neutral": T1_TABLE_N},
    ("t1_solar", "info"): {"type": "I", "asset": "t1_fact_info.png", "neutral_asset": "t1_fact_info_neutral.png"},
    ("t3_ev", "chart"): {"type": "C", "asset": "t3_chart.png", "neutral_asset": "t3_chart_neutral.png"},
    ("t3_ev", "text"): {"type": "T", "content": T3_TEXT},
    ("t3_ev", "table"): {"type": "B", "content": T3_TABLE, "neutral": T3_TABLE_N},
    ("t3_ev", "info"): {"type": "I", "asset": "t3_fact_info.png", "neutral_asset": "t3_fact_info_neutral.png"},
}
CARRIER_TAG = {"t1_solar": 4, "t3_ev": 4}

def build_blocks_rep(task_key, rep, state):
    """state: 6 x {-1,+1}. Carrier unit replaced by the rep-specific definition."""
    t = TASKS[task_key]
    carrier = CARRIERS[(task_key, rep)]
    tag = CARRIER_TAG[task_key]
    blocks = []
    for i, u in enumerate(t["units"]):
        present = state[i] == 1
        if u["tag"] == tag:
            b = {"tag": tag, "type": carrier["type"], "caption": False}
            if carrier["type"] in ("I", "C"):
                b["image"] = os.path.join(ASSET_DIR,
                                          carrier["asset"] if present else carrier["neutral_asset"])
            else:
                if present:
                    b["text"] = carrier["content"]
                else:
                    b["text"] = carrier.get("neutral") or neutral_text(carrier["content"])
            blocks.append(b)
            continue
        b = {"tag": u["tag"], "type": u["type"], "caption": bool(u.get("caption"))}
        if u["type"] in ("I", "C"):
            b["image"] = os.path.join(ASSET_DIR, u["asset"] if present else u["neutral_asset"])
        else:
            b["text"] = (u["content"] if version_ok(u, present) else neutral_text(u["content"]))
        blocks.append(b)
    return blocks

def version_ok(u, present):
    return present

if __name__ == "__main__":
    build_assets()
    print("rep assets built:", [f for f in os.listdir(ASSET_DIR) if "fact_info" in f])
    b = build_blocks_rep("t1_solar", "info", [1, 1, 1, -1, 1, 1])
    for x in b:
        print(" ", x["tag"], x["type"], (x.get("text") or x.get("image", ""))[:60])
