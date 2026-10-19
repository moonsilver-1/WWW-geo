# -*- coding: utf-8 -*-
"""P0-1 data layer: 4 tasks x 6 typed evidence units, neutralize/delete variants,
twin version B for task 4, gold supporting sets G(q), reference answers.

State convention: x in {-1,+1}^6, x_i=+1 unit present, x_i=-1 unit NEUTRALIZED
(info-neutral control preserving slot/layout/token budget)."""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSET_DIR = os.path.join(ROOT, "data", "evidence_units")
os.makedirs(ASSET_DIR, exist_ok=True)

FONT_PATHS = []
def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

# ---------------------------------------------------------------- text filler
FILLER = ("This section provides general background information about the general topic "
          "of this page and does not contain specific measurements, names, or specifications. ")
def neutral_text(s: str) -> str:
    n = max(len(s), 40)
    out = (FILLER * (n // len(FILLER) + 1))[:n]
    return out.rstrip()

# ---------------------------------------------------------------- image assets
def _card(w, h, bg=(246, 244, 238)):
    img = Image.new("RGB", (w, h), bg)
    return img, ImageDraw.Draw(img)

def make_t1_device(path):
    img, d = _card(640, 420)
    d.rounded_rectangle([200, 90, 440, 340], 18, fill=(52, 62, 78), outline=(30, 36, 46), width=4)
    d.rounded_rectangle([220, 110, 420, 190], 8, fill=(96, 125, 160))
    for i in range(4):
        d.line([226 + i * 48, 116, 226 + i * 48, 184], fill=(70, 95, 125), width=3)
    for (cx, label) in [(270, "USB-C"), (370, "USB-C")]:
        d.rounded_rectangle([cx - 34, 250, cx + 34, 278], 10, outline=(230, 230, 230), width=3)
        d.text((cx, 264), label, font=_font(15), fill=(235, 235, 235), anchor="mm")
    d.rounded_rectangle([300, 300, 340, 326], 6, outline=(230, 230, 230), width=3)
    d.text((320, 313), "DC", font=_font(14), fill=(235, 235, 235), anchor="mm")
    d.text((320, 62), "SunRay S200", font=_font(26), fill=(40, 44, 52), anchor="mm")
    img.save(path)

def make_t2_label(path):
    img, d = _card(640, 420, bg=(250, 248, 242))
    d.rectangle([150, 50, 490, 370], outline=(60, 60, 60), width=4)
    d.text((320, 95), "SENCHA", font=_font(34), fill=(30, 60, 40), anchor="mm")
    d.text((320, 145), "First Flush", font=_font(22), fill=(60, 60, 60), anchor="mm")
    d.ellipse([180, 185, 230, 215], fill=(90, 140, 90))
    d.ellipse([410, 185, 460, 215], fill=(90, 140, 90))
    d.text((320, 245), "Uji, Kyoto", font=_font(26), fill=(50, 50, 50), anchor="mm")
    d.text((320, 295), "Harvested: April 2025", font=_font(20), fill=(70, 70, 70), anchor="mm")
    d.text((320, 335), "Net 100 g", font=_font(18), fill=(90, 90, 90), anchor="mm")
    img.save(path)

def make_t3_station(path):
    img, d = _card(640, 420, bg=(232, 236, 240))
    d.rectangle([280, 110, 360, 360], fill=(70, 76, 88))
    d.rounded_rectangle([262, 70, 378, 200], 10, fill=(240, 240, 240), outline=(90, 90, 90), width=3)
    d.text((320, 110), "STATION 7", font=_font(24), fill=(30, 30, 30), anchor="mm")
    d.text((320, 155), "MAX 150 kW", font=_font(22), fill=(160, 60, 40), anchor="mm")
    d.circle = d.ellipse([296, 260, 344, 308], outline=(220, 220, 220), width=4)
    d.line([312, 284, 328, 284], fill=(220, 220, 220), width=4)
    d.rectangle([120, 360, 520, 376], fill=(120, 120, 120))
    img.save(path)

def make_t4_pump(path, with_voltage):
    img, d = _card(640, 420, bg=(238, 240, 236))
    d.ellipse([220, 150, 420, 350], fill=(80, 90, 104), outline=(50, 56, 66), width=5)
    d.ellipse([272, 202, 368, 298], fill=(120, 130, 146), outline=(60, 66, 78), width=3)
    d.rectangle([300, 90, 340, 150], fill=(100, 108, 122))
    d.text((320, 400 - 62), "INLINE PUMP", font=_font(24), fill=(45, 50, 60), anchor="mm")
    if with_voltage:
        d.rounded_rectangle([380, 120, 520, 170], 10, fill=(250, 220, 80), outline=(120, 110, 30), width=3)
        d.text((450, 145), "12V DC", font=_font(26), fill=(60, 50, 10), anchor="mm")
    img.save(path)

def _blur(src, dst, radius=12):
    Image.open(src).filter(ImageFilter.GaussianBlur(radius)).save(dst)

# ---------------------------------------------------------------- chart assets
def _blank_axes(path, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=100)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def make_chart(path, title, xlabel, ylabel, xs, ys, kind="line", highlight=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=100)
    if kind == "line":
        ax.plot(xs, ys, marker="o", color="#2a6f97", linewidth=2)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:g}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    else:
        highlight = highlight or []
        colors = ["#c1440e" if x in highlight else "#2a6f97" for x in xs]
        ax.bar([str(x) for x in xs], ys, color=colors)
        for i, y in enumerate(ys):
            ax.text(i, y, f"{y:g}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

# ---------------------------------------------------------------- build assets
def build_assets():
    a = ASSET_DIR
    made = {}
    make_t1_device(f"{a}/t1_device.png"); made["t1_device"] = 1
    make_chart(f"{a}/t1_chart.png", "Recharge time vs panel wattage (full sun)",
               "Panel wattage (W)", "Recharge time (h)",
               [30, 60, 100, 120], [7, 4, 2, 1.7], kind="line")
    make_t2_label(f"{a}/t2_label.png")
    make_chart(f"{a}/t2_chart.png", "EGCG content per 240 ml cup vs brewing time",
               "Brewing time (min)", "EGCG (mg)",
               [1, 2, 3, 4, 5], [25, 48, 70, 88, 100], kind="line")
    make_t3_station(f"{a}/t3_station.png")
    make_chart(f"{a}/t3_chart.png", "Public chargers by year", "Year", "Number of chargers",
               [2022, 2023, 2024, 2025], [600, 850, 1100, 1500], kind="bar")
    make_t4_pump(f"{a}/t4_pump_A.png", with_voltage=False)
    make_t4_pump(f"{a}/t4_pump_B.png", with_voltage=True)
    make_chart(f"{a}/t4_chart_A.png", "Target EC by crop", "Crop", "EC (mS/cm)",
               ["Basil", "Tomato", "Pepper"], [1.6, 2.5, 2.0], kind="bar")
    make_chart(f"{a}/t4_chart_B.png", "Target EC by crop", "Crop", "EC (mS/cm)",
               ["Lettuce", "Basil", "Tomato", "Pepper"], [1.2, 1.6, 2.5, 2.0], kind="bar",
               highlight=["Lettuce"])
    # neutral (info-neutral) variants: blurred images, blank axes charts
    for name in ["t1_device", "t2_label", "t3_station", "t4_pump_A"]:
        _blur(f"{a}/{name}.png", f"{a}/{name}_neutral.png")
    _blank_axes(f"{a}/t1_chart_neutral.png", "Panel wattage", "Recharge time")
    _blank_axes(f"{a}/t2_chart_neutral.png", "Brewing time", "EGCG")
    _blank_axes(f"{a}/t3_chart_neutral.png", "Year", "Number of chargers")
    _blank_axes(f"{a}/t4_chart_neutral.png", "Crop", "EC")
    return made

# ---------------------------------------------------------------- task table
T1_TABLE = ("| Spec | Value |\n|---|---|\n| Battery capacity | 200 Wh |\n"
            "| Max AC output | 100 W |\n| Weight | 2.3 kg |\n| Cycle life | 800 cycles |")
T1_TABLE_N = "| Spec | Value |\n|---|---|\n| | |\n| | |\n| | |\n| | |"
T2_TABLE = ("| Brewing time | Caffeine per cup |\n|---|---|\n| 1 min | 15 mg |\n"
            "| 3 min | 28 mg |\n| 5 min | 40 mg |")
T2_TABLE_N = "| Brewing time | Caffeine per cup |\n|---|---|\n| | |\n| | |\n| | |"
T3_TABLE = ("| Charger class | Power | Tariff |\n|---|---|---|\n| AC | 22 kW | $0.22/kWh |\n"
            "| DC | 50 kW | $0.30/kWh |\n| DC fast | 150 kW | $0.35/kWh |")
T3_TABLE_N = "| Charger class | Power | Tariff |\n|---|---|---|\n| | | |\n| | | |\n| | | |"
T4_TABLE = ("| Mixing parameter | Value |\n|---|---|\n| A:B concentrate ratio | 1:100 |\n"
            "| Water temperature | 18-22 C |\n| Reservoir size | 40 L |")
T4_TABLE_N = "| Mixing parameter | Value |\n|---|---|\n| | |\n| | |\n| | |"

TASKS = {
    "t1_solar": {
        "domain": "consumer-product",
        "version": "A",
        "units": [
            {"tag": 1, "type": "T",
             "content": "The SunRay S200 is a portable power station designed for camping and emergency backup. It uses a LiFePO4 battery pack and supports pass-through charging. The unit is rated for outdoor use in temperatures from -10 C to 45 C.",
             "neutral": None},
            {"tag": 2, "type": "T",
             "content": "The SunRay S200 accepts up to 120 W of solar input. Recharge time depends on the panel wattage; the charging chart reports measured times under full sun.",
             "neutral": None},
            {"tag": 3, "type": "I", "asset": "t1_device.png", "neutral_asset": "t1_device_neutral.png"},
            {"tag": 4, "type": "C", "asset": "t1_chart.png", "neutral_asset": "t1_chart_neutral.png"},
            {"tag": 5, "type": "B", "content": T1_TABLE, "neutral": T1_TABLE_N},
            {"tag": 6, "type": "T", "caption": True,
             "content": "Figure 1: The SunRay S200 portable power station with dual USB-C ports and a DC input port.",
             "neutral": None},
        ],
        "query": "How long does it take to fully recharge the SunRay S200 with a 60 W panel, and what is its maximum AC output?",
        "gold": [4, 5],
        "refs": ["With a 60 W panel, the SunRay S200 takes about 4 hours to fully recharge, and its maximum AC output is 100 W.",
                 "A full recharge with a 60 W panel takes roughly 4 hours; the unit can output up to 100 W of AC power."],
    },
    "t2_tea": {
        "domain": "health",
        "version": "A",
        "units": [
            {"tag": 1, "type": "T",
             "content": "Green tea contains catechins, of which EGCG is the most abundant and the most studied. Brewing temperature and leaf-to-water ratio also shape the flavor profile of the finished cup.",
             "neutral": None},
            {"tag": 2, "type": "T",
             "content": "Brewing guidelines: use one teaspoon of leaves per 240 ml cup. Longer brewing extracts more catechins but also more bitterness; the chart shows measured EGCG content by brewing time.",
             "neutral": None},
            {"tag": 3, "type": "I", "asset": "t2_label.png", "neutral_asset": "t2_label_neutral.png"},
            {"tag": 4, "type": "C", "asset": "t2_chart.png", "neutral_asset": "t2_chart_neutral.png"},
            {"tag": 5, "type": "B", "content": T2_TABLE, "neutral": T2_TABLE_N},
            {"tag": 6, "type": "T", "caption": True,
             "content": "Figure 1: Package label of the sencha sample used for the measurements on this page.",
             "neutral": None},
        ],
        "query": "What is the minimum brewing time to reach at least 70 mg of EGCG per cup, and where were these tea leaves harvested?",
        "gold": [3, 4],
        "refs": ["Brewing for at least 3 minutes reaches about 70 mg of EGCG per cup, and the label says the leaves were harvested in Uji, Kyoto.",
                 "You need at least a 3-minute brew for roughly 70 mg of EGCG per cup; the package label lists the origin as Uji, Kyoto."],
    },
    "t3_ev": {
        "domain": "policy-econ",
        "version": "A",
        "units": [
            {"tag": 1, "type": "T",
             "content": "The city's public charging network has expanded rapidly since 2022. The transport authority publishes yearly statistics and per-session tariffs for every charger class in the network.",
             "neutral": None},
            {"tag": 2, "type": "T",
             "content": "In 2025 the network added 400 new public chargers, bringing the total to 1,500. Utilization peaked at 65 percent during summer weekends.",
             "neutral": None},
            {"tag": 3, "type": "I", "asset": "t3_station.png", "neutral_asset": "t3_station_neutral.png"},
            {"tag": 4, "type": "C", "asset": "t3_chart.png", "neutral_asset": "t3_chart_neutral.png"},
            {"tag": 5, "type": "B", "content": T3_TABLE, "neutral": T3_TABLE_N},
            {"tag": 6, "type": "T", "caption": True,
             "content": "Figure 1: A DC fast-charging station operated by the city network.",
             "neutral": None},
        ],
        "query": "How many public chargers did the city have at the end of 2024, and what is the tariff for a DC fast-charging session?",
        "gold": [4, 5],
        "refs": ["At the end of 2024 the city had 1,100 public chargers, and DC fast charging is tariffed at $0.35 per kWh.",
                 "There were 1,100 public chargers by the end of 2024; a DC fast session costs $0.35 per kWh."],
    },
    "t4_hydro_twin": {
        "domain": "agritech",
        "version": "A",
        "twin": True,
        "units": [
            {"tag": 1, "type": "T",
             "content": "The system uses a 12 V DC inline pump to circulate nutrient solution from the reservoir to the grow channels. A timer runs the pump for 15 minutes every hour.",
             "content_B": "The system uses a low-voltage inline pump to circulate nutrient solution from the reservoir to the grow channels; the equipment photo lists its electrical specifications. A timer runs the pump for 15 minutes every hour.",
             "neutral": None},
            {"tag": 2, "type": "T",
             "content": "For lettuce, maintain the nutrient solution's electrical conductivity at 1.2 mS/cm, and check and top up the reservoir every three days.",
             "content_B": "For lettuce, maintain the nutrient solution's electrical conductivity at the crop-specific target shown in the chart, and check and top up the reservoir every three days.",
             "neutral": None},
            {"tag": 3, "type": "I", "asset": "t4_pump_A.png", "neutral_asset": "t4_pump_A_neutral.png",
             "asset_B": "t4_pump_B.png"},
            {"tag": 4, "type": "C", "asset": "t4_chart_A.png", "neutral_asset": "t4_chart_neutral.png",
             "asset_B": "t4_chart_B.png"},
            {"tag": 5, "type": "B", "content": T4_TABLE, "neutral": T4_TABLE_N},
            {"tag": 6, "type": "T", "caption": True,
             "content": "Figure 1: The inline circulation pump of the hydroponic lettuce system.",
             "neutral": None},
        ],
        "query": "What EC level should the nutrient solution be kept at for lettuce, and what voltage does the circulation pump require?",
        "gold_A": [1, 2],
        "gold_B": [3, 4],
        "gold": [1, 2],
        "refs": ["Keep the nutrient solution at an EC of 1.2 mS/cm for lettuce, and the circulation pump runs on a 12 V DC supply.",
                 "Lettuce needs an EC of 1.2 mS/cm; the circulation pump requires a 12 V DC supply."],
    },
}

def neutralize_text_unit(unit):
    return unit.get("neutral") or neutral_text(unit["content"])

def build_blocks(task_key, state, version="A"):
    """state: list of 6 ints in {-1,+1} (index 0 -> tag 1). Returns list of blocks."""
    t = TASKS[task_key]
    blocks = []
    for i, u in enumerate(t["units"]):
        present = state[i] == 1
        b = {"tag": u["tag"], "type": u["type"], "caption": bool(u.get("caption"))}
        if u["type"] in ("I", "C"):
            asset = u["asset"] if present else u["neutral_asset"]
            if not present and version == "B" and u.get("neutral_asset"):
                asset = u["neutral_asset"]
            if present and version == "B" and u.get("asset_B"):
                asset = u["asset_B"]
            b["image"] = os.path.join(ASSET_DIR, asset)
        else:
            if present:
                c = u["content"] if version == "A" else u.get("content_B", u["content"])
            else:
                c = neutralize_text_unit(u)
            b["text"] = c
        blocks.append(b)
    return blocks

def gold_set(task_key, version="A"):
    t = TASKS[task_key]
    if task_key == "t4_hydro_twin":
        return t["gold_A"] if version == "A" else t["gold_B"]
    return t["gold"]

def states(n=6):
    out = []
    for v in range(2 ** n):
        out.append([1 if (v >> (n - 1 - i)) & 1 else -1 for i in range(n)])
    return out

# ---------------------------------------------------------------- fact statements (0.2-B)
# Each task's reference answer decomposes into two atomic facts. "unit" = the evidence
# unit that carries the fact by construction (gold for that fact). "regex" detects the
# fact's presence in free-generated text (P0-1 generations) for the binary utility B_j.
FACTS = {
    "t1_solar": [
        {"id": "f1", "unit": 4, "regex": r"4\s*(hours|hrs)\b",
         "statement": "The SunRay S200 takes about 4 hours to fully recharge with a 60 W panel."},
        {"id": "f2", "unit": 5, "regex": r"100\s*(W|watts|watt)\b",
         "statement": "The maximum AC output of the SunRay S200 is 100 W."},
    ],
    "t2_tea": [
        {"id": "f1", "unit": 4, "regex": r"3\s*(minutes|mins|min)\b",
         "statement": "Brewing the tea for at least 3 minutes reaches about 70 mg of EGCG per cup."},
        {"id": "f2", "unit": 3, "regex": r"Uji|Kyoto",
         "statement": "The tea leaves were harvested in Uji, Kyoto."},
    ],
    "t3_ev": [
        {"id": "f1", "unit": 4, "regex": r"1,100|1100",
         "statement": "The city had 1,100 public chargers at the end of 2024."},
        {"id": "f2", "unit": 5, "regex": r"0\.35",
         "statement": "DC fast charging costs $0.35 per kWh."},
    ],
    "t4_hydro_twin": [
        {"id": "f1", "unit": 2, "regex": r"1\.2",
         "statement": "For lettuce, the nutrient solution should be kept at an EC of 1.2 mS/cm."},
        {"id": "f2", "unit": 1, "regex": r"12\s*V\b|12V",
         "statement": "The circulation pump runs on a 12 V DC supply."},
    ],
}

STAT_SYSTEM = ("You are a precise research assistant. You will be given a web page as a "
               "sequence of tagged evidence blocks, a question, and a candidate statement. "
               "Your task is to quantify how much each evidence block contributes to "
               "establishing that statement.")

STAT_SUFFIX = ("Statement to verify: {stmt}\n\n"
               "Task: Score the contribution of EVERY evidence block [EV-1] through [EV-6] "
               "to establishing this statement. Give each block a number between 0.0 (no "
               "contribution) and 1.0 (fully establishes the statement). Output ONLY a JSON "
               "object mapping tags to scores, for example: {{\"EV-1\": 0.2, \"EV-2\": 0.9, "
               "\"EV-3\": 0.0, \"EV-4\": 0.8, \"EV-5\": 0.7, \"EV-6\": 0.1}} — include all six "
               "tags, no other text.")

def summary():
    return {
        "tasks": {k: {"domain": v["domain"], "units": [
            {"tag": u["tag"], "type": u["type"]} for u in v["units"]],
            "query": v["query"], "gold": v["gold"], "n_refs": len(v["refs"]),
            "twin": bool(v.get("twin"))} for k, v in TASKS.items()},
        "facts": {k: [{"id": f["id"], "unit": f["unit"], "statement": f["statement"]}
                      for f in v] for k, v in FACTS.items()},
        "n_states": 64,
        "state_convention": "x_i=+1 present, x_i=-1 neutralized (info-neutral control)",
    }

if __name__ == "__main__":
    made = build_assets()
    meta = {"assets": sorted(os.listdir(ASSET_DIR)), **summary()}
    with open(os.path.join(ROOT, "data", "p0_tasks_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"assets: {len(made)} built; total files: {len(os.listdir(ASSET_DIR))}")
    b = build_blocks("t2_tea", [1, 1, -1, 1, 1, 1])
    print("sample state t2 x=[+1,+1,-1,+1,+1,+1]:")
    for blk in b:
        print("  ", blk["tag"], blk["type"], (blk.get("text") or blk.get("image", ""))[:70])
