"""Twelve additional fixed offline Wikipedia pages for the six-slot audit.

The specification was written before model inference. `fetch_extension.py`
saves the HTML, selected text, and image assets under data/realweb_extension.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "realweb_extension"

SPEC = {
    "rw_mars2020": {"title": "Mars_2020", "domain": "technology", "paras": [0, 1, 2, 3], "image": 0,
        "question": "On what date was the Mars 2020 mission launched?",
        "statement": "The Mars 2020 mission was launched on July 30, 2020."},
    "rw_voyager1": {"title": "Voyager_1", "domain": "technology", "paras": [0, 1, 2, 3], "image": 0,
        "question": "On what date was Voyager 1 launched?",
        "statement": "Voyager 1 was launched on September 5, 1977."},
    "rw_kepler": {"title": "Kepler_space_telescope", "domain": "technology", "paras": [0, 1, 2, 3], "image": 0,
        "question": "In what year did NASA launch the Kepler space telescope?",
        "statement": "NASA launched the Kepler space telescope in 2009."},
    "rw_iss": {"title": "International_Space_Station", "domain": "technology", "paras": [0, 1, 2, 3], "image": 0,
        "question": "Which five agencies operate the International Space Station?",
        "statement": "The International Space Station is operated by NASA, Roscosmos, ESA, JAXA, and CSA."},
    "rw_switch": {"title": "Nintendo_Switch", "domain": "product", "paras": [0, 1, 2, 3], "image": 0,
        "question": "On what date was the Nintendo Switch released worldwide in most regions?",
        "statement": "The Nintendo Switch was released worldwide in most regions on March 3, 2017."},
    "rw_ipad10": {"title": "iPad_%2810th_generation%29", "domain": "product", "paras": [1, 2, 3, 4], "image": 0,
        "question": "On what date was the tenth-generation iPad released?",
        "statement": "The tenth-generation iPad was released on October 26, 2022."},
    "rw_xboxseries": {"title": "Xbox_Series_X_and_Series_S", "domain": "product", "paras": [0, 1, 2, 3], "image": 0,
        "question": "On what date were the Xbox Series X and Series S released?",
        "statement": "The Xbox Series X and Series S were released on November 10, 2020."},
    "rw_model3": {"title": "Tesla_Model_3", "domain": "product", "paras": [0, 1, 2, 3], "image": 0,
        "question": "In what year was the Tesla Model 3 introduced?",
        "statement": "The Tesla Model 3 was introduced in 2017."},
    "rw_penicillin": {"title": "Penicillin", "domain": "health", "paras": [2, 0, 1, 3], "image": 1,
        "question": "Who discovered penicillin in 1928?",
        "statement": "Penicillin was discovered in 1928 by Alexander Fleming."},
    "rw_insulin": {"title": "Insulin", "domain": "health", "paras": [2, 3, 4, 5], "image": 0,
        "question": "Which cells produce insulin?",
        "statement": "Insulin is produced by beta cells of the pancreatic islets."},
    "rw_vitamind": {"title": "Vitamin_D", "domain": "health", "paras": [0, 1, 2, 3], "image": 2,
        "question": "Absorption of which mineral is increased by vitamin D?",
        "statement": "Vitamin D increases intestinal absorption of calcium."},
    "rw_aspirin": {"title": "Aspirin", "domain": "health", "paras": [0, 1, 2, 3], "image": 0,
        "question": "What chemical compound is aspirin?",
        "statement": "Aspirin is acetylsalicylic acid."},
}


def build_blocks(pid):
    data = json.loads((DATA / pid / "blocks.json").read_text(encoding="utf-8"))
    return [{**unit, "image": str(DATA / pid / unit["image"]) if unit["type"] == "I" else unit.get("image")}
            for unit in data["units"]]
