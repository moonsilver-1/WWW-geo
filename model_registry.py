"""Pinned models and memory requirements shared by preparation and evaluation."""
import json
from pathlib import Path


def load_registry():
    return json.loads(Path(__file__).with_name('model_registry.json').read_text())
