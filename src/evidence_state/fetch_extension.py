"""Freeze twelve public article excerpts and one image per article."""
from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from PIL import Image
from realweb_extension import SPEC, DATA

HEADERS = {"User-Agent": "Mozilla/5.0 (research paper offline evaluation)"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=25).read()


def image_urls(soup):
    result = []
    for img in soup.select("figure img, table.infobox img"):
        src = img.get("src", "")
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = "https://en.wikipedia.org" + src
        if "wikimedia" in src and "svg" not in src.lower():
            cap = img.get("alt", "")
            fig = img.find_parent("figure")
            if fig and fig.find("figcaption"):
                cap = fig.find("figcaption").get_text(" ", strip=True) or cap
            result.append((src, cap))
    return result


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for pid, spec in SPEC.items():
        folder = DATA / pid
        folder.mkdir(exist_ok=True)
        url = f"https://en.wikipedia.org/wiki/{spec['title']}"
        html = get(url)
        (folder / "source.html").write_bytes(html)
        soup = BeautifulSoup(html, "html.parser")
        paras = [clean(p.get_text(" ", strip=True)) for p in soup.select("div.mw-parser-output p")]
        paras = [p for p in paras if len(p) > 80]
        imgs = image_urls(soup)
        source_img, caption = imgs[spec["image"]]
        image = get(source_img)
        suffix = ".jpg" if image[:3] == b"\xff\xd8\xff" else (".gif" if image[:4] == b"GIF8" else ".png")
        (folder / ("source_image" + suffix)).write_bytes(image)
        if suffix == ".gif":
            image_name = "image.png"
            Image.open(io.BytesIO(image)).convert("RGB").save(folder / image_name)
        else:
            image_name = "image" + suffix
            (folder / image_name).write_bytes(image)
        fact, context, adjacent, distractor = [paras[i][:420] for i in spec["paras"]]
        caption_source = "article caption or image alt" if clean(caption) else "generic page label"
        caption = clean(caption)[:200] or f"{urllib.parse.unquote(spec['title']).replace('_', ' ')} image"
        units = [
            {"tag": 1, "type": "T", "role": "fact", "caption": False, "text": fact},
            {"tag": 2, "type": "T", "role": "context", "caption": False, "text": context},
            {"tag": 3, "type": "I", "role": "image", "caption": False, "image": image_name},
            {"tag": 4, "type": "T", "role": "caption", "caption": True, "text": caption},
            {"tag": 5, "type": "T", "role": "adjacent", "caption": False, "text": adjacent},
            {"tag": 6, "type": "T", "role": "distractor", "caption": False, "text": distractor},
        ]
        (folder / "blocks.json").write_text(json.dumps({"units": units}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        manifest[pid] = {"url": url, "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "domain": spec["domain"], "question": spec["question"],
                         "statement": spec["statement"], "para_indices": spec["paras"],
                         "image_index": spec["image"], "image_url": source_img,
                         "caption_source": caption_source,
                         "html_sha256": hashlib.sha256(html).hexdigest(),
                         "source_image_sha256": hashlib.sha256(image).hexdigest(),
                         "image_sha256": hashlib.sha256((folder / image_name).read_bytes()).hexdigest(),
                         "blocks_sha256": hashlib.sha256((folder / "blocks.json").read_bytes()).hexdigest()}
        print(pid, len(html), len(image), len(paras), len(imgs), flush=True)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
