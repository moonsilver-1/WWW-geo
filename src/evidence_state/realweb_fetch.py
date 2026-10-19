# -*- coding: utf-8 -*-
"""RealWeb fetcher: download real public pages + assets, extract candidate evidence blocks.
Pilot: 6 Wikipedia articles (2 per domain) via the REST HTML API (stable, offline-renderable).
Original HTML saved; images downloaded to local files; candidate blocks printed for curation."""
import json, os, re, sys, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "realweb")
os.makedirs(OUT, exist_ok=True)

PAGES = {
    "rw_jwst":  {"domain": "technology", "title": "James_Webb_Space_Telescope"},
    "rw_hubble": {"domain": "technology", "title": "Hubble_Space_Telescope"},
    "rw_iphone15": {"domain": "product", "title": "IPhone_15"},
    "rw_ps5": {"domain": "product", "title": "PlayStation_5"},
    "rw_mrna": {"domain": "health", "title": "MRNA_vaccine"},
    "rw_vitaminc": {"domain": "health", "title": "Vitamin_C"},
}

def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; contact: local)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def wiki_html(title):
    return f"https://en.wikipedia.org/api/rest_v1/page/html/{title}"

def extract_blocks(html):
    """stdlib HTML parse: paragraphs, headings, tables (text), images, captions."""
    from html.parser import HTMLParser
    import html as H
    blocks = {"paras": [], "tables": [], "images": [], "captions": []}
    class P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack = []
            self.buf = []
            self.mode = None
            self.cur_img = None
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            self.stack.append(tag)
            if tag == "p" and len(self.stack) <= 6:
                self.mode = "p"; self.buf = []
            elif tag == "table":
                self.mode = "table"; self.buf = []
            elif tag == "figcaption":
                self.mode = "cap"; self.buf = []
            elif tag == "img":
                src = a.get("src", "")
                res = a.get("resource", "")
                low = (src + " " + res).lower()
                if not src or any(x in low for x in ("oojs", "edit-ltr", "icon", "logo", "ambox",
                                                     "question_book", "wiki_letter", "symbol_", "/wikipedia/en/",
                                                     "commons-logo", "wikisource", "wiktionary")):
                    return
                if src.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    src = "https://en.wikipedia.org" + src
                m = re.search(r"(\d{2,4})px-", src)
                if m and int(m.group(1)) < 200:
                    return
                if ("wikimedia.org" in src or "wikimedia.org" in res) and src not in blocks["images"]:
                    blocks["images"].append(src.split(" ")[0])
        def handle_endtag(self, tag):
            if self.stack and self.stack[-1] == tag:
                self.stack.pop()
            if self.mode == "p" and tag == "p":
                t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
                if len(t) > 80:
                    blocks["paras"].append(t)
                self.mode = None
            elif self.mode == "cap" and tag == "figcaption":
                t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
                if len(t) > 20:
                    blocks["captions"].append(t)
                self.mode = None
            elif self.mode == "table" and tag == "table":
                t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
                if len(t) > 60:
                    blocks["tables"].append(t[:2000])
                self.mode = None
        def handle_data(self, data):
            if self.mode:
                self.buf.append(data)
    p = P()
    p.feed(html)
    return blocks

def main():
    summary = {}
    for pid, meta in PAGES.items():
        d = os.path.join(OUT, pid)
        os.makedirs(os.path.join(d, "assets"), exist_ok=True)
        html_b = fetch(wiki_html(meta["title"]))
        with open(os.path.join(d, "page.html"), "wb") as f:
            f.write(html_b)
        bl = extract_blocks(html_b.decode("utf-8", "ignore"))
        # download up to 4 content images (with retry)
        kept = []
        for src in bl["images"]:
            if len(kept) >= 4:
                break
            ok = False
            for attempt in range(3):
                try:
                    data = fetch(src, timeout=25)
                    ext = ".jpg" if ".jpg" in src or ".jpeg" in src else (".png" if ".png" in src else ".jpg")
                    fn = f"img{len(kept)+1}{ext}"
                    with open(os.path.join(d, "assets", fn), "wb") as f:
                        f.write(data)
                    kept.append({"file": f"assets/{fn}", "src": src[:120]})
                    ok = True
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  img fail {pid}: {str(e)[:60]}")
                    import time as _t
                    _t.sleep(2)
            _ = ok
        summary[pid] = {"domain": meta["domain"], "title": meta["title"],
                        "n_paras": len(bl["paras"]), "n_tables": len(bl["tables"]),
                        "n_captions": len(bl["captions"]), "images": kept}
        with open(os.path.join(d, "blocks.json"), "w", encoding="utf-8") as f:
            json.dump(bl, f, ensure_ascii=False, indent=1)
        print(f"{pid}: paras={len(bl['paras'])} tables={len(bl['tables'])} "
              f"caps={len(bl['captions'])} imgs={len(kept)}")
    with open(os.path.join(OUT, "fetch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
