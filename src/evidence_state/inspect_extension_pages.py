"""Inspect candidate Wikipedia articles for an offline RealWeb extension."""
import sys
import urllib.request
from bs4 import BeautifulSoup

for title in sys.argv[1:]:
    url = f"https://en.wikipedia.org/wiki/{title}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research paper data curation)"})
    html = urllib.request.urlopen(req, timeout=20).read()
    soup = BeautifulSoup(html, "html.parser")
    paras = [p.get_text(" ", strip=True) for p in soup.select("div.mw-parser-output p")]
    paras = [p for p in paras if len(p) > 80]
    figures = []
    for img in soup.select("figure img, table.infobox img"):
        src = img.get("src", "")
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = "https://en.wikipedia.org" + src
        if "wikimedia" in src and "svg" not in src.lower():
            figures.append((src, img.get("alt", "")))
    print(f"\n## {title} bytes={len(html)} paras={len(paras)} images={len(figures)}")
    for i, p in enumerate(paras[:8]):
        print(f"P{i}: {p[:320]}")
    for i, (src, alt) in enumerate(figures[:3]):
        print(f"I{i}: {alt[:100]} {src[:100]}")
