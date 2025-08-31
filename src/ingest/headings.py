# src/ingest/headings.py
from typing import List, Dict, Any
import statistics

def detect_headings(blocks) -> List[Dict[str, Any]]:
    """
    Heuristic: treat lines where median span size is in the top ~10%
    for the page as candidate headings; bold boosts score.
    Returns flat list: {title, page, level_guess}
    """
    candidates = []
    for blk in blocks:
        sizes = [s["size"] for s in blk.spans if s["text"].strip()]
        if not sizes: 
            continue
        p90 = statistics.quantiles(sizes, n=10)[-1] if len(sizes) >= 10 else max(sizes)
        # build lines with their avg size + bold ratio
        lines = []
        cur_line = []
        prev_y2 = None
        for s in blk.spans:
            t = s["text"].strip()
            if not t:
                continue
            # crude line split via bbox y2
            y2 = s["bbox"][3] if s["bbox"] else None
            if prev_y2 is not None and y2 and abs(y2 - prev_y2) > 4:
                if cur_line:
                    lines.append(cur_line); cur_line = []
            cur_line.append(s)
            prev_y2 = y2
        if cur_line:
            lines.append(cur_line)

        for line in lines:
            line_text = " ".join([x["text"].strip() for x in line]).strip()
            if not line_text or len(line_text) < 3:
                continue
            avg_size = sum([x["size"] for x in line])/len(line)
            bold_ratio = sum(1 for x in line if x["bold"])/len(line)
            score = avg_size + (2.0 * bold_ratio)
            if avg_size >= p90 or score >= p90 + 1.0:
                # guess levels: very big → level 1, medium → 2, else 3
                lvl = 1 if avg_size >= p90 + 2 else (2 if avg_size >= p90 else 3)
                candidates.append({"title": line_text, "page": blk.page, "level_guess": lvl})
    return _dedupe_consecutive(candidates)

def _dedupe_consecutive(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    seen = set()
    for it in items:
        key = (it["page"], it["title"].lower())
        if key not in seen:
            out.append(it); seen.add(key)
    return out
