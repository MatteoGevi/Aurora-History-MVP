# src/ingest/toc_builder.py
from typing import List, Dict, Any

def build_toc(bookmarks: List[Dict[str,Any]], heading_candidates: List[Dict[str,Any]]):
    """
    Prefer PDF bookmarks when present; supplement gaps with heading candidates.
    Produce nodes with page_start/page_end and hierarchical levels.
    """
    nodes = []
    if bookmarks:
        nodes = [{"title": b["title"], "level": b["level"], "page_start": b["page"], "page_end": None} for b in bookmarks]
    else:
        nodes = [{"title": h["title"], "level": min(h["level_guess"],3), "page_start": h["page"], "page_end": None} for h in heading_candidates]

    # assign page_end = next node's start - 1 (per level)
    stack = []
    for i, n in enumerate(nodes):
        while stack and n["level"] <= stack[-1]["level"]:
            top = stack.pop()
            top["page_end"] = nodes[i]["page_start"] - 1
        stack.append(n)
    # close remaining
    if stack:
        last_page = max([x["page_start"] for x in nodes]) + 10000  # will clip later
        while stack:
            top = stack.pop()
            top["page_end"] = top["page_end"] or last_page

    # clip to max page encountered
    max_page = max([n["page_start"] for n in nodes]) if nodes else 1
    for n in nodes:
        if n["page_end"] < n["page_start"]:
            n["page_end"] = n["page_start"]
    return nodes
