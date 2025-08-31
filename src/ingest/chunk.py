# src/ingest/chunk.py
from typing import List, Dict, Any
import regex as re

def split_sentences(text: str) -> List[str]:
    # lightweight sentence split (avoid heavy models for MVP)
    # handles dates/names crudely; refine later if needed
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])', text.strip())
    return [p.strip() for p in parts if p.strip()]

def make_chunks(blocks, toc_nodes, token_target=900, overlap_sents=2):
    """
    For each ToC node, gather page texts within [page_start, page_end],
    sentence-split, then assemble ~token_target chunks with small overlap.
    """
    # naive token approximation = words
    page_text = {blk.page: blk.text for blk in blocks}
    chunks = []
    for node in toc_nodes:
        pages = [p for p in range(node["page_start"], node["page_end"]+1)]
        text = "\n".join([page_text.get(p,"") for p in pages])
        sents = split_sentences(text)
        cur = []
        cur_len = 0
        idx = 0
        while idx < len(sents):
            cur.append(sents[idx]); cur_len += len(sents[idx].split())
            idx += 1
            if cur_len >= token_target or idx == len(sents):
                start_idx = max(0, idx - len(cur))
                chunk_text = " ".join(cur).strip()
                if chunk_text:
                    chunks.append({
                        "toc_title": node["title"],
                        "page_start": node["page_start"],
                        "page_end": node["page_end"],
                        "header_path": node["title"],  # extend to include parent path later
                        "text": chunk_text
                    })
                # overlap
                back = max(0, len(cur) - overlap_sents)
                cur = cur[back:]
                cur_len = sum(len(s.split()) for s in cur)
    return chunks
