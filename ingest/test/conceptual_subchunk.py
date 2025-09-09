# conceptual_subchunk.py
# Normalize paragraph chunks to align with toc.json and emit stable chunk_id.
# - Split ONLY on blank lines (true paragraph breaks), never on '.'.
# - Uses toc.json to attach section_node_id, section_level, ancestors, level_path.
# - No doc_id, no header_h* in the output.

from __future__ import annotations
import argparse, json, os, re, sys, string
from typing import Dict, Any, Iterable, List, Tuple

# ---- sentence-aware safety window (only used if a block is very long) ----
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z“"(\[])')

def safety_window(unit: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = re.sub(r'\s+', ' ', (unit or '')).strip()
    if len(text) <= max_chars:
        return [text]
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    out, cur, cur_len = [], [], 0
    for s in sents:
        if cur and cur_len + len(s) > max_chars:
            out.append(" ".join(cur).strip())
            # overlap from tail
            tail, L = [], 0
            for ts in reversed(cur):
                tail.insert(0, ts); L += len(ts)
                if L >= overlap: break
            cur, cur_len = tail, sum(len(x) for x in tail)
        cur.append(s); cur_len += len(s)
    if cur:
        out.append(" ".join(cur).strip())
    return [c for c in out if c.strip()]

# -------------------- IO helpers --------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(rows: Iterable[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------- text normalization ----------------
def normalize_paragraph_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # de-hyphenate across linebreaks: "govern-\nment" -> "government"
    t = re.sub(r"-\n([a-z])", r"\1", t)
    # collapse single linebreaks (soft wraps) to spaces, keep blank lines as separators
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # compress stray whitespace
    t = re.sub(r"[ \t\f\v]+", " ", t)
    return t.strip()

def split_on_blank_lines(text: str) -> List[str]:
    norm = normalize_paragraph_text(text or "")
    return [b.strip() for b in re.split(r"\n\s*\n+", norm) if b.strip()]

# ---------------- ToC mapping ----------------
# keep -, _, : ; strip other punctuation for robust title matching if needed
_PUNCT = "".join(ch for ch in string.punctuation if ch not in ["-", "_", ":"])
_PTABLE = str.maketrans("", "", _PUNCT)

def norm_title(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip().lower().translate(_PTABLE)
    s = re.sub(r"\s+", " ", s)
    return s

def flatten_toc(nodes: List[Dict[str, Any]]) -> Tuple[
    Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, List[str]]
]:
    """
    Returns:
      id2node: node_id -> node (with level/title/children)
      title2id: normalized title -> node_id (deepest wins; DFS)
      ancestors_incl_self: node_id -> [ancestor_ids..., node_id]
    """
    id2node: Dict[str, Dict[str, Any]] = {}
    title2id: Dict[str, str] = {}
    ancestors_incl_self: Dict[str, List[str]] = {}
    stack: List[str] = []

    def walk(nlist: List[Dict[str, Any]]):
        for n in nlist:
            nid, title = n["id"], norm_title(n.get("title"))
            id2node[nid] = n
            title2id[title] = nid
            ancestors_incl_self[nid] = stack + [nid]
            stack.append(nid)
            walk(n.get("children", []))
            stack.pop()
    walk(nodes)
    return id2node, title2id, ancestors_incl_self

def titles_for_ids(id2node: Dict[str, Dict[str, Any]], ids: List[str]) -> List[str]:
    out = []
    for nid in ids:
        t = id2node.get(nid, {}).get("title")
        out.append(t if t else nid)
    return out

# ---------------- main normalization ----------------
def normalize_chunks(toc_path: str, in_path: str, out_path: str,
                     max_chars: int, overlap: int):
    if not os.path.exists(toc_path):
        print(f"[error] toc.json not found: {toc_path}", file=sys.stderr)
        sys.exit(1)

    toc = json.loads(open(toc_path, "r", encoding="utf-8").read())
    id2node, title2id, ancestors_incl_self = flatten_toc(toc["nodes"])

    # per-section running counters -> stable chunk_id per ToC node
    per_section_counts: Dict[str, int] = {}
    global_count = 0

    def next_chunk_id(section_id: str | None) -> str:
        nonlocal global_count
        if section_id:
            per_section_counts[section_id] = per_section_counts.get(section_id, 0) + 1
            return f"{section_id}::c{per_section_counts[section_id]:06d}"
        # fallback: global sequence if no section
        global_count += 1
        return f"c{global_count:08d}"

    def gen():
        for base in read_jsonl(in_path):
            # We prefer an existing section_node_id from the input (already ToC-aligned)
            section_id = base.get("section_node_id")
            # Get section properties from ToC if available
            anc_ids = ancestors_incl_self.get(section_id, []) if section_id else []
            level_path = " > ".join(titles_for_ids(id2node, anc_ids)) if section_id else ""
            section_level = id2node.get(section_id, {}).get("level") if section_id else None

            # Split ONLY on blank lines, never on '.'
            blocks = split_on_blank_lines(base.get("content", ""))

            for block in blocks:
                # Safety window only if very long
                pieces = safety_window(block, max_chars=max_chars, overlap=overlap)
                for piece in pieces:
                    yield {
                        "chunk_id": next_chunk_id(section_id),
                        "section_node_id": section_id,
                        "section_level": section_level,
                        "section_ancestor_ids": anc_ids,      # includes self
                        "level_path": level_path,
                        "content": piece.strip(),
                        "token_len": max(1, len(piece) // 4),
                    }

    rows = list(gen())
    write_jsonl(rows, out_path)
    print(f"[ok] wrote {len(rows)} chunks → {out_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Produce ToC-aligned paragraph chunks with stable chunk_id (no doc_id, no headers)."
    )
    ap.add_argument("--toc",   default="data/out/toc.json", help="Path to toc.json")
    ap.add_argument("--in",    dest="in_path",  default="data/out/chunks.paragraphs.jsonl",
                    help="Input JSONL (paragraphs from your chunker)")
    ap.add_argument("--out",   dest="out_path", default="data/out/chunks.normalized.jsonl",
                    help="Output JSONL (normalized chunks)")
    ap.add_argument("--max-chars", type=int, default=1200,
                    help="Max chars before sentence-aware safety window is applied")
    ap.add_argument("--overlap",   type=int, default=150,
                    help="Overlap between safety windows")
    args = ap.parse_args()

    normalize_chunks(args.toc, args.in_path, args.out_path, args.max_chars, args.overlap)

if __name__ == "__main__":
    main()