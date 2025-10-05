# conceptual_subchunk.py  (v2 - split only on BLANK LINES)
import argparse, json, os, re
from typing import List, Dict, Any

# Safety window for very long blocks
MAX_CHARS_DEFAULT = 1200
OVERLAP_DEFAULT   = 150

# Sentence splitter for safety windows
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z“"(\[])')

# --- helpers ---------------------------------------------------------------

def flatten_soft_wraps(s: str) -> str:
    """
    Inside a logical block, collapse SINGLE newlines to spaces, keep double+ newlines.
    Also fix hyphenation across line breaks: "hyphen-\nnation" -> "hyphenation".
    """
    if not s:
        return ""
    # fix hyphenation at end of line
    s = re.sub(r'-\s*\n\s*', '', s)  # join hyphenated breaks
    # collapse single newlines to spaces (but keep blank-line boundaries)
    # approach: split on double+ newlines, then join parts with preserved blanklines outside this function
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', s)

def split_blocks_punto_a_capo(text: str) -> List[str]:
    """
    'Punto a capo' = blocks separated by one or more BLANK lines (\\n\\n+).
    We don't split on single newlines; they are soft wraps and get flattened.
    """
    if not text:
        return []
    # Normalize CRLF
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    # Split on blank lines (one or more)
    raw_blocks = re.split(r'\n\s*\n+', text)
    blocks = []
    for b in raw_blocks:
        b = b.strip()
        if not b:
            continue
        blocks.append(' '.join(flatten_soft_wraps(b).split()))
    return blocks

def safety_window(unit: str, max_chars: int, overlap: int) -> List[str]:
    """If a conceptual unit is too long, split sentence-aware with overlap."""
    if len(unit) <= max_chars:
        return [unit]
    sents = [t.strip() for t in SENT_SPLIT.split(re.sub(r'\s+', ' ', unit).strip()) if t.strip()]
    out, cur, cur_len = [], [], 0
    for s in sents:
        if cur and cur_len + len(s) > max_chars:
            out.append(' '.join(cur).strip())
            # build overlap from tail
            tail, L = [], 0
            for ts in reversed(cur):
                tail.insert(0, ts); L += len(ts)
                if L >= overlap: break
            cur = tail
            cur_len = sum(len(x) for x in cur)
        cur.append(s); cur_len += len(s)
    if cur:
        out.append(' '.join(cur).strip())
    return [c for c in out if c.strip()]

def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Create conceptual chunks (true 'punto a capo') from paragraphs.jsonl")
    ap.add_argument('--in',  dest='in_path',  default='data/out/chunks.paragraphs.jsonl')
    ap.add_argument('--out', dest='out_path', default='data/out/chunks.paragraphs.conceptual.jsonl')
    ap.add_argument('--max_chars', type=int, default=MAX_CHARS_DEFAULT, help='safety max per unit')
    ap.add_argument('--overlap',   type=int, default=OVERLAP_DEFAULT,   help='overlap for safety windows')
    args = ap.parse_args()

    base_rows = list(read_jsonl(args.in_path))
    out_rows: List[Dict[str, Any]] = []

    for p_idx, r in enumerate(base_rows):
        para_id = r.get('paragraph_id', f'para-{p_idx}')
        content = r.get('content', '') or ''
        blocks = split_blocks_punto_a_capo(content)

        for u_idx, unit in enumerate(blocks):
            pieces = safety_window(unit, args.max_chars, args.overlap)
            if len(pieces) == 1:
                o = dict(r)
                o['content'] = unit
                o['paragraph_id'] = para_id
                o['concept_unit_idx'] = u_idx
                o['id'] = o.get('id', f'{para_id}::c{u_idx}')
                out_rows.append(o)
            else:
                for s_idx, sub in enumerate(pieces):
                    o = dict(r)
                    o['content'] = sub
                    o['paragraph_id'] = para_id
                    o['concept_unit_idx'] = u_idx
                    o['sub_idx'] = s_idx
                    o['id'] = f'{para_id}::c{u_idx}::s{s_idx}'
                    out_rows.append(o)

    write_jsonl(args.out_path, out_rows)
    print(f'Wrote {len(out_rows)} conceptual chunks → {args.out_path}')

if __name__ == '__main__':
    main()