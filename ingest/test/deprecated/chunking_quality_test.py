# tests/test_chunks.py
import json
from pathlib import Path
from collections import defaultdict
import statistics

def test_chunk_quality(chunks_jsonl: Path):
    """Test chunk length distribution and detect duplicates"""
    rows = [json.loads(line) for line in open(chunks_jsonl) if line.strip()]
    
    # Length distribution
    lengths = [len(r["text"]) for r in rows]
    assert len(rows) > 0, "No chunks generated"
    assert statistics.mean(lengths) > 500, "Chunks too short on average"
    assert sum(1 for L in lengths if L > 2000) < len(rows) * 0.05, "Too many oversized chunks"
    
    # Duplicate detection
    seen, dups = set(), 0
    for r in rows:
        key = r["text"][:400].strip().lower()
        if key in seen:
            dups += 1
        seen.add(key)
    
    dup_rate = dups / len(rows)
    print(f"Duplicate rate: {dup_rate:.2%} ({dups}/{len(rows)})")
    assert dup_rate < 0.30, f"Duplicate rate too high: {dup_rate:.2%}"
    
    # Section coverage
    sections = {r["section_id"] for r in rows}
    print(f"Sections covered: {len(sections)}")
    assert len(sections) > 10, "Too few sections covered"
    
    print("✅ Chunk quality tests passed")

if __name__ == "__main__":
    test_chunk_quality(Path("data/out/chunks.jsonl"))