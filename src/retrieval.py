from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Import from your config
from config.constants import supabase, EMBEDDING_MODEL

print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✓ Model loaded (dimension: {_model.get_sentence_embedding_dimension()})")

def embed_query(text: str) -> List[float]:
    """
    Generate embedding for a query string.
    
    Args:
        text: Query text to embed
        
    Returns:
        List of floats representing the embedding
    """
    embedding = _model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def search(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """
    Basic semantic search - your main retrieval function.
    
    Args:
        query: What the student is asking about or being tested on
        top_k: How many relevant chunks to retrieve
        similarity_threshold: Minimum similarity score (0-1)
        
    Returns:
        List of relevant text chunks with metadata
        
    Example:
        >>> results = search("What caused the American Revolution?", top_k=3)
        >>> for r in results:
        ...     print(r['text'][:100])
    """
    query_embedding = embed_query(query)
    
    response = supabase.rpc(
        'match_chunks',
        {
            'query_embedding': query_embedding,
            'match_threshold': similarity_threshold,
            'match_count': top_k
        }
    ).execute()
    
    return response.data if response.data else []


def search_with_context(
    query: str,
    top_k: int = 3,
    context_window: int = 2
) -> List[Dict]:
    """
    Search and include surrounding chunks for more context.
    Useful when you need the full story around a match.
    
    Args:
        query: Search query
        top_k: Number of matches
        context_window: How many chunks before/after to include
        
    Returns:
        List of dicts with 'match' and 'context' keys
        
    Example:
        >>> results = search_with_context("Revolutionary War battles", top_k=2)
        >>> match = results[0]['match']
        >>> context_chunks = results[0]['context']  # surrounding chunks
    """
    initial_results = search(query, top_k=top_k)
    
    if not initial_results:
        return []
    
    enriched = []
    
    for result in initial_results:
        chunk_seq = result['chunk_seq']
        toc_node_id = result['toc_node_id']
        
        # Get surrounding chunks
        context_response = supabase.from_('chunks') \
            .select('chunk_id, text, chunk_seq, section_title, page_start, page_end') \
            .eq('toc_node_id', toc_node_id) \
            .gte('chunk_seq', chunk_seq - context_window) \
            .lte('chunk_seq', chunk_seq + context_window) \
            .order('chunk_seq') \
            .execute()
        
        # Mark which is the match
        context_chunks = []
        for chunk in context_response.data:
            chunk['position'] = 'match' if chunk['chunk_seq'] == chunk_seq else \
                               'before' if chunk['chunk_seq'] < chunk_seq else 'after'
            context_chunks.append(chunk)
        
        enriched.append({
            'match': result,
            'context': context_chunks
        })
    
    return enriched


def search_by_page(
    query: str,
    page_start: int,
    page_end: int,
    top_k: int = 5
) -> List[Dict]:
    """
    Search within a specific page range.
    Useful when you know which chapter/section to test.
    
    Args:
        query: Search query
        page_start: Starting page number
        page_end: Ending page number
        top_k: Number of results
        
    Returns:
        List of chunks within the page range
        
    Example:
        >>> # Test on Chapter 3 (pages 45-67)
        >>> results = search_by_page("taxation policy", 45, 67, top_k=3)
    """
    query_embedding = embed_query(query)
    
    response = supabase.rpc(
        'match_chunks_by_page',
        {
            'query_embedding': query_embedding,
            'min_page': page_start,
            'max_page': page_end,
            'match_threshold': 0.3,
            'match_count': top_k
        }
    ).execute()
    
    return response.data if response.data else []


def search_by_section(
    query: str,
    section_pattern: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Search within sections matching a pattern.
    Useful for topic-specific assessment.
    
    Args:
        query: Search query
        section_pattern: Text to match in section titles (case-insensitive)
        top_k: Number of results
        
    Returns:
        List of chunks from matching sections
        
    Example:
        >>> # Only search in sections about "Civil War"
        >>> results = search_by_section("battle strategy", "civil war", top_k=5)
    """
    query_embedding = embed_query(query)
    
    # Get initial results
    response = supabase.rpc(
        'match_chunks',
        {
            'query_embedding': query_embedding,
            'match_threshold': 0.0,
            'match_count': 100
        }
    ).execute()
    
    results = response.data if response.data else []
    
    # Filter by section pattern
    filtered = [
        r for r in results 
        if section_pattern.lower() in r['section_title'].lower()
    ]
    
    return filtered[:top_k]


def get_section_content(section_title_pattern: str) -> List[Dict]:
    """
    Get all chunks from a section by title pattern.
    Useful for retrieving complete chapter content.
    
    Args:
        section_title_pattern: Pattern to match section title
        
    Returns:
        All chunks from matching section(s)
        
    Example:
        >>> # Get entire "American Revolution" section
        >>> chunks = get_section_content("American Revolution")
    """
    # Find matching sections
    toc_response = supabase.from_('toc_nodes') \
        .select('id') \
        .ilike('title', f'%{section_title_pattern}%') \
        .execute()
    
    if not toc_response.data:
        return []
    
    section_ids = [node['id'] for node in toc_response.data]
    
    # Get all chunks from these sections
    chunks_response = supabase.from_('chunks') \
        .select('*') \
        .in_('toc_node_id', section_ids) \
        .order('chunk_seq') \
        .execute()
    
    return chunks_response.data if chunks_response.data else []


def get_toc() -> List[Dict]:
    """
    Get the table of contents.
    Useful for showing students what topics are available.
    
    Returns:
        List of all sections with titles, levels, and page ranges
        
    Example:
        >>> toc = get_toc()
        >>> for section in toc[:10]:
        ...     print(f"H{section['level']}: {section['title']} (p.{section['page_start']})")
    """
    response = supabase.from_('toc_nodes') \
        .select('*') \
        .order('page_start') \
        .execute()
    
    return response.data if response.data else []


# ============================================================================
# UTILITY FUNCTIONS FOR YOUR ASSESSOR
# ============================================================================

def get_reference_content(topic: str, top_k: int = 3) -> str:
    """
    Get reference content for grading student answers.
    Returns concatenated text from top matches.
    
    Args:
        topic: The topic being assessed
        top_k: Number of chunks to retrieve
        
    Returns:
        Combined reference text
        
    Example:
        >>> # Student answered a question about Boston Tea Party
        >>> reference = get_reference_content("Boston Tea Party causes", top_k=3)
        >>> # Now use reference to grade the student's answer
    """
    results = search(topic, top_k=top_k)
    
    if not results:
        return ""
    
    # Combine text from all chunks
    texts = [r['text'] for r in results]
    return "\n\n".join(texts)


def format_context(chunks: List[Dict], max_length: int = 2000) -> str:
    """
    Format chunks into a readable context string for LLM.
    Truncates if too long.
    
    Args:
        chunks: List of chunk dicts from search()
        max_length: Max characters to return
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return ""
    
    formatted = []
    total_chars = 0
    
    for i, chunk in enumerate(chunks, 1):
        section = chunk['section_title']
        pages = f"p.{chunk['page_start']}-{chunk['page_end']}"
        text = chunk['text']
        
        chunk_text = f"[Source {i}: {section} ({pages})]\n{text}\n"
        
        if total_chars + len(chunk_text) > max_length:
            break
        
        formatted.append(chunk_text)
        total_chars += len(chunk_text)
    
    return "\n".join(formatted)


def get_stats() -> Dict:
    """Get database statistics"""
    try:
        response = supabase.rpc('get_vector_stats').execute()
        return response.data[0] if response.data else {}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# EXAMPLE USAGE FOR AURORA ASSESSOR
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("AURORA ASSESSOR - RETRIEVAL EXAMPLES")
    print("="*70)
    
    # Example 1: Simple search
    print("\n📚 Example 1: Basic Search")
    print("-"*70)
    results = search("What were the causes of the American Revolution?", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['section_title']} (similarity: {r.get('similarity', 0):.3f})")
        print(f"   Pages {r['page_start']}-{r['page_end']}")
        print(f"   {r['text'][:150]}...")
    
    # Example 2: Get reference content for grading
    print("\n\n🎓 Example 2: Get Reference for Grading")
    print("-"*70)
    topic = "Boston Tea Party"
    reference = get_reference_content(topic, top_k=2)
    print(f"Topic: {topic}")
    print(f"Reference length: {len(reference)} chars")
    print(f"\nReference preview:\n{reference[:300]}...")
    
    # Example 3: Format for LLM
    print("\n\n🤖 Example 3: Format Context for LLM")
    print("-"*70)
    results = search("Declaration of Independence", top_k=2)
    context = format_context(results, max_length=500)
    print("Formatted context for LLM prompt:")
    print(context)
    
    # Example 4: Check stats
    print("\n\n📊 Database Stats")
    print("-"*70)
    stats = get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")