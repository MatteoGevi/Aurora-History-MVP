# retrieval/retrieval.py - Assessment-focused retrieval
from typing import List, Dict, Optional
import json
import numpy as np
from config.constants import supabase, MODEL_NAME

# ============================================================================
# CORE FUNCTIONS FOR ASSESSMENT APP
# ============================================================================

def get_document_list() -> List[Dict]:
    """
    Get all documents uploaded by user.
    Called when: User first opens the app
    
    Returns:
        List of documents with metadata
    """
    response = supabase.table("documents") \
        .select("id, title, total_pages, total_sections, total_chunks, created_at") \
        .order("created_at", desc=True) \
        .execute()
    
    return response.data if response.data else []


def build_tree(flat_nodes: List[dict]) -> List[dict]:
    """
    Convert flat ToC nodes into nested tree structure for UI.
    Called when: User selects a document and needs to browse sections
    
    This enables the UI to show expandable/collapsible tree:
    📖 Chapter 1
       └ 📄 Section 1.1
       └ 📄 Section 1.2
          └ 📄 Subsection 1.2.1
    """
    if not flat_nodes:
        return []
    
    nodes = sorted(flat_nodes, key=lambda x: (x['page_start'], x['level']))
    
    for node in nodes:
        node['children'] = []
    
    root_nodes = []
    stack = []
    
    for node in nodes:
        while stack and stack[-1]['level'] >= node['level']:
            stack.pop()
        
        if stack:
            stack[-1]['children'].append(node)
        else:
            root_nodes.append(node)
        
        stack.append(node)
    
    return root_nodes


def get_document_toc(document_id: str) -> List[dict]:
    """
    Get hierarchical table of contents for a document.
    Called when: User selects a document and wants to choose sections to study
    
    Returns:
        Nested tree structure for UI display
        
    Example UI flow:
        1. User uploads PDF → ingestion runs
        2. User clicks "Study" → calls get_document_toc()
        3. UI shows tree → user expands chapters
        4. User selects "Chapter 3" → calls get_section_content()
    """
    result = supabase.table("toc_nodes") \
        .select("id, node_id, title, level, page_start, page_end") \
        .eq("document_id", document_id) \
        .order("page_start") \
        .execute()
    
    return build_tree(result.data)


def get_section_content(
    document_id: str,
    node_id: str,
    include_children: bool = True
) -> Dict:
    """
    Get all content for a selected section.
    Called when: User selects a section to be tested on
    
    This is your MAIN retrieval function!
    
    Args:
        document_id: UUID of the document
        node_id: The node_id string (e.g., "h2-1-1__section-title")
        include_children: Whether to include subsections
        
    Returns:
        {
            "section": {...},      # ToC node metadata
            "text": "...",         # Full concatenated text
            "chunks": [...],       # Individual chunks for citation
            "total_words": 1234,
            "page_range": "45-67"
        }
        
    Example:
        >>> # User selects "Chapter 3: Newton's Laws"
        >>> content = get_section_content(doc_id, "h1-3__newtons-laws")
        >>> # Now generate questions from content['text']
    """
    # 1. Get the ToC node
    node_result = supabase.table("toc_nodes") \
        .select("*") \
        .eq("document_id", document_id) \
        .eq("node_id", node_id) \
        .single() \
        .execute()
    
    node = node_result.data
    
    # 2. Determine which nodes to include
    if include_children:
        # Get all descendants (subsections)
        all_nodes = supabase.table("toc_nodes") \
            .select("id") \
            .eq("document_id", document_id) \
            .gte("page_start", node["page_start"]) \
            .lte("page_end", node["page_end"]) \
            .gte("level", node["level"]) \
            .execute() \
            .data
        
        toc_node_ids = [n["id"] for n in all_nodes]
    else:
        toc_node_ids = [node["id"]]
    
    # 3. Get chunks
    chunks = supabase.table("chunks") \
        .select("id, chunk_seq, section_title, text, page_start, page_end") \
        .eq("document_id", document_id) \
        .in_("toc_node_id", toc_node_ids) \
        .order("page_start, chunk_seq") \
        .execute() \
        .data
    
    # 4. Combine text
    full_text = "\n\n".join(c["text"] for c in chunks)
    
    return {
        "section": node,
        "text": full_text,
        "chunks": chunks,
        "total_words": len(full_text.split()),
        "total_chars": len(full_text),
        "page_range": f"{node['page_start']}-{node['page_end']}"
    }


def get_ancestor_path(document_id: str, node_id: str) -> List[str]:
    """
    Get the full path to a section (for breadcrumbs in UI).
    Called when: Showing user where they are in the document
    
    Returns:
        ["Part 1", "Chapter 3", "Section 3.2"]
        
    Example UI:
        Home > My Textbook > Part 1 > Chapter 3 > Section 3.2
    """
    # Get all nodes
    all_nodes = supabase.table("toc_nodes") \
        .select("*") \
        .eq("document_id", document_id) \
        .order("page_start") \
        .execute() \
        .data
    
    # Find target node
    target = next((n for n in all_nodes if n['node_id'] == node_id), None)
    if not target:
        return []
    
    # Find ancestors
    path = []
    for node in all_nodes:
        if (node['page_start'] <= target['page_start'] 
            and node['page_end'] >= target['page_end']
            and node['level'] < target['level']):
            path.append(node['title'])
    
    path.append(target['title'])
    return path

def search_within_section(
    document_id: str,
    node_id: str,
    query: str,
    top_k: int = 3
) -> List[Dict]:
    """
    Search within a specific section using semantic similarity.
    Called when: User asks a follow-up question about the section
    
    This is OPTIONAL - only needed if you add chat functionality
    
    Args:
        document_id: Document UUID
        node_id: Section node_id
        query: User's question
        top_k: Number of relevant chunks
        
    Returns:
        Top-k most relevant chunks with similarity scores
        
    Example:
        >>> # User studying Chapter 3, asks: "What's Newton's second law?"
        >>> results = search_within_section(doc_id, "h1-3__physics", 
        ...                                  "Newton's second law", top_k=2)
    """
    from ingest.embedding_import import generate_embeddings
    
    # Get the section's ToC node
    node_result = supabase.table("toc_nodes") \
        .select("id") \
        .eq("document_id", document_id) \
        .eq("node_id", node_id) \
        .single() \
        .execute()
    
    toc_node_db_id = node_result.data['id']
    
    # Generate query embedding
    query_embedding = generate_embeddings(
        [query],
        show_progress=False
    )[0]
    
    # Get all chunks from this section
    chunks = supabase.table("chunks") \
        .select("id, chunk_id, text, section_title, page_start, page_end, embedding") \
        .eq("document_id", document_id) \
        .eq("toc_node_id", toc_node_db_id) \
        .execute() \
        .data
    
    # Compute similarities
    results = []
    for chunk in chunks:
        # Convert embedding to numpy array
        chunk_emb_raw = chunk['embedding']
        if isinstance(chunk_emb_raw, str):
            chunk_emb = np.array(json.loads(chunk_emb_raw))
        elif isinstance(chunk_emb_raw, list):
            chunk_emb = np.array(chunk_emb_raw)
        else:
            chunk_emb = chunk_emb_raw
        
        chunk_emb = chunk_emb.astype(np.float32)
        similarity = float(np.dot(query_embedding, chunk_emb))
        
        results.append({
            'chunk_id': chunk['chunk_id'],
            'text': chunk['text'],
            'section_title': chunk['section_title'],
            'page_start': chunk['page_start'],
            'page_end': chunk['page_end'],
            'similarity': similarity
        })
    
    # Sort by similarity and return top-k
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

def get_section_progress(user_id: str, document_id: str) -> List[Dict]:
    """
    Get user's progress across all sections.
    Called when: Showing user which sections they've mastered
    
    Returns:
        List of sections with progress stats
        
    Future enhancement - requires user_section_progress table
    """
    # TODO: Implement after adding progress tracking table
    pass


def update_section_progress(
    user_id: str,
    toc_node_id: int,
    correct: bool,
    total_questions: int
):
    """
    Update user's progress on a section after assessment.
    Called when: User completes a quiz on a section
    
    Future enhancement
    """
    # TODO: Implement after adding progress tracking
    pass

if __name__ == "__main__":
    print("="*80)
    print("ASSESSMENT APP - RETRIEVAL WORKFLOW")
    print("="*80)
    
    # Step 1: User selects document
    print("\n📚 Step 1: Get available documents")
    docs = get_document_list()
    if docs:
        print(f"Found {len(docs)} documents:")
        for doc in docs[:3]:
            print(f"  - {doc['title']} ({doc['total_sections']} sections)")
    
    # Step 2: User browses ToC
    if docs:
        document_id = docs[0]['id']
        print(f"\n📖 Step 2: Get ToC for: {docs[0]['title']}")
        toc = get_document_toc(document_id)
        print(f"Found {len(toc)} chapters")
        if toc:
            print(f"First chapter: {toc[0]['title']}")
            if toc[0].get('children'):
                print(f"  Subsections: {len(toc[0]['children'])}")
    
    # Step 3: User selects section
    print(f"\n🎯 Step 3: User selects a section to study")
    if toc:
        # Get first H2 section
        first_section = None
        for chapter in toc:
            if chapter.get('children'):
                first_section = chapter['children'][0]
                break
        
        if first_section:
            print(f"Selected: {first_section['title']}")
            content = get_section_content(
                document_id, 
                first_section['node_id'],
                include_children=False
            )
            
            print(f"\n📊 Section content:")
            print(f"  Total words: {content['total_words']:,}")
            print(f"  Pages: {content['page_range']}")
            print(f"  Chunks: {len(content['chunks'])}")
            print(f"\n  Preview: {content['text'][:200]}...")
    
    print("\n" + "="*80)
    print("✅ Retrieval workflow complete!")