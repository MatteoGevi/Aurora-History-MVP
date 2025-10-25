from config.constants import supabase



def get_section_chunks(document_id: str, node_id: str):
    """Get all chunks for a selected ToC node"""
    return supabase.table("chunks") \
        .select("*") \
        .eq("document_id", document_id) \
        .eq("toc_node_id", get_db_id(node_id)) \
        .order("chunk_seq") \
        .execute()

def get_toc_tree(document_id: str):
    """Return hierarchical ToC for user to browse"""
    nodes = supabase.table("toc_nodes") \
        .select("*") \
        .eq("document_id", document_id) \
        .execute()
    
    # Rebuild tree structure for UI
    return build_tree(nodes.data)

def prepare_assessment_context(
    document_id: str, 
    selected_node_id: str,
    include_subsections: bool = True
) -> dict:
    """
    Prepare context for LLM assessment generation.
    
    Returns:
        {
            "section_title": "Chapter 3: Newton's Laws",
            "full_text": "...",  # All chunks concatenated
            "chunks": [...],     # Individual chunks for citation
            "hierarchy": "Part 1 > Unit 2 > Chapter 3",
            "page_range": "45-67",
            "total_words": 8234
        }
    """
    # 1. Get ToC node
    node = get_toc_node(selected_node_id)
    
    # 2. Get hierarchy path
    hierarchy = get_ancestor_titles(node)
    
    # 3. Get chunks (with subsections if requested)
    if include_subsections:
        chunks = get_section_with_descendants(selected_node_id)
    else:
        chunks = get_section_chunks(selected_node_id)
    
    # 4. Concatenate text with headers
    full_text = build_hierarchical_text(chunks, hierarchy)
    
    return {
        "section_title": node["title"],
        "full_text": full_text,
        "chunks": chunks,
        "hierarchy": " > ".join(hierarchy),
        "page_range": f"{node['page_start']}-{node['page_end']}",
        "total_words": len(full_text.split())
    }