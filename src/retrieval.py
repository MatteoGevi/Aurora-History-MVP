"""
Retrieval module using Supabase client for pgvector queries.
Reuses existing pipeline components for consistency.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

# Reuse existing pipeline components
from constants import supabase, EMBEDDING_MODEL
from embedding_import import generate_embeddings
from toc_chunk import flatten, get_ancestor_titles


class DocumentRetriever:
    """Handle vector similarity search and retrieval from Supabase"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize retriever with embedding model.
        Uses the SAME model from your ingestion pipeline for consistency.
        
        Args:
            model_name: Hugging Face model name (defaults to EMBEDDING_MODEL from constants)
        """
        from sentence_transformers import SentenceTransformer
        
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded (dimension: {self.embedding_dim})")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for query text.
        Reuses generate_embeddings() from pipeline for consistency.
        """
        # Use your existing embedding function with single text
        embedding = generate_embeddings(
            texts=[text],
            model_name=EMBEDDING_MODEL,
            batch_size=1,
            normalize=True,
            show_progress=False
        )
        return embedding[0].tolist()
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Basic semantic search using cosine similarity
        
        Args:
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of matching chunks with metadata
        """
        query_embedding = self.embed_query(query)
        
        # Supabase RPC call for vector similarity search
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
        self,
        query: str,
        top_k: int = 3,
        context_window: int = 2
    ) -> List[Dict]:
        """
        Retrieve chunks with surrounding context.
        Uses chunk_seq to get neighboring chunks from same section.
        
        Args:
            query: Search query
            top_k: Number of initial matches
            context_window: Number of chunks before/after to include
        
        Returns:
            List of results with context chunks
        """
        # Get initial matches
        initial_results = self.semantic_search(query, top_k=top_k)
        
        if not initial_results:
            return []
        
        enriched_results = []
        
        for result in initial_results:
            chunk_seq = result['chunk_seq']
            toc_node_id = result['toc_node_id']
            
            # Get surrounding chunks from same section
            context_response = supabase.from_('chunks') \
                .select('chunk_id, text, chunk_seq, section_title, page_start, page_end') \
                .eq('toc_node_id', toc_node_id) \
                .gte('chunk_seq', chunk_seq - context_window) \
                .lte('chunk_seq', chunk_seq + context_window) \
                .order('chunk_seq') \
                .execute()
            
            # Mark which chunk is the match
            context_chunks = []
            for chunk in context_response.data:
                chunk['position'] = 'match' if chunk['chunk_seq'] == chunk_seq else \
                                   'before' if chunk['chunk_seq'] < chunk_seq else 'after'
                context_chunks.append(chunk)
            
            enriched_results.append({
                'match': result,
                'context': context_chunks
            })
        
        return enriched_results
    
    def get_full_section_context(self, chunk_id: str) -> Dict:
        """
        Get a chunk with its full ToC hierarchy context.
        Uses get_ancestor_titles logic from toc_chunk.py
        
        Args:
            chunk_id: ID of the chunk to contextualize
            
        Returns:
            Dict with chunk, ancestors, and siblings
        """
        # Get the chunk
        chunk_response = supabase.from_('chunks') \
            .select('*, toc_nodes!inner(*)') \
            .eq('chunk_id', chunk_id) \
            .single() \
            .execute()
        
        if not chunk_response.data:
            return None
        
        chunk = chunk_response.data
        toc_node = chunk['toc_nodes']
        
        # Get all ToC nodes for hierarchy
        all_toc = supabase.from_('toc_nodes') \
            .select('*') \
            .order('page_start') \
            .execute()
        
        flat_toc = all_toc.data
        
        # Build hierarchy using your existing function logic
        ancestors = []
        current_level = toc_node['level']
        current_page = toc_node['page_start']
        
        for n in reversed(flat_toc):
            if n['page_start'] <= current_page and n['level'] < current_level:
                ancestors.insert(0, {
                    'level': n['level'],
                    'title': n['title'],
                    'node_id': n['node_id']
                })
                current_level = n['level']
                if current_level == 1:
                    break
        
        # Get sibling chunks from same section
        siblings_response = supabase.from_('chunks') \
            .select('chunk_id, chunk_seq, text') \
            .eq('toc_node_id', chunk['toc_node_id']) \
            .order('chunk_seq') \
            .execute()
        
        return {
            'chunk': chunk,
            'section': toc_node,
            'ancestors': ancestors,
            'siblings': siblings_response.data,
            'hierarchy_path': ' > '.join(a['title'] for a in ancestors) + f" > {toc_node['title']}"
        }
    
    def hierarchical_search(
        self,
        query: str,
        top_sections: int = 3,
        chunks_per_section: int = 3
    ) -> List[Dict]:
        """
        Two-stage retrieval: sections first, then chunks.
        Mimics your ToC-first approach from ingestion.
        
        Args:
            query: Search query
            top_sections: Number of top sections to retrieve
            chunks_per_section: Number of chunks per section
        
        Returns:
            List of sections with their best chunks
        """
        query_embedding = self.embed_query(query)
        
        # Use RPC for hierarchical search
        response = supabase.rpc(
            'hierarchical_search',
            {
                'query_embedding': query_embedding,
                'top_sections': top_sections,
                'chunks_per_section': chunks_per_section
            }
        ).execute()
        
        return response.data if response.data else []
    
    def filtered_search(
        self,
        query: str,
        top_k: int = 5,
        page_range: Optional[Tuple[int, int]] = None,
        max_level: Optional[int] = None,
        section_pattern: Optional[str] = None
    ) -> List[Dict]:
        """
        Semantic search with filters on hierarchy metadata.
        Uses the same level/page structure from your ToC.
        
        Args:
            query: Search text
            top_k: Number of results
            page_range: Tuple of (min_page, max_page) or None
            max_level: Maximum heading level (e.g., 3 for up to h3)
            section_pattern: Pattern for section titles (case-insensitive)
        
        Returns:
            List of filtered results
        """
        query_embedding = self.embed_query(query)
        
        # If we have page_range, use optimized RPC
        if page_range and not max_level and not section_pattern:
            response = supabase.rpc(
                'match_chunks_by_page',
                {
                    'query_embedding': query_embedding,
                    'min_page': page_range[0],
                    'max_page': page_range[1],
                    'match_threshold': 0.3,
                    'match_count': top_k
                }
            ).execute()
            return response.data if response.data else []
        
        # Otherwise, get more results and filter in Python
        response = supabase.rpc(
            'match_chunks',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.0,
                'match_count': 100  # Get more for filtering
            }
        ).execute()
        
        results = response.data if response.data else []
        
        # Apply filters
        filtered = []
        for r in results:
            # Page range filter
            if page_range:
                if not (page_range[0] <= r['page_start'] <= page_range[1]):
                    continue
            
            # Level filter
            if max_level:
                if r['level'] > max_level:
                    continue
            
            # Section pattern filter
            if section_pattern:
                if section_pattern.lower() not in r['section_title'].lower():
                    continue
            
            filtered.append(r)
            
            if len(filtered) >= top_k:
                break
        
        return filtered
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Combine semantic and keyword search
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic score (0-1)
        
        Returns:
            List of results ranked by hybrid score
        """
        query_embedding = self.embed_query(query)
        keyword_weight = 1 - semantic_weight
        
        # Use RPC for hybrid search
        response = supabase.rpc(
            'hybrid_search',
            {
                'query_embedding': query_embedding,
                'query_text': query,
                'semantic_weight': semantic_weight,
                'keyword_weight': keyword_weight,
                'match_count': top_k
            }
        ).execute()
        
        return response.data if response.data else []
    
    def get_toc_structure(self) -> List[Dict]:
        """
        Get the full table of contents structure.
        Returns flattened ToC like your flatten() function.
        """
        response = supabase.from_('toc_nodes') \
            .select('*') \
            .order('page_start') \
            .execute()
        
        return response.data if response.data else []
    
    def get_section_chunks(self, section_node_id: str, limit: int = 50) -> List[Dict]:
        """
        Get all chunks from a specific ToC section.
        Useful for browsing full sections after finding relevant chunks.
        
        Args:
            section_node_id: The node_id from toc_nodes table (e.g., "h3-5-1-3__from-foundation...")
            limit: Max chunks to return
        """
        # First, get the toc_node database ID
        toc_response = supabase.from_('toc_nodes') \
            .select('id') \
            .eq('node_id', section_node_id) \
            .single() \
            .execute()
        
        if not toc_response.data:
            return []
        
        toc_id = toc_response.data['id']
        
        # Get chunks
        response = supabase.from_('chunks') \
            .select('*') \
            .eq('toc_node_id', toc_id) \
            .order('chunk_seq') \
            .limit(limit) \
            .execute()
        
        return response.data if response.data else []
    
    def get_database_stats(self) -> Dict:
        """Get statistics about your vector database"""
        try:
            response = supabase.rpc('get_vector_stats').execute()
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"⚠️  Could not get stats (run SQL functions first): {e}")
            return {}


# Utility functions for display and analysis
def format_results(results: List[Dict], show_text_length: int = 150):
    """Pretty print search results"""
    if not results:
        print("No results found.")
        return
    
    for i, r in enumerate(results, 1):
        similarity = r.get('similarity', 0)
        print(f"\n{i}. [{similarity:.3f}] {r['section_title']}")
        print(f"   Level {r['level']} | Pages {r['page_start']}-{r['page_end']} | Chunk {r.get('chunk_seq', '?')}")
        print(f"   ID: {r['chunk_id']}")
        text = r['text'][:show_text_length]
        print(f"   {text}{'...' if len(r['text']) > show_text_length else ''}")


def format_hierarchical_results(results: List[Dict]):
    """Format results from hierarchical_search"""
    if not results:
        print("No results found.")
        return
    
    for i, section_data in enumerate(results, 1):
        print(f"\n{'='*70}")
        print(f"📚 Section {i}: {section_data['section_title']}")
        print(f"   Level {section_data['section_level']} | "
              f"Pages {section_data['section_page_start']}-{section_data['section_page_end']}")
        print(f"   Avg similarity: {section_data['avg_similarity']:.3f} "
              f"({section_data['chunk_count']} chunks)")
        print(f"{'='*70}")
        
        chunks = section_data.get('chunks', [])
        for j, chunk in enumerate(chunks, 1):
            print(f"\n  {j}. [{chunk['similarity']:.3f}] Chunk {chunk['chunk_seq']}")
            print(f"     Pages {chunk['page_start']}-{chunk['page_end']}")
            text_preview = chunk['text'][:120].replace('\n', ' ')
            print(f"     {text_preview}...")


def analyze_result_quality(results: List[Dict]):
    """Analyze the quality and diversity of search results"""
    if not results:
        print("No results to analyze")
        return
    
    similarities = [r.get('similarity', 0) for r in results]
    sections = [r['section_title'] for r in results]
    levels = [r['level'] for r in results]
    pages = [r['page_start'] for r in results]
    
    print("\n" + "="*60)
    print("Result Quality Analysis")
    print("="*60)
    print(f"Total results: {len(results)}")
    print(f"Average similarity: {np.mean(similarities):.3f}")
    print(f"Similarity range: {np.min(similarities):.3f} - {np.max(similarities):.3f}")
    print(f"Unique sections: {len(set(sections))} / {len(results)}")
    
    # Level distribution
    from collections import Counter
    level_counts = Counter(levels)
    print(f"Level distribution: {dict(level_counts)}")
    print(f"Page spread: {np.min(pages)} - {np.max(pages)} ({np.max(pages) - np.min(pages)} pages)")
    
    # Check for redundancy
    unique_texts = len(set(r['text'][:100] for r in results))
    print(f"Unique content: {unique_texts} / {len(results)}")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("TESTING DOCUMENT RETRIEVAL")
    print("="*70)
    
    # Initialize retriever (uses EMBEDDING_MODEL from constants.py)
    retriever = DocumentRetriever()
    
    # Check database stats
    print("\n📊 Database Statistics:")
    stats = retriever.get_database_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test 1: Basic search
    print("\n" + "="*70)
    print("TEST 1: BASIC SEMANTIC SEARCH")
    print("="*70)
    query = "What are neural networks?"
    print(f"\nQuery: '{query}'")
    results = retriever.semantic_search(query, top_k=5)
    format_results(results)
    analyze_result_quality(results)
    
    # Test 2: Search with context
    print("\n" + "="*70)
    print("TEST 2: SEARCH WITH CONTEXT")
    print("="*70)
    query = "machine learning algorithms"
    print(f"\nQuery: '{query}'")
    results = retriever.search_with_context(query, top_k=2, context_window=1)
    
    for i, result in enumerate(results, 1):
        match = result['match']
        print(f"\n📍 Match {i}: {match['section_title']}")
        print(f"   Similarity: {match.get('similarity', 0):.3f}")
        print(f"   Pages {match['page_start']}-{match['page_end']}")
        print("\n   Context:")
        
        for chunk in result['context']:
            marker = ">>>" if chunk['position'] == 'match' else "   "
            pos_label = f"[{chunk['position'].upper():^7}]"
            print(f"\n   {marker} {pos_label} Chunk {chunk['chunk_seq']}")
            text_preview = chunk['text'][:100].replace('\n', ' ')
            print(f"   {marker} {text_preview}...")
    
    # Test 3: Hierarchical search
    print("\n" + "="*70)
    print("TEST 3: HIERARCHICAL SEARCH")
    print("="*70)
    query = "deep learning"
    print(f"\nQuery: '{query}'")
    results = retriever.hierarchical_search(query, top_sections=2, chunks_per_section=3)
    format_hierarchical_results(results)