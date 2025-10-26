import streamlit as st
import pymupdf as fitz
from PIL import Image
import io

def load_pdf(pdf_file):
    """Load PDF and extract table of contents"""
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        toc = doc.get_toc()
        return doc, toc
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None, []

def render_pdf_page(doc, page_num):
    """Render a specific PDF page as an image"""
    try:
        page = doc[page_num]
        # Render page at 2x resolution for better quality
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        return img
    except Exception as e:
        st.error(f"Error rendering page: {e}")
        return None

def display_toc(toc):
    """Display clickable table of contents"""
    st.markdown("### 📚 Table of Contents")
    
    if not toc:
        st.info("No table of contents found in this PDF")
        return
    
    for i, item in enumerate(toc):
        level, title, page = item
        indent = "　" * (level - 1)  # Indent based on heading level
        
        # Create button for each ToC item
        button_label = f"{indent}{title}"
        if st.button(button_label, key=f"toc_{i}"):
            st.session_state.current_page = page - 1
            st.rerun()

def build_tree(flat_nodes: List[dict]) -> List[dict]:
    """
    Convert flat ToC nodes into nested tree structure.
    
    Input: [
        {"id": 1, "node_id": "h1-1", "level": 1, "title": "Chapter 1", ...},
        {"id": 2, "node_id": "h2-1-1", "level": 2, "title": "Section 1.1", ...},
        {"id": 3, "node_id": "h2-1-2", "level": 2, "title": "Section 1.2", ...},
    ]
    
    Output: [
        {
            "id": 1, 
            "title": "Chapter 1",
            "children": [
                {"id": 2, "title": "Section 1.1", "children": []},
                {"id": 3, "title": "Section 1.2", "children": []}
            ]
        }
    ]
    """
    if not flat_nodes:
        return []
    
    # Sort by page_start to maintain document order
    nodes = sorted(flat_nodes, key=lambda x: (x['page_start'], x['level']))
    
    # Add children array to each node
    for node in nodes:
        node['children'] = []
    
    # Build tree using stack
    root_nodes = []
    stack = []
    
    for node in nodes:
        # Pop nodes from stack that aren't ancestors
        while stack and stack[-1]['level'] >= node['level']:
            stack.pop()
        
        # Add to parent or root
        if stack:
            stack[-1]['children'].append(node)
        else:
            root_nodes.append(node)
        
        stack.append(node)
    
    return root_nodes

def display_chat():
    """Display chat interface"""
    st.markdown("### 💬 Learning Assistant")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the content..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response (placeholder for now)
        current_section = "the selected section"
        if st.session_state.toc and st.session_state.current_page < len(st.session_state.pdf_doc):
            # Find current section based on page
            for item in reversed(st.session_state.toc):
                if item[2] - 1 <= st.session_state.current_page:
                    current_section = item[1]
                    break
        
        response = f"I see you're on {current_section}. This is a placeholder response. Your SML model will be integrated here to provide intelligent feedback based on the user's answer and the current section context."
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()