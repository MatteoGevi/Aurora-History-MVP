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
        # 1.5x is sharp enough for a container-width display and ~44% faster than 2x
        mat = fitz.Matrix(1.5, 1.5)
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
    if prompt := st.chat_input("Ask a question or discuss the current section..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = "This is a placeholder response. Your model will be integrated here to provide intelligent feedback based on the current section context."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()