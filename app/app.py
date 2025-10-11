import streamlit as st
import pymupdf as fitz
from PIL import Image
import io

from components import load_pdf, render_pdf_page, display_toc, display_chat

# Page configuration
st.set_page_config(
    page_title="AI Learning Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .toc-container {
        height: 70vh;
        overflow-y: auto;
        padding: 10px;
        border-radius: 8px;
        background-color: #f8f9fa;
    }
    .toc-item {
        padding: 8px;
        margin: 4px 0;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .toc-item:hover {
        background-color: #e9ecef;
    }
    .stButton>button {
        width: 100%;
        text-align: left;
        border: none;
        background-color: transparent;
        padding: 8px 12px;
    }
    .stButton>button:hover {
        background-color: #e9ecef;
    }
    .chat-container {
        height: 70vh;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        background-color: #ffffff;
    }
    /* Hide the default streamlit padding */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'pdf_doc' not in st.session_state:
    st.session_state.pdf_doc = None
if 'toc' not in st.session_state:
    st.session_state.toc = []
if 'show_toc' not in st.session_state:
    st.session_state.show_toc = True

# Main app header
col_title, col_toggle = st.columns([4, 1])
with col_title:
    st.title("🦉 AI Engineering Learning Platform")
    st.markdown("*MVP - Building Applications with Foundation Models*")
with col_toggle:
    if st.button("📚 Toggle ToC" if not st.session_state.show_toc else "✖️ Close ToC"):
        st.session_state.show_toc = not st.session_state.show_toc
        st.rerun()

# File uploader
uploaded_file = st.file_uploader("Upload your PDF document", type=['pdf'])

if uploaded_file is not None:
    # Load PDF if not already loaded or if it's a new file
    if st.session_state.pdf_doc is None or uploaded_file.name != st.session_state.get('current_pdf_name'):
        with st.spinner("Loading PDF..."):
            doc, toc = load_pdf(uploaded_file)
            if doc:
                st.session_state.pdf_doc = doc
                st.session_state.toc = toc
                st.session_state.current_pdf_name = uploaded_file.name
                st.success(f"✅ Loaded: {uploaded_file.name}")
    
    # Create dynamic column layout based on ToC visibility
    if st.session_state.show_toc:
        col1, col2, col3 = st.columns([1, 2, 1.5])
    else:
        col1 = None
        col2, col3 = st.columns([2, 1.5])
    
    # Column 1: Table of Contents (if visible)
    if st.session_state.show_toc and col1:
        with col1:
            st.markdown("### 📚 Table of Contents")
            display_toc(st.session_state.toc)
            
            # Page navigation in ToC column
            st.markdown("---")
            st.markdown("### 📄 Page Navigation")
            total_pages = len(st.session_state.pdf_doc)
            page_num = st.number_input(
                "Go to page:", 
                min_value=1, 
                max_value=total_pages,
                value=st.session_state.current_page + 1,
                key="page_input"
            )
            if page_num - 1 != st.session_state.current_page:
                st.session_state.current_page = page_num - 1
                st.rerun()
            
            st.text(f"Page {st.session_state.current_page + 1} of {total_pages}")
    
    # Column 2: PDF Viewer
    with col2:
        st.markdown("### 📖 Document Viewer")
        img = render_pdf_page(st.session_state.pdf_doc, st.session_state.current_page)
        if img:
            st.image(img, use_container_width=True)
        
        # Navigation buttons
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        with nav_col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_page == 0):
                st.session_state.current_page -= 1
                st.rerun()
        with nav_col2:
            total_pages = len(st.session_state.pdf_doc)
            st.markdown(f"<center>{st.session_state.current_page + 1} / {total_pages}</center>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("Next ➡️", disabled=st.session_state.current_page >= len(st.session_state.pdf_doc) - 1):
                st.session_state.current_page += 1
                st.rerun()
        
        # Show page navigation if ToC is hidden
        if not st.session_state.show_toc:
            st.markdown("---")
            total_pages = len(st.session_state.pdf_doc)
            page_num = st.number_input(
                "Go to page:", 
                min_value=1, 
                max_value=total_pages,
                value=st.session_state.current_page + 1,
                key="page_input_main"
            )
            if page_num - 1 != st.session_state.current_page:
                st.session_state.current_page = page_num - 1
                st.rerun()
    
    # Column 3: Chat Interface
    with col3:
        display_chat()

else:
    st.info("👆 Please upload a PDF document to get started")