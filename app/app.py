# app.py - MINIMAL INTEGRATION (keeping your existing code)
import sys
from pathlib import Path

# Add parent directory to path so we can import from src/
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import streamlit as st
import pymupdf as fitz
from PIL import Image
import io
from typing import List

from components import load_pdf, render_pdf_page, display_chat
from src.retrieval import get_document_list, get_document_toc as get_db_toc, get_section_content
from src.assessment import generate_questions
from src.evaluation import evaluate_answer

# Page configuration
st.set_page_config(
    page_title="AI Learning Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (your existing CSS)
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
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state (your existing + new for assessment)
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
# NEW: Assessment mode
if 'assessment_mode' not in st.session_state:
    st.session_state.assessment_mode = False
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = None
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None

# Helper function to display ToC from database
def display_db_toc():
    """Display ToC from ingested documents with 'Study' buttons"""
    st.markdown("### 📚 Study Sections")
    
    # Get documents from database
    docs = get_document_list()
    
    if not docs:
        st.info("No documents ingested yet. Upload and ingest first.")
        return
    
    # For now, use first document (later can add selector)
    doc = docs[0]
    toc = get_db_toc(doc['id'])
    
    if not toc:
        st.info("No table of contents found")
        return
    
    def render_toc_node(nodes, level=0):
        for node in nodes:
            indent = "  " * level
            
            # Create columns for title and button
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"{indent}**{node['title']}**")
                st.caption(f"{indent}Pages {node['page_start']}-{node['page_end']}")
            
            with col2:
                if st.button("Study", key=f"study_{node['node_id']}"):
                    st.session_state.assessment_mode = True
                    st.session_state.selected_section = node
                    st.session_state.current_page = node['page_start'] - 1  # Jump to page
                    st.rerun()
            
            # Render children
            if node.get('children'):
                render_toc_node(node['children'], level + 1)
    
    render_toc_node(toc)

# Helper to display PDF ToC (your existing function - keep it)
def display_toc(toc):
    """Display clickable table of contents"""
    st.markdown("### 📚 Table of Contents")
    
    if not toc:
        st.info("No table of contents found in this PDF")
        return
    
    for i, item in enumerate(toc):
        level, title, page = item
        indent = "　" * (level - 1)
        
        button_label = f"{indent}{title}"
        if st.button(button_label, key=f"toc_{i}"):
            st.session_state.current_page = page - 1
            st.rerun()

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
    # Load PDF if not already loaded
    if st.session_state.pdf_doc is None or uploaded_file.name != st.session_state.get('current_pdf_name'):
        with st.spinner("Loading PDF..."):
            doc, toc = load_pdf(uploaded_file)
            if doc:
                st.session_state.pdf_doc = doc
                st.session_state.toc = toc
                st.session_state.current_pdf_name = uploaded_file.name
                st.success(f"✅ Loaded: {uploaded_file.name}")
    
    # Create dynamic column layout
    if st.session_state.show_toc:
        col1, col2, col3 = st.columns([1, 2, 1.5])
    else:
        col1 = None
        col2, col3 = st.columns([2, 1.5])
    
    # Column 1: Table of Contents
    if st.session_state.show_toc and col1:
        with col1:
            # Toggle between PDF ToC and Study ToC
            toc_mode = st.radio("ToC Mode:", ["PDF Navigation", "Study Mode"], horizontal=True)
            
            if toc_mode == "PDF Navigation":
                display_toc(st.session_state.toc)
            else:
                display_db_toc()
            
            # Page navigation
            st.markdown("---")
            st.markdown("### 📄 Page Navigation")
            total_pages = len(st.session_state.pdf_doc)
            
            # Use on_change to avoid conflicts with button navigation
            def update_page():
                new_page = st.session_state.page_input_widget - 1
                if new_page != st.session_state.current_page:
                    st.session_state.current_page = new_page
            
            st.number_input(
                "Go to page:", 
                min_value=1, 
                max_value=total_pages,
                value=st.session_state.current_page + 1,
                key="page_input_widget",
                on_change=update_page
            )
            
            st.text(f"Page {st.session_state.current_page + 1} of {total_pages}")
    
    # Column 2: PDF Viewer (your existing code - unchanged)
    with col2:
        st.markdown("### 📖 Document Viewer")
        img = render_pdf_page(st.session_state.pdf_doc, st.session_state.current_page)
        if img:
            st.image(img, use_container_width=True)
        
        # Navigation buttons
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        with nav_col1:
            if st.button("⬅️ Previous", key="prev_page", disabled=st.session_state.current_page == 0):
                st.session_state.current_page -= 1
                st.rerun()
        with nav_col2:
            total_pages = len(st.session_state.pdf_doc)
            st.markdown(f"<center>{st.session_state.current_page + 1} / {total_pages}</center>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("Next ➡️", key="next_page", disabled=st.session_state.current_page >= len(st.session_state.pdf_doc) - 1):
                st.session_state.current_page += 1
                st.rerun()
    
    # Column 3: Chat/Assessment Interface
    with col3:
        # NEW: Show assessment if in assessment mode
        if st.session_state.assessment_mode and st.session_state.selected_section:
            st.markdown("### 📝 Assessment")
            
            section = st.session_state.selected_section
            st.info(f"**Section:** {section['title']}\n\n**Pages:** {section['page_start']}-{section['page_end']}")
            
            # Step 1: Generate questions
            if not st.session_state.questions:
                num_q = st.slider("Number of questions:", 1, 5, 3)
                difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard", "mixed"])
                
                if st.button("🎲 Generate Questions", type="primary"):
                    with st.spinner("Generating..."):
                        docs = get_document_list()
                        result = generate_questions(
                            document_id=docs[0]['id'],
                            node_id=section['node_id'],
                            num_questions=num_q,
                            difficulty=difficulty
                        )
                        st.session_state.questions = result.get('questions', [])
                        st.rerun()
            
            # Step 2: Answer question
            elif st.session_state.questions and not st.session_state.evaluation_result:
                question = st.session_state.questions[0]
                
                st.markdown(f"**Question:**\n{question['question']}")
                st.caption(f"Difficulty: {question.get('difficulty')} • Pages: {question.get('page_reference')}")
                
                answer = st.text_area("Your Answer:", height=150, key="answer_input")
                
                if st.button("Submit Answer", type="primary"):
                    if answer.strip():
                        with st.spinner("Evaluating..."):
                            docs = get_document_list()
                            evaluation = evaluate_answer(
                                document_id=docs[0]['id'],
                                node_id=section['node_id'],
                                question=question['question'],
                                student_answer=answer
                            )
                            st.session_state.evaluation_result = evaluation
                            st.rerun()
                    else:
                        st.warning("Please enter an answer")
            
            # Step 3: Show results
            else:
                eval_data = st.session_state.evaluation_result
                
                st.markdown("### 📊 Results")
                
                score = eval_data.get('total_score', 0)
                max_score = eval_data.get('max_score', 10)
                percentage = (score / max_score * 100) if max_score > 0 else 0
                
                st.metric("Score", f"{score}/{max_score} ({percentage:.0f}%)")
                
                if eval_data.get('overall_feedback'):
                    st.info(f"💡 {eval_data['overall_feedback']}")
                
                if eval_data.get('criteria_scores'):
                    with st.expander("📋 Detailed Breakdown"):
                        for criterion in eval_data['criteria_scores']:
                            st.markdown(f"**{criterion['criterion_title']}:** {criterion['score']}/2")
                            st.caption(criterion['feedback'])
                
                # Reset button
                if st.button("Try Another Section"):
                    st.session_state.assessment_mode = False
                    st.session_state.selected_section = None
                    st.session_state.questions = []
                    st.session_state.evaluation_result = None
                    st.rerun()
        
        else:
            # Original chat interface
            display_chat()

else:
    st.info("👆 Please upload a PDF document to get started")