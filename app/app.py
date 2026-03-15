# app.py - Load PDFs from Supabase Storage using existing fetch function
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import anthropic
import streamlit as st
import pymupdf as fitz
from PIL import Image
import io
from typing import List

from components import render_pdf_page
from src.retrieval import get_document_list, get_document_toc as get_db_toc, get_section_content
from src.pipeline import run_section_recall
from config.constants import get_supabase, STORAGE_BUCKET, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from ingest.toc_chunk import fetch_pdf_from_storage

# Page configuration
st.set_page_config(
    page_title="AI Learning Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'pdf_doc' not in st.session_state:
    st.session_state.pdf_doc = None
if 'selected_document_id' not in st.session_state:
    st.session_state.selected_document_id = None
if 'show_toc' not in st.session_state:
    st.session_state.show_toc = True
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = None
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None

def get_pdf_from_storage(document_id: str) -> tuple:
    """
    Download PDF from Supabase Storage using the existing fetch function
    
    Returns:
        (pdf_doc, filename) or (None, None) if error
    """
    try:
        # Get document metadata from database
        doc_result = get_supabase().table("documents").select("*").eq("id", document_id).single().execute()
        doc_data = doc_result.data
        
        # Get the filename - prefer original_filename (just filename) over storage_path (bucket/filename)
        filename = doc_data.get('original_filename') or doc_data.get('file_path') or doc_data.get('filename')
        
        # If we got storage_path, strip the bucket prefix if present
        if not filename:
            storage_path = doc_data.get('storage_path')
            if storage_path:
                # storage_path is stored as "bucket/filename", so strip bucket prefix
                if storage_path.startswith(f"{STORAGE_BUCKET}/"):
                    filename = storage_path[len(f"{STORAGE_BUCKET}/"):]
                else:
                    filename = storage_path
        
        if not filename:
            # Fallback: use title + .pdf
            title = doc_data.get('title', document_id)
            filename = title if title.endswith('.pdf') else f"{title}.pdf"
        
        print(f"📥 Fetching: {filename} from bucket: {STORAGE_BUCKET}")
        
        # Use the existing fetch function from toc_chunk.py
        pdf_bytes = fetch_pdf_from_storage(
            supabase_url=SUPABASE_URL,
            service_role_key=SUPABASE_SERVICE_ROLE_KEY,
            bucket=STORAGE_BUCKET,
            filename=filename
        )
        
        # Open with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        return doc, doc_data.get('title', 'Document')
        
    except Exception as e:
        st.error(f"Error loading PDF from storage: {e}")
        filename_display = locals().get('filename', '<unknown>')
        st.info(f"Tried to load: {filename_display} from bucket '{STORAGE_BUCKET}'")
        
        # Try to list available files
        try:
            files = get_supabase().storage.from_(STORAGE_BUCKET).list()
            if files:
                file_names = [f.get('name') for f in files]
                st.info(f"Available files in bucket: {', '.join(file_names)}")
        except:
            pass
        
        return None, None

@st.fragment
def display_db_toc(document_id: str):
    """Collapsible ToC — expand/collapse reruns only this fragment;
    navigation triggers a full rerun to update the PDF viewer."""
    st.markdown("### 📚 Contents")

    toc = get_db_toc(document_id)

    if not toc:
        st.info("No table of contents found")
        return

    selected_node_id = (
        st.session_state.selected_section.get("node_id")
        if st.session_state.get("selected_section")
        else None
    )

    def render_node(nodes, level=0):
        for node in nodes:
            node_id   = node["node_id"]
            has_kids  = bool(node.get("children"))
            exp_key   = f"toc_expanded_{node_id}"
            expanded  = st.session_state.get(exp_key, False)

            pad   = "　" * level
            arrow = ("▼ " if expanded else "▶ ") if has_kids else "    "
            label = f"{pad}{arrow}{node['title']}"

            if st.button(label, key=f"toc_{node_id}", use_container_width=True):
                if has_kids:
                    st.session_state[exp_key] = not expanded
                # always navigate to the node's page
                st.session_state.current_page       = node["page_start"] - 1
                st.session_state.page_input_widget  = node["page_start"]
                st.session_state.selected_section   = node
                st.session_state.evaluation_result  = None
                st.rerun()

            if has_kids and expanded:
                render_node(node["children"], level + 1)

    render_node(toc)

# Main app header
col_title, col_toggle = st.columns([4, 1])
with col_title:
    st.title("🦉 AI Engineering Learning Platform")
    st.markdown("*MVP - Building Applications with Foundation Models*")
with col_toggle:
    if st.button("📚 Toggle ToC" if not st.session_state.show_toc else "✖️ Close ToC"):
        st.session_state.show_toc = not st.session_state.show_toc
        st.rerun()

# Document selector
st.markdown("---")
try:
    docs = get_document_list()
except Exception as e:
    st.error(f"Could not reach the database: {e}")
    if st.button("Retry"):
        st.rerun()
    st.stop()

if not docs:
    st.warning("⚠️ No documents found in database. Please ingest a document first.")
    st.info("Run your ingestion script to add documents to the database.")
    st.stop()

# Create document selector
doc_options = {doc['title']: doc['id'] for doc in docs}
selected_title = st.selectbox(
    "📚 Select a document to study:",
    options=list(doc_options.keys()),
    key="doc_selector"
)

selected_doc_id = doc_options[selected_title]

# Load PDF if selection changed
if selected_doc_id != st.session_state.selected_document_id:
    with st.spinner(f"Loading {selected_title}..."):
        doc, title = get_pdf_from_storage(selected_doc_id)
        if doc:
            st.session_state.pdf_doc = doc
            st.session_state.selected_document_id = selected_doc_id
            st.session_state.current_page = 0
            st.success(f"✅ Loaded: {title}")
            st.rerun()

# Main interface (only show if PDF is loaded)
if st.session_state.pdf_doc is not None:
    
    # Create dynamic column layout
    if st.session_state.show_toc:
        col1, col2, col3 = st.columns([1, 2, 1.5])
    else:
        col1 = None
        col2, col3 = st.columns([2, 1.5])
    
    # Column 1: Table of Contents
    if st.session_state.show_toc and col1:
        with col1:
            display_db_toc(st.session_state.selected_document_id)
            
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
    
    # Column 2: PDF Viewer
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
                st.session_state.page_input_widget = st.session_state.current_page + 1
                st.rerun()
        with nav_col2:
            total_pages = len(st.session_state.pdf_doc)
            st.markdown(f"<center>{st.session_state.current_page + 1} / {total_pages}</center>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("Next ➡️", key="next_page", disabled=st.session_state.current_page >= len(st.session_state.pdf_doc) - 1):
                st.session_state.current_page += 1
                st.session_state.page_input_widget = st.session_state.current_page + 1
                st.rerun()
    
    # Column 3: Assessment Interface
    with col3:
        if st.session_state.selected_section:
            section = st.session_state.selected_section
            st.markdown("### 📝 Assessment")
            st.info(f"**Section:** {section['title']}  •  Pages {section['page_start']}–{section['page_end']}")

            if not st.session_state.evaluation_result:
                recall = st.text_area(
                    "Write everything you remember about this section:",
                    height=250,
                    key="recall_input",
                    placeholder="Type your free recall here..."
                )
                if st.button("Submit", type="primary"):
                    if recall.strip():
                        with st.spinner("Evaluating..."):
                            try:
                                result = run_section_recall(
                                    document_id=st.session_state.selected_document_id,
                                    node_id=section['node_id'],
                                    student_recall=recall
                                )
                                st.session_state.evaluation_result = result
                                st.rerun()
                            except anthropic.AuthenticationError:
                                st.error("Invalid Claude API key. Check CLAUDE_API_KEY in your .env.")
                            except anthropic.BadRequestError as e:
                                if "credit" in str(e).lower():
                                    st.error("Insufficient Claude API credits. Top up at console.anthropic.com.")
                                else:
                                    st.error(f"Claude API request error: {e}")
                            except anthropic.APIConnectionError:
                                st.error("Could not reach the Claude API. Check your internet connection.")
                            except anthropic.RateLimitError:
                                st.error("Claude API rate limit hit. Wait a moment and try again.")
                            except ValueError as e:
                                st.error(f"Grader failed to parse a valid response after retries: {e}")
                    else:
                        st.warning("Please write something before submitting")
            else:
                eval_data = st.session_state.evaluation_result
                st.markdown("### 📊 Results")

                score = eval_data.get('total_score', 0)
                max_score = eval_data.get('max_score', 10)
                st.metric("Score", f"{score}/{max_score} ({eval_data.get('percentage', 0):.0f}%)",
                          delta=eval_data.get('performance_level'))

                if eval_data.get('overall_feedback'):
                    st.info(f"💡 {eval_data['overall_feedback']}")

                if eval_data.get('criteria_scores'):
                    with st.expander("📋 Detailed Breakdown"):
                        for criterion in eval_data['criteria_scores']:
                            st.markdown(f"**{criterion['id']}:** {criterion['score']}/2")
                            st.caption(criterion['feedback'])

                if st.button("Try Another Section"):
                    st.session_state.selected_section = None
                    st.session_state.evaluation_result = None
                    st.rerun()
        else:
            st.markdown("### 📝 Assessment")
            st.info("Select a section from the Table of Contents to begin your recall assessment.")

else:
    st.info("👆 Please select a document to get started")