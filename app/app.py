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
from config.constants import get_supabase, get_supabase_for_user, STORAGE_BUCKET, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_ANON_KEY
from ingest.toc_chunk import fetch_pdf_from_storage
from ingest.ingest import ingest_document, DuplicateDocumentError

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
if 'recalled_text' not in st.session_state:
    st.session_state.recalled_text = None
if 'page_input_widget' not in st.session_state:
    st.session_state.page_input_widget = 1

# Auth session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'supabase_client' not in st.session_state:
    st.session_state.supabase_client = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'user_jwt' not in st.session_state:
    st.session_state.user_jwt = None

# ── Login gate ────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    st.title("🦉 Aurora Learning Platform")
    st.markdown("Sign in to continue")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary", key="login_btn"):
            if login_email and login_password:
                try:
                    from supabase import create_client
                    auth_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                    response = auth_client.auth.sign_in_with_password(
                        {"email": login_email, "password": login_password}
                    )
                    st.session_state.authenticated = True
                    st.session_state.user_jwt = response.session.access_token
                    st.session_state.supabase_client = get_supabase_for_user(
                        response.session.access_token
                    )
                    st.session_state.user = response.user
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
            else:
                st.warning("Please enter your email and password")

    with tab_signup:
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up", type="primary", key="signup_btn"):
            if signup_email and signup_password:
                try:
                    from supabase import create_client
                    auth_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                    auth_client.auth.sign_up({"email": signup_email, "password": signup_password})
                    st.success("Account created! Check your email to confirm your registration, then log in.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")
            else:
                st.warning("Please enter email and password")

    st.stop()

def get_pdf_from_storage(document_id: str, sb=None) -> tuple:
    """
    Download PDF from Supabase Storage using the existing fetch function

    Returns:
        (pdf_doc, filename) or (None, None) if error
    """
    try:
        # Get document metadata from database
        sb = sb or get_supabase()
        doc_result = sb.table("documents").select("*").eq("id", document_id).single().execute()
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
            auth_token=SUPABASE_SERVICE_ROLE_KEY,
            bucket=STORAGE_BUCKET,
            filename=filename,
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
            files = sb.storage.from_(STORAGE_BUCKET).list()
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

    toc = get_db_toc(document_id, sb=st.session_state.supabase_client)

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
col_title, col_toggle, col_logout = st.columns([4, 1, 1])
with col_title:
    st.title("🦉 AI Engineering Learning Platform")
    st.markdown("*MVP - Building Applications with Foundation Models*")
with col_toggle:
    if st.button("📚 Toggle ToC" if not st.session_state.show_toc else "✖️ Close ToC"):
        st.session_state.show_toc = not st.session_state.show_toc
        st.rerun()
with col_logout:
    user_email = getattr(st.session_state.user, 'email', '') if st.session_state.user else ''
    if st.button(f"Logout", help=user_email):
        st.session_state.clear()
        st.rerun()

def show_upload_widget():
    """Upload a PDF to Supabase Storage and run the ingestion pipeline."""
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        key="pdf_uploader",
        help="The PDF will be uploaded to storage and indexed for assessment.",
    )

    if uploaded_file is not None:
        col_btn, col_force = st.columns([3, 2])
        with col_btn:
            ingest_clicked = st.button("Ingest Document", type="primary", key="ingest_btn")
        with col_force:
            force_reingest = st.checkbox("Replace if duplicate", key="force_reingest_cb")

        if ingest_clicked:
            pdf_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
            user_id = st.session_state.user.id if st.session_state.user else None

            # Upload to Supabase Storage (service role bypasses storage RLS)
            with st.spinner(f"Uploading {filename} to storage..."):
                try:
                    sb_admin = get_supabase()
                    sb_admin.storage.from_(STORAGE_BUCKET).upload(
                        filename,
                        pdf_bytes,
                        {"content-type": "application/pdf", "upsert": "true"},
                    )
                except Exception as e:
                    st.error(f"Storage upload failed: {e}")
                    return

            # Run ingestion pipeline
            with st.spinner("Indexing document — this may take a few minutes for large PDFs..."):
                try:
                    doc_id = ingest_document(
                        filename=filename,
                        pdf_bytes=pdf_bytes,
                        user_id=user_id,
                        user_jwt=st.session_state.user_jwt,
                        supabase_client=st.session_state.supabase_client,
                        force_reingest=force_reingest,
                    )
                    st.success(f"✅ **{filename}** ingested successfully! (ID: `{doc_id}`)")
                    st.rerun()
                except DuplicateDocumentError as e:
                    st.warning(
                        f"⚠️ **{e.title}** is already in the database.\n\n"
                        "Check **Replace if duplicate** and click Ingest again to re-index it."
                    )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")


# Document selector
st.markdown("---")
try:
    docs = get_document_list(sb=st.session_state.supabase_client)
except Exception as e:
    st.error(f"Could not reach the database: {e}")
    if st.button("Retry"):
        st.rerun()
    st.stop()

with st.expander("📤 Upload New Document", expanded=not docs):
    show_upload_widget()

if not docs:
    st.info("No documents yet — upload a PDF above to get started.")
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
        doc, title = get_pdf_from_storage(selected_doc_id, sb=st.session_state.supabase_client)
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
    
    # Column 1: Page Navigation + Table of Contents
    if st.session_state.show_toc and col1:
        with col1:
            # Page navigation at the top
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
                key="page_input_widget",
                on_change=update_page
            )

            st.text(f"Page {st.session_state.current_page + 1} of {total_pages}")

            st.markdown("---")
            display_db_toc(st.session_state.selected_document_id)
    
    # Column 2: PDF Viewer
    with col2:
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
                                    student_recall=recall,
                                    sb=st.session_state.supabase_client,
                                )
                                st.session_state.evaluation_result = result
                                st.session_state.recalled_text = recall
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

                if st.session_state.recalled_text:
                    with st.expander("📝 Your answer"):
                        st.write(st.session_state.recalled_text)

                col_retry, col_next = st.columns(2)
                with col_retry:
                    if st.button("Try Again", use_container_width=True):
                        st.session_state.evaluation_result = None
                        st.session_state.recalled_text = None
                        st.rerun()
                with col_next:
                    if st.button("Next Section", use_container_width=True):
                        st.session_state.selected_section = None
                        st.session_state.evaluation_result = None
                        st.session_state.recalled_text = None
                    st.rerun()
        else:
            st.markdown("### 📝 Assessment")
            st.info("Select a section from the Table of Contents to begin your recall assessment.")

else:
    st.info("👆 Please select a document to get started")