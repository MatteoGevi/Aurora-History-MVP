import streamlit as st
import pymupdf as fitz
from PIL import Image
import io
import unicodedata
import re

from src.retrieval import get_document_toc as get_db_toc, save_user_session, load_last_assessment, load_document_scores
from config.constants import get_supabase, STORAGE_BUCKET, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from ingest.toc_chunk import fetch_pdf_from_storage
from ingest.ingest import ingest_document, DuplicateDocumentError


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

def get_pdf_from_storage(document_id: str, sb=None) -> tuple:
    """
    Download PDF from Supabase Storage using the existing fetch function

    Returns:
        (pdf_doc, filename) or (None, None) if error
    """
    try:
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

        pdf_bytes = fetch_pdf_from_storage(
            supabase_url=SUPABASE_URL,
            auth_token=SUPABASE_SERVICE_ROLE_KEY,
            bucket=STORAGE_BUCKET,
            filename=filename,
        )

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        return doc, doc_data.get('title', 'Document')

    except Exception as e:
        st.error(f"Error loading PDF from storage: {e}")
        filename_display = locals().get('filename', '<unknown>')
        st.info(f"Tried to load: {filename_display} from bucket '{STORAGE_BUCKET}'")

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

    toc = get_db_toc(document_id, sb=get_supabase())

    if not toc:
        st.info("No table of contents found")
        return

    # Load scores for all nodes in one query
    scores: dict = {}
    user = st.session_state.get("user")
    if user:
        try:
            scores = load_document_scores(
                user_id=user.id,
                document_id=document_id,
                sb=st.session_state.supabase_client,
            )
        except Exception:
            pass

    def render_node(nodes, level=0):
        for node in nodes:
            node_id  = node["node_id"]
            has_kids = bool(node.get("children"))
            exp_key  = f"toc_expanded_{node_id}"
            expanded = st.session_state.get(exp_key, False)

            pad   = "　" * level
            arrow = ("▼ " if expanded else "▶ ") if has_kids else "    "

            node_score = scores.get(node_id)
            if node_score and node_score.get("percentage") is not None:
                pct = node_score["percentage"]
                if pct >= 90:
                    bullet = "🟣"
                elif pct >= 71:
                    bullet = "🟢"
                elif pct >= 51:
                    bullet = "🟡"
                elif pct >= 31:
                    bullet = "🟠"
                else:
                    bullet = "🔴"
                score_tag = f" {bullet} {pct:.0f}%"
            else:
                score_tag = ""

            label = f"{pad}{arrow}{node['title']}{score_tag}"

            if st.button(label, key=f"toc_{node_id}", use_container_width=True):
                if has_kids:
                    st.session_state[exp_key] = not expanded
                # always navigate to the node's page
                st.session_state.current_page      = node["page_start"] - 1
                st.session_state.page_input_widget = node["page_start"]
                st.session_state.selected_section  = node
                st.session_state.evaluation_result = None
                st.session_state.recalled_text     = None
                # Persist position to Supabase
                try:
                    user = st.session_state.get("user")
                    if user:
                        save_user_session(
                            user_id=user.id,
                            document_id=st.session_state.selected_document_id,
                            node_id=node_id,
                            page_num=node["page_start"] - 1,
                            sb=st.session_state.supabase_client,
                        )
                        saved = load_last_assessment(
                            user_id=user.id,
                            document_id=st.session_state.selected_document_id,
                            node_id=node_id,
                            sb=st.session_state.supabase_client,
                        )
                        if saved:
                            st.session_state.recalled_text     = saved.pop("_recalled_text", None)
                            st.session_state.evaluation_result = saved
                except Exception:
                    pass  # non-critical
                st.rerun(scope="app")

            if has_kids and expanded:
                render_node(node["children"], level + 1)

    render_node(toc)


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
            raw_name = uploaded_file.name
            # Normalize accented chars to ASCII equivalents, then strip anything remaining non-safe
            normalized = unicodedata.normalize("NFKD", raw_name).encode("ascii", "ignore").decode("ascii")
            filename = re.sub(r"[^\w\-. ]", "-", normalized).strip()
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
