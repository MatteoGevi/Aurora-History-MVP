# app.py - Load PDFs from Supabase Storage using existing fetch function
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import anthropic
import streamlit as st

from components import render_pdf_page, get_pdf_from_storage, display_db_toc, show_upload_widget
from src.retrieval import get_document_list, load_user_session, save_assessment, load_last_assessment
from src.pipeline import run_section_recall
from config.constants import get_supabase_for_user, SUPABASE_URL, SUPABASE_ANON_KEY



# Page configuration
st.set_page_config(
    page_title="Aurora - Empowered Learning Environment",
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
if 'page_cache' not in st.session_state:
    st.session_state.page_cache = {}

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
# ── Login gate ────────────────────────────────────────────────────────────────
# Replace lines 66–132 in app.py with this block

if not st.session_state.authenticated:
    st.title("🦉 Aurora Learning Platform")
    st.markdown("*AI-powered recall assessment grounded in your own study materials.*")
    st.markdown("---")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    # ── LOGIN TAB ──────────────────────────────────────────────────────────────
    with tab_login:
        login_email    = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary", key="login_btn", use_container_width=True):
            if login_email and login_password:
                try:
                    from supabase import create_client
                    auth_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                    response = auth_client.auth.sign_in_with_password(
                        {"email": login_email, "password": login_password}
                    )
                    st.session_state.authenticated = True
                    st.session_state.user_jwt      = response.session.access_token
                    st.session_state.supabase_client = get_supabase_for_user(
                        response.session.access_token
                    )
                    st.session_state.user = response.user

                    # Restore last session position
                    try:
                        prev = load_user_session(
                            response.user.id,
                            sb=st.session_state.supabase_client,
                        )
                        if prev:
                            st.session_state.selected_document_id = prev["document_id"]
                            st.session_state.current_page         = prev["page_num"]
                            st.session_state.page_input_widget    = prev["page_num"] + 1
                            if prev.get("node_id"):
                                st.session_state.selected_section = {"node_id": prev["node_id"]}
                                saved = load_last_assessment(
                                    user_id=response.user.id,
                                    document_id=prev["document_id"],
                                    node_id=prev["node_id"],
                                    sb=st.session_state.supabase_client,
                                )
                                if saved:
                                    st.session_state.recalled_text     = saved.pop("_recalled_text", None)
                                    st.session_state.evaluation_result = saved
                    except Exception:
                        pass  # non-critical — proceed without restoring

                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
            else:
                st.warning("Please enter your email and password.")

    # ── SIGN UP TAB ────────────────────────────────────────────────────────────
    with tab_signup:
        signup_email    = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")

        # ── GDPR Consent block ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Before you continue")

        st.info(
            "**Aurora is an AI-powered learning assessment tool in closed beta.**\n\n"
            "By creating an account, Aurora will:\n"
            "- Store your **email address** to authenticate you\n"
            "- Store **PDF documents** you upload as study materials\n"
            "- Process your **written responses** using the OpenAI API (GPT-4o mini) "
            "to generate assessments — this means your text is sent to OpenAI's servers "
            "for inference. OpenAI does not use API data for model training.\n"
            "- Store your **evaluation results** linked to your account\n\n"
            "All data is hosted on EU infrastructure (Supabase, Frankfurt). "
            "You can request deletion of your data at any time by contacting "
            "**privacy@aurora-app.io**.\n\n"
            "This is a **beta product** — it may contain bugs and is not guaranteed to be "
            "available at all times. All beta data will be deleted within 30 days of the "
            "beta programme ending."
        )

        consent = st.checkbox(
            "I have read and understood how Aurora uses my data, "
            "and I agree to the [Terms & Conditions](https://ai-aurora.com/legal/terms) "
            "and [Privacy Policy](https://ai-aurora.com/legal/privacy).",
            key="consent_checkbox"
        )

        beta_feedback = st.checkbox(
            "I'm happy to be contacted for feedback during the beta programme. *(optional)*",
            key="beta_feedback_checkbox"
        )

        if st.button(
            "Create Account",
            type="primary",
            key="signup_btn",
            use_container_width=True,
            disabled=not consent,   # button is greyed out until consent is given
        ):
            if signup_email and signup_password:
                try:
                    from supabase import create_client
                    import datetime

                    auth_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                    response = auth_client.auth.sign_up({
                        "email": signup_email,
                        "password": signup_password,
                        "options": {
                            "data": {
                                # Stored in auth.users raw_user_meta_data
                                "consent_given_at":      datetime.datetime.utcnow().isoformat(),
                                "consent_policy_version": "beta_v1",
                                "beta_feedback_opt_in":   beta_feedback,
                            }
                        }
                    })

                    # Also write consent timestamp to profiles table via service client
                    # The trigger creates the profile row on signup; we update it here
                    try:
                        from config.constants import get_supabase
                        admin_client = get_supabase()
                        if response.user:
                            admin_client.table("profiles").update({
                                "consent_given_at":      datetime.datetime.utcnow().isoformat(),
                                "consent_policy_version": "beta_v1",
                            }).eq("id", response.user.id).execute()
                    except Exception:
                        pass  # non-critical — consent is already in auth metadata

                    st.success(
                        "✅ Account created! Check your email to confirm your registration, "
                        "then log in."
                    )
                except Exception as e:
                    st.error(f"Sign up failed: {e}")
            else:
                st.warning("Please enter your email and password.")

        if not consent:
            st.caption(
                "You must accept the Terms & Conditions and Privacy Policy to create an account."
            )

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Aurora · Closed Beta · "
        "[Privacy Policy](https://ai-aurora.com/legal/privacy) · "
        "[Terms & Conditions](https://ai-aurora.com/legal/terms) · "
        "Questions? privacy@aurora-app.io"
    )

    st.stop()

# Main app header
col_title, col_toggle, col_logout = st.columns([4, 1, 1])
with col_title:
    st.title("🦉 Aurora - Empowered Learning Environment")
    st.markdown("*Upload your study materials. Write down what you've understood. Aurora will evaluate how much you've actually learned.*")
with col_toggle:
    if st.button("📚 Toggle ToC" if not st.session_state.show_toc else "✖️ Close ToC"):
        st.session_state.show_toc = not st.session_state.show_toc
        st.rerun()
with col_logout:
    user_email = getattr(st.session_state.user, 'email', '') if st.session_state.user else ''
    if st.button(f"Logout", help=user_email):
        st.session_state.clear()
        st.rerun()

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
            st.session_state.page_cache = {}
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
        _cache_key = (st.session_state.selected_document_id, st.session_state.current_page)
        if _cache_key not in st.session_state.page_cache:
            st.session_state.page_cache[_cache_key] = render_pdf_page(
                st.session_state.pdf_doc, st.session_state.current_page
            )
        img = st.session_state.page_cache[_cache_key]
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
                                try:
                                    user = st.session_state.get("user")
                                    if user:
                                        save_assessment(
                                            user_id=user.id,
                                            document_id=st.session_state.selected_document_id,
                                            node_id=section['node_id'],
                                            recalled_text=recall,
                                            result=result,
                                            sb=st.session_state.supabase_client,
                                        )
                                except Exception:
                                    pass  # non-critical
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