import os, base64, requests, fitz, urllib.parse
from pathlib import Path

import streamlit as st

# ---------------- CONFIG ----------------
SIGNED_URL = "https://hyktlmgolbfnqwaptkqy.supabase.co/storage/v1/object/sign/textbook/ai-engineering-building-applications-with-foundation-models.pdf?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV82OTc2NTQ2Mi05OWYzLTQyODMtYTRiZi03ZjI3MjhlMTEyZTMiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJ0ZXh0Ym9vay9haS1lbmdpbmVlcmluZy1idWlsZGluZy1hcHBsaWNhdGlvbnMtd2l0aC1mb3VuZGF0aW9uLW1vZGVscy5wZGYiLCJpYXQiOjE3NTk2NzU5NTUsImV4cCI6MTc5MTIxMTk1NX0.ZhlDxxdTH8amUpYIuZSJY0oNwTwLqZu7k4wW0oqv_N4"
PDF_PATH = Path("textbook.pdf")
PDFJS_VIEWER = "https://mozilla.github.io/pdf.js/web/viewer.html"

st.set_page_config(page_title="PDF + ToC + Chat (Streamlit)", layout="wide")

# ---------------- HELPERS ----------------
@st.cache_data(show_spinner=False)
def ensure_pdf() -> bytes:
    """Return PDF bytes (download if missing)."""
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 0:
        return PDF_PATH.read_bytes()
    r = requests.get(SIGNED_URL, timeout=60)
    r.raise_for_status()
    PDF_PATH.write_bytes(r.content)
    return r.content

@st.cache_data(show_spinner=False)
def extract_toc(pdf_bytes: bytes):
    """Return ToC as list of (level, title, page). Page is 1-based."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        toc = doc.get_toc(simple=True) or []
    out = []
    for item in toc:
        try:
            lvl, title, page = int(item[0]), str(item[1]), int(item[2])
            out.append((lvl, title, page))
        except Exception:
            pass
    return out

def pdf_viewer_url(pdf_bytes: bytes, page: int) -> str:
    """Build a PDF.js viewer URL that loads the PDF via a data URL and navigates to #page."""
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    data_url = f"data:application/pdf;base64,{b64}"
    encoded = urllib.parse.quote(data_url, safe="")
    # zoom=page-fit keeps a nice layout; append #page to anchor
    return f"{PDFJS_VIEWER}?file={encoded}#page={page}&zoom=page-fit"

def llm_reply(history):
    """Minimal chat: use OpenAI if key present, else echo last user message."""
    user_last = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"(echo) {user_last}"

    try:
        # Works with OpenAI Python SDK v1.x
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(local) OpenAI error: {e}"

# ---------------- STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a concise, helpful assistant in a local demo app."}
    ]
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# ---------------- UI ----------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### 📚 Table of Contents + PDF")
    pdf_bytes = ensure_pdf()
    toc = extract_toc(pdf_bytes)

    toc_box = st.container()
    with toc_box:
        if toc:
            for i, (lvl, title, page) in enumerate(toc):
                indent = "&nbsp;" * (lvl - 1) * 4
                col1, col2 = st.columns([1, 8])
                with col1:
                    clicked = st.button(f"P{page}", key=f"jump_{i}")
                with col2:
                    st.markdown(f"{indent}**{title}**", unsafe_allow_html=True)
                if clicked:
                    st.session_state.current_page = page
        else:
            st.info("No outline found in this PDF. You can still scroll the viewer below.")

    viewer_url = pdf_viewer_url(pdf_bytes, st.session_state.current_page)
    st.components.v1.html(
        f'<iframe src="{viewer_url}" width="100%" height="820" style="border:0;"></iframe>',
        height=830,
        scrolling=True,
    )

with right:
    st.markdown("### 💬 Chat")
    # Show conversation
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            continue
        is_user = msg["role"] == "user"
        with st.chat_message("user" if is_user else "assistant"):
            st.write(msg["content"])

    # Composer
    prompt = st.chat_input("Ask something…")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        reply = llm_reply(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# Small help footer
st.caption(
    "Tip: Click any ToC entry to jump pages. Set `OPENAI_API_KEY` to use a real model; "
    "otherwise the chat echoes your message."
)