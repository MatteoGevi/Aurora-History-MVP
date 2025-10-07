import os
from pathlib import Path
import requests
import pymupdf as fitz  # PyMuPDF
import panel as pn

# ---------------- config ----------------
os.environ["DATABASE_URL"] = "postgresql://postgres:YOUR_PASSWORD@db.YOUR_REF.supabase.co:5432/postgres"

SIGNED_URL = "https://hyktlmgolbfnqwaptkqy.supabase.co/storage/v1/object/sign/textbook/ai-engineering-building-applications-with-foundation-models.pdf?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV82OTc2NTQ2Mi05OWYzLTQyODMtYTRiZi03ZjI3MjhlMTEyZTMiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJ0ZXh0Ym9vay9haS1lbmdpbmVlcmluZy1idWlsZGluZy1hcHBsaWNhdGlvbnMtd2l0aW9uLW1vZGVscy5wZGYiLCJpYXQiOjE3NTk2NzU5NTUsImV4cCI6MTc5MTIxMTk1NX0.ZhlDxxdTH8amUpYIuZSJY0oNwTwLqZu7k4wW0oqv_N4"
STATIC = Path("static"); STATIC.mkdir(exist_ok=True)
PDF_PATH = STATIC / "textbook.pdf"

# ---------------- helpers ----------------
def ensure_pdf():
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 0:
        return
    r = requests.get(SIGNED_URL, timeout=60); r.raise_for_status()
    PDF_PATH.write_bytes(r.content)

def get_toc(path: Path):
    with fitz.open(path) as doc:
        toc = doc.get_toc(simple=True) or []
    # list of (level, title, page)
    return [(int(l), str(t), int(p)) for l, t, p in toc]

ensure_pdf()
toc = get_toc(PDF_PATH)

# --------------- UI pieces ---------------
pn.extension(notifications=True)

# Left: PDF viewer (native browser viewer inside iframe)
# Panel’s PDF pane simply embeds the file URL — anchors like #page= work.
pdf_pane = pn.pane.PDF(PDF_PATH, height=900, sizing_mode="stretch_both")

# ToC list
def make_toc_item(level, title, page):
    pad = (level - 1) * 12
    btn = pn.widgets.Button(name=f"{title}", button_type="default", sizing_mode="stretch_width")
    btn.styles = dict(padding_left=f"{pad}px")
    def _go(_):
        # update the pane's URL with #page=
        pdf_pane.object = f"{PDF_PATH}#page={page}&zoom=page-fit"
    btn.on_click(_go)
    return btn, pn.pane.Markdown(f"<span style='color:#8aa'>p.{page}</span>", sizing_mode="fixed", width=60)

toc_rows = []
for l, t, p in toc:
    b, pmd = make_toc_item(l, t, p)
    toc_rows.append(pn.Row(b, pmd))

toc_col = pn.Column(
    "### Table of Contents",
    *toc_rows if toc_rows else [pn.pane.Markdown("_No outline found._")],
    sizing_mode="stretch_both",
    scroll=True,
)

# Right: tiny chat (echo for now)
chat_area = pn.Column(sizing_mode="stretch_both", scroll=True)
input_box = pn.widgets.TextInput(placeholder="Ask something…", sizing_mode="stretch_width")
send_btn  = pn.widgets.Button(name="Send", button_type="primary", width=90)

def send(_):
    text = input_box.value.strip()
    if not text: return
    chat_area.append(pn.pane.Markdown(f"**You:** {text}"))
    # plug your assessor logic here:
    reply = f"(echo) {text}"
    chat_area.append(pn.pane.Markdown(f"**Bot:** {reply}"))
    input_box.value = ""

send_btn.on_click(send)

composer = pn.Row(input_box, send_btn)

# --------------- layout ------------------
left = pn.Column(toc_col, pdf_pane, sizing_mode="stretch_both")
right = pn.Column("### Chat", chat_area, composer, sizing_mode="stretch_both")

app = pn.Row(left, right, sizing_mode="stretch_both")
app.servable(title="PDF + ToC + Chat")