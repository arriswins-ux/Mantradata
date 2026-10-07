import os
import re
import base64
import glob
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# Mantradata - Gita Guidance  |  Team Brahmastra
# Setup:  pip install -U streamlit groq pypdf scikit-learn
# Put your Bhagavad Gita book (PDF or TXT) in the same folder as this file
# Run:    streamlit run app.py
# ---------------------------------------------------------------

# llama-3.3-70b-versatile was shut down by Groq on 16 Aug 2026.
# Models are tried in this order (first one that works is used):
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
BOOK_NAME = "Bhagavad-gita-Swami-BG-Narasingha"   # your Gita PDF name (with or without .pdf)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_IMG = os.path.join(BASE_DIR, "krishna_dashboard.jpg")
WITH_YOU_IMG = os.path.join(BASE_DIR, "krishna_with_you.jpg")
# Full-page background: put a GIF named background.gif in the folder to use it (animated).
# If it is missing, the Krishna picture is used instead. You can also set any file name here.
BACKGROUND_GIF = os.path.join(BASE_DIR, "background.gif")
BACKGROUND_IMG = BACKGROUND_GIF if os.path.exists(BACKGROUND_GIF) else os.path.join(BASE_DIR, "krishna_with_you.jpg")

st.set_page_config(page_title="Mantradata — Gita Guidance", page_icon="🕊️", layout="centered")

SYSTEM_PROMPT = """
You are Parthasarathi, a compassionate guide who speaks in the spirit of Lord Krishna
counselling Arjuna on the battlefield of Kurukshetra. The person talking to you is
feeling troubled, sad or confused. You are given BOOK EXCERPTS taken from the user's own
Bhagavad Gita book with every message.

CORE TEACHINGS: Karma Yoga (act without anxiety over results), Sthitaprajna (steady mind),
Swadharma (sincerely doing one's own duty), mind control through practice and detachment.

HOW YOU RESPOND (every turn)
1. Warmly acknowledge the person's feeling, using their name. Never hurry past their pain.
2. VERSE: choose ONE verse that actually appears in the BOOK EXCERPTS and fits their situation.
   Give its chapter and verse number exactly as printed in the excerpts, and quote or closely
   translate the text given there. NEVER invent a verse or a verse number. If no excerpt fits,
   say you are drawing on the general teaching of the Gita and give no verse number.
3. EXPLANATION: explain the verse in simple words and tie it to THEIR situation.
4. ACTION: give 2-3 small practical steps suited to their age and profession.
5. End with ONE gentle reflective question.

STYLE: calm, kind, simple language, not preachy. About 250-300 words.
You are a guide for reflection, not a doctor or therapist.

SAFETY: If the person mentions self-harm, suicide or danger, respond with care, say they
matter, urge them to contact a trusted person and a helpline right now (India: Tele-MANAS
14416 or 1-800-891-4416), and do not push philosophy at that moment.
""".strip()


# ---------------- Gita book loading + search ----------------
def find_book():
    """Look for the named Gita PDF first; otherwise fall back to any PDF/TXT with 'gita' in its name."""
    stem = os.path.splitext(BOOK_NAME)[0].lower()
    paths = []
    for folder in (BASE_DIR, os.path.join(BASE_DIR, "gita_book")):
        for ext in ("*.pdf", "*.txt"):
            paths += glob.glob(os.path.join(folder, ext))
    paths = [p for p in paths if os.path.basename(p).lower() != "requirements.txt"]
    for p in paths:  # exact name match (ignores capital letters and extension)
        if os.path.splitext(os.path.basename(p))[0].lower() == stem:
            return p
    paths.sort(key=lambda p: ("gita" not in os.path.basename(p).lower(), p))
    return paths[0] if paths else None


@st.cache_resource(show_spinner="Reading the Bhagavad Gita...")
def load_book(path, mtime):
    pages = []
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        for i, p in enumerate(reader.pages):
            pages.append((i + 1, p.extract_text() or ""))
    else:
        text = open(path, encoding="utf-8", errors="ignore").read()
        pages = [(i // 3000 + 1, text[i:i + 3000]) for i in range(0, len(text), 3000)]

    chunks = []
    for page_no, text in pages:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 40:
            continue
        start = 0
        while start < len(text):
            chunks.append({"page": page_no, "text": text[start:start + 1200]})
            start += 900
    if not chunks:
        return None
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return chunks, vectorizer, matrix


def retrieve(book, query, k=4):
    chunks, vectorizer, matrix = book
    sims = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    top = sims.argsort()[::-1][:k]
    return [chunks[i] for i in top if sims[i] > 0]


def call_groq(client, messages, max_tokens, temperature):
    """Call Groq, trying each model in MODELS until one works."""
    last_error = None
    for model in MODELS:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                reasoning_effort="low",   # gpt-oss accepts: low / medium / high
            ).choices[0].message.content
        except Exception as e:
            last_error = e
    raise last_error


def gita_keywords(client, text):
    """Turn the person's situation into Gita-style search keywords (grief, duty, fear...)."""
    try:
        return call_groq(
            client,
            [
                {"role": "system", "content": (
                    "Convert the person's situation into 10 single-word keywords useful for "
                    "searching the Bhagavad Gita (examples: grief, sorrow, duty, attachment, fear, "
                    "action, results, mind, anger, equanimity, doubt, self, surrender, failure). "
                    "Output only the keywords, comma-separated.")},
                {"role": "user", "content": text},
            ],
            max_tokens=400,
            temperature=0,
        ) or ""
    except Exception:
        return ""


KEY_FILES = [
    ".streamlit/secrets.toml", ".streamlit/secrets.toml.txt",
    "secrets.toml", "secrets.toml.txt",
    ".env", "groq_key.txt", "groq_key.txt.txt", "key.txt",
]


def load_saved_key():
    """Find the Groq key: environment variable, Streamlit secrets, or a key file next to app.py."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    for name in KEY_FILES:  # looked up next to app.py, no matter where you start streamlit from
        path = next((p for p in (os.path.join(BASE_DIR, *name.split("/")),
                                 os.path.join(os.getcwd(), *name.split("/")))
                     if os.path.isfile(p)), None)
        if path:
            try:
                text = open(path, encoding="utf-8-sig", errors="ignore").read()
            except OSError:
                continue
            match = re.search(r"gsk_[A-Za-z0-9_\-]+", text)  # Groq keys start with gsk_
            if match:
                return match.group(0)
            # fallback: take the first non-empty line, remove "GROQ_API_KEY =" and quotes
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    line = line.split("=", 1)[-1].strip().strip("\"'").strip()
                    if len(line) > 20:
                        return line
    return ""


@st.cache_resource(show_spinner=False)
def encode_file(path, mtime):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_background(path):
    """Use an image or GIF as the page background and make text/panels readable on top of it."""
    if not os.path.exists(path):
        return
    ext = os.path.splitext(path)[1].lower().strip(".")
    mime = {"png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
    encoded = encode_file(path, os.path.getmtime(path))
    css = """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.60), rgba(0,0,0,0.60)),
                          url("data:__MIME__;base64,__DATA__");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: rgba(0,0,0,0.55); }
    [data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent !important; }
    h1, h2, h3, h4,
    [data-testid="stCaptionContainer"] *,
    [data-testid="stWidgetLabel"] *,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * { color: #f5f0e6 !important; }
    [data-testid="stChatMessage"] {
        background: rgba(0,0,0,0.45);
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
    }
    [data-testid="stExpander"] { background: rgba(0,0,0,0.40); border-radius: 10px; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #f5f0e6 !important; }
    </style>
    """.replace("__MIME__", mime).replace("__DATA__", encoded)
    st.markdown(css, unsafe_allow_html=True)


def show_image(path):
    if not os.path.exists(path):
        st.warning(f"Image not found: {os.path.basename(path)}\n\nLooking in: {BASE_DIR}\n\n"
                   f"Files there: {os.listdir(BASE_DIR)}")
        return
    try:
        st.image(path, use_container_width=True)
    except TypeError:
        st.image(path, use_column_width=True)


set_background(BACKGROUND_IMG)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🕊️ About You")
    user_name = st.text_input("Your name", value="")
    user_age = st.number_input("Your age", min_value=10, max_value=100, value=19, step=1)
    user_profession = st.text_input("Profession / Studying", value="")
    saved_key = load_saved_key()
    if saved_key:
        api_key = saved_key
        st.success("🔑 Groq API key loaded")
    else:
        api_key = st.text_input("Groq API key", type="password")
        found = [n for n in KEY_FILES if os.path.isfile(os.path.join(BASE_DIR, *n.split("/")))]
        st.caption(f"No key found. app.py is in: {BASE_DIR}")
        st.caption(f"Command was started from: {os.getcwd()}")
        st.caption(f"Key files seen there: {found if found else 'none'}")
    if st.button("Start new conversation"):
        st.session_state.pop("messages", None)
        st.rerun()

# ---------------- Header ----------------
st.title("🕊️ Mantradata")
st.caption("Gita Guidance — share what weighs on your heart, as Arjuna did on the battlefield.")
show_image(DASHBOARD_IMG)

# ---------------- Load the Gita book ----------------
book_path = find_book()
book = None
if book_path:
    book = load_book(book_path, os.path.getmtime(book_path))
    if book:
        st.sidebar.success(f"📖 {os.path.basename(book_path)} ({len(book[0])} passages)")
    else:
        st.sidebar.error("Book found but no text could be read (scanned PDF?). "
                         "Use a text-based PDF or a .txt file.")
else:
    st.sidebar.warning(f"No Gita book found. Put a PDF/TXT in: {BASE_DIR}")

if not api_key:
    st.info("Enter your Groq API key in the sidebar to begin.")
    st.stop()

client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if sources:
        with st.expander("📖 Passages from the book used for this answer"):
            for s in sources:
                st.markdown(f"**Page {s['page']}**")
                st.write(s["text"])


# ---------------- Render history ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🕊️" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"))

# ---------------- Chat loop ----------------
if prompt := st.chat_input("Tell me what is troubling you..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    waiting = st.empty()
    with waiting.container():
        show_image(WITH_YOU_IMG)
        st.markdown("<h4 style='text-align:center'>🕊️ Krishna is there for you...</h4>",
                    unsafe_allow_html=True)

    # 1) find relevant passages in the book
    sources = []
    if book:
        keywords = gita_keywords(client, prompt)
        sources = retrieve(book, f"{prompt} {keywords}")
    excerpts = "\n\n".join(f"[{i + 1}] (page {s['page']}) {s['text']}" for i, s in enumerate(sources))
    if not excerpts:
        excerpts = "(No excerpts found. Do not quote a verse number.)"

    # 2) build messages: system + profile + recent history + this turn with excerpts
    profile_line = (f"User profile -> Name: {user_name.strip() or 'Friend'}, Age: {user_age}, "
                    f"Profession: {user_profession.strip() or 'not specified'}. "
                    f"Address the user by this name.")
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + profile_line}]
        + history
        + [{"role": "user", "content": f"{prompt}\n\n---\nBOOK EXCERPTS:\n{excerpts}"}]
    )

    # 3) ask Groq
    try:
        reply = call_groq(client, messages, max_tokens=2000, temperature=0.6)
    except Exception as e:
        reply = f"Sorry, something went wrong: {e}"

    waiting.empty()

    with st.chat_message("assistant", avatar="🕊️"):
        st.markdown(reply)
        render_sources(sources)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
