import uuid

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/chat"
UPLOAD_API_URL = "http://127.0.0.1:8000/upload-resume"
REQUEST_TIMEOUT_SECONDS = 180

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "resume_upload_state" not in st.session_state:
    st.session_state.resume_upload_state = "none"

if "active_resume_filename" not in st.session_state:
    st.session_state.active_resume_filename = None

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ink: #172b4d;
        --muted: #66758b;
        --line: #e3e9f1;
        --surface: #ffffff;
        --canvas: #f6f8fb;
        --accent: #147d75;
        --accent-dark: #10665f;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3 {
        color: var(--ink) !important;
        letter-spacing: -0.025em;
    }

    p, label, .stMarkdown {
        color: #34445b;
    }

    .hero {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 30px 32px;
        margin-bottom: 22px;
    }

    .eyebrow {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 750;
        margin-bottom: 10px;
    }

    .hero-title {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 750;
        line-height: 1.12;
        letter-spacing: -0.045em;
        margin-bottom: 12px;
    }

    .hero-description {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        max-width: 680px;
        margin-bottom: 0;
    }

    .section-label {
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .section-description {
        color: var(--muted);
        font-size: 0.91rem;
        line-height: 1.6;
        margin-bottom: 15px;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .upload-symbol {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: #e9f5f3;
        color: var(--accent);
        font-size: 1.35rem;
        margin-bottom: 12px;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        background: #e9f5f3;
        color: #10665f;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .answer-panel {
        background: #ffffff;
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        padding: 22px 24px;
        margin-top: 8px;
        margin-bottom: 18px;
        color: var(--ink);
        line-height: 1.8;
        overflow-wrap: anywhere;
    }

    .answer-label {
        color: var(--accent);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 750;
        margin-bottom: 10px;
    }

    .footer-note {
        color: #8290a3;
        font-size: 0.8rem;
        text-align: center;
        padding-top: 25px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 43px;
        font-weight: 600;
        border: 1px solid #d9e2ed;
        transition: border-color 0.15s ease;
    }

    div.stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent-dark);
    }

    div.stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
    }

    div.stButton > button[kind="primary"]:hover {
        background: var(--accent-dark);
        border-color: var(--accent-dark);
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: #fbfcfe;
        border: 1px dashed #c8d4e3;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    textarea {
        border-radius: 12px !important;
    }

    hr {
        border-color: var(--line);
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1rem;
        }

        .hero {
            padding: 23px 20px;
        }

        .panel {
            padding: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND REQUESTS
# ============================================================

def ask_backend(question):
    try:
        response = requests.post(
            API_URL,
            json={
                "question": question,
                "session_id": st.session_state.session_id,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        return None, (
            "Could not connect to the AI backend. "
            "Check that FastAPI is running at http://127.0.0.1:8000."
        )

    except requests.exceptions.Timeout:
        return None, (
            "The AI backend took too long to respond. "
            "The model may still be loading. Please try again."
        )

    except requests.exceptions.HTTPError as error:
        try:
            detail = error.response.json().get("detail", "")
        except (ValueError, AttributeError):
            detail = ""

        message = (
            f"The AI backend returned an error "
            f"(HTTP {error.response.status_code})."
        )
        if detail:
            message += f" Details: {detail}"
        return None, message

    except requests.exceptions.RequestException:
        return None, "Something went wrong while contacting the AI backend."

    try:
        data = response.json()
    except ValueError:
        return None, "The AI backend did not return valid JSON."

    if not isinstance(data, dict) or "answer" not in data:
        return None, "The AI backend response did not include an answer."

    return data, None


def upload_resume_to_backend(uploaded_file):
    try:
        response = requests.post(
            UPLOAD_API_URL,
            data={"session_id": st.session_state.session_id},
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/pdf",
                )
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

    except requests.exceptions.HTTPError as error:
        try:
            detail = error.response.json().get("detail", "")
        except (ValueError, AttributeError):
            detail = ""

        message = f"Upload failed (HTTP {error.response.status_code})."
        if detail:
            message += f" {detail}"
        return None, message, "rejected"

    except requests.exceptions.Timeout:
        return (
            None,
            "The upload timed out, so its result could not be confirmed.",
            "unknown",
        )

    except requests.exceptions.ConnectionError:
        return (
            None,
            "Could not connect to the AI backend. The upload result is unconfirmed.",
            "unknown",
        )

    except requests.exceptions.RequestException:
        return (
            None,
            "The upload could not be completed. Its result is unconfirmed.",
            "unknown",
        )

    try:
        data = response.json()
    except ValueError:
        return None, "The upload response was not valid JSON.", "unknown"

    if (
        response.status_code != 200
        or not isinstance(data, dict)
        or not isinstance(data.get("filename"), str)
        or not data["filename"].strip()
        or not isinstance(data.get("pages"), int)
    ):
        return None, "The upload response was missing valid resume details.", "unknown"

    return data, None, "success"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered resume review</div>
        <div class="hero-title">Understand a resume.<br>Ask better questions.</div>
        <p class="hero-description">
            Upload a PDF resume and ask questions about its contents.
            The analyzer helps you explore listed skills, education,
            and work experience using information from the document.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESUME UPLOAD
# ============================================================

st.markdown('<div class="section-label">01 &nbsp; Upload a resume</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">'
    'Choose a PDF to begin. You can ask questions after the upload is confirmed.'
    '</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="upload-symbol">↥</div>', unsafe_allow_html=True)
    st.markdown("**Your resume document**")
    st.caption("PDF format · Use a resume you have permission to analyze")

    with st.form("resume_upload_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "Select a PDF resume",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        upload_clicked = st.form_submit_button(
            "Upload and analyze",
            type="primary",
            use_container_width=True,
        )

    if upload_clicked:
        if uploaded_file is None:
            st.error("Please select a PDF file first.")
        else:
            with st.spinner("Uploading and validating your document..."):
                upload_data, upload_error, outcome = upload_resume_to_backend(
                    uploaded_file
                )

            if outcome == "success":
                st.session_state.resume_upload_state = "active"
                st.session_state.active_resume_filename = upload_data["filename"]
                st.session_state.latest_result = None
                st.session_state.question_input = ""
                st.success(
                    f"Resume ready: {upload_data['filename']} "
                    f"({upload_data['pages']} page(s))."
                )

            elif outcome == "rejected":
                st.error(upload_error)
                if st.session_state.resume_upload_state == "active":
                    st.info(
                        "Your previously active resume remains selected: "
                        f"{st.session_state.active_resume_filename}."
                    )
                else:
                    st.info("Upload a valid resume before asking questions.")

            else:
                st.session_state.resume_upload_state = "unknown"
                st.session_state.active_resume_filename = None
                st.session_state.latest_result = None
                st.error(upload_error)
                st.warning(
                    "The upload could not be confirmed. Upload the PDF again "
                    "before asking questions."
                )

    if st.session_state.resume_upload_state == "active":
        st.markdown(
            f'<span class="status-pill">✓ Resume ready</span> '
            f'`{st.session_state.active_resume_filename}`',
            unsafe_allow_html=True,
        )
    elif st.session_state.resume_upload_state == "unknown":
        st.warning("The active resume could not be confirmed. Please upload it again.")
    else:
        st.caption("No resume uploaded yet.")


# ============================================================
# QUESTION AREA
# ============================================================

st.markdown("---")
st.markdown('<div class="section-label">02 &nbsp; Ask about the resume</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">'
    'Choose a suggested question or write your own. Answers depend on the content '
    'of the uploaded resume.'
    '</div>',
    unsafe_allow_html=True,
)

can_chat = st.session_state.resume_upload_state == "active"

suggestions = [
    "What skills are listed in the resume?",
    "Summarize the educational background.",
    "What work experience is mentioned?",
]

col_a, col_b, col_c = st.columns(3)

for column, suggestion in zip((col_a, col_b, col_c), suggestions):
    with column:
        if st.button(
            suggestion,
            key=f"suggest_{suggestion}",
            use_container_width=True,
            disabled=not can_chat,
        ):
            st.session_state.question_input = suggestion

question = st.text_area(
    "Your question",
    key="question_input",
    height=100,
    placeholder="For example: What programming languages are mentioned?",
    disabled=not can_chat,
    help="Ask about information contained in the uploaded resume.",
)

ask_clicked = st.button(
    "Ask AI assistant",
    type="primary",
    use_container_width=True,
    disabled=not can_chat,
)

if not can_chat:
    st.info("Upload a resume above to unlock the question assistant.")

if ask_clicked:
    clean_question = question.strip()

    if not clean_question:
        st.warning("Enter a question before continuing.")
    else:
        with st.spinner("Reviewing the resume and preparing an answer..."):
            data, error_message = ask_backend(clean_question)

        if error_message:
            st.error(error_message)
        else:
            st.session_state.latest_result = {
                "question": clean_question,
                "data": data,
            }


# ============================================================
# ANSWER
# ============================================================

result = st.session_state.latest_result

if result:
    data = result["data"]
    answer = data.get("answer", "")
    grounded = data.get("grounded", False)

    st.markdown("---")
    st.markdown('<div class="section-label">03 &nbsp; Answer</div>', unsafe_allow_html=True)
    st.caption(f"Question: {result['question']}")

    st.markdown('<div class="answer-label">AI response</div>', unsafe_allow_html=True)

    if answer and answer.strip():
        st.markdown(
            f'<div class="answer-panel">{__import__("html").escape(answer).replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("The AI did not return an answer.")

    if grounded:
        st.success("The response was marked as grounded by the backend.")
    else:
        st.warning(
            "The backend did not mark this response as grounded. "
            "Verify it against the uploaded resume."
        )

    with st.expander("Technical details"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Iterations", data.get("iterations", 0))

        with col2:
            st.metric("Tools executed", data.get("tools_executed", 0))

        with col3:
            st.metric("Grounded", "Yes" if grounded else "No")

        st.write(f"**Status:** {data.get('status', 'unknown')}")
        st.write(
            "**Duplicate tool calls skipped:** "
            f"{data.get('duplicates_skipped', 0)}"
        )
        st.write(
            "**Tool calls refused:** "
            f"{data.get('tools_refused', 0)}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        AI Resume Analyzer · A local AI prototype for exploring resume content
        <br>
        Review AI-generated answers against the source document.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# ACCESSIBLE COLOR OVERRIDES
# ============================================================

st.markdown(
    """
    <style>
    /* Overall page */
    .stApp {
        background-color: #F7F8FA;
        color: #273444;
    }

    /* Main text and headings */
    .stApp,
    .stApp p,
    .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"] {
        color: #344256;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3 {
        color: #243447;
    }

    /* Question box and other text inputs */
    .stTextArea textarea,
    .stTextInput input,
    [data-testid="stTextArea"] textarea {
        background-color: #FFFFFF !important;
        color: #263648 !important;
        border: 1px solid #C7D2DE !important;
        border-radius: 8px !important;
        caret-color: #355D7A;
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #718096 !important;
        opacity: 1;
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #7198B8 !important;
        box-shadow: 0 0 0 2px #DCE8F2 !important;
    }

    /* Main action buttons */
    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #496A85 !important;
        color: #FFFFFF !important;
        border: 1px solid #496A85 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: background-color 0.15s ease;
    }

    div.stButton > button p,
    div.stFormSubmitButton > button p {
        color: #FFFFFF !important;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #3D5C75 !important;
        border-color: #3D5C75 !important;
        color: #FFFFFF !important;
    }

    div.stButton > button:focus-visible,
    div.stFormSubmitButton > button:focus-visible {
        outline: 3px solid #9CBBD4 !important;
        outline-offset: 2px;
    }

    /* Resume upload area */
    [data-testid="stFileUploader"] {
        background-color: #F0F3F7;
        border-radius: 10px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #F0F4F8 !important;
        border: 1px dashed #AEBECD !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #34485D !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background-color: #FFFFFF !important;
        color: #355D7A !important;
        border: 1px solid #AEBECD !important;
        border-radius: 6px !important;
    }

    [data-testid="stFileUploaderDropzone"] button p {
        color: #355D7A !important;
    }

    /* File names and secondary information */
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: #526274;
    }

    /* Result panel */
    .answer-panel {
        background-color: #FFFFFF !important;
        color: #29394B !important;
        border: 1px solid #D7E0E8 !important;
        border-radius: 10px;
        padding: 20px;
        line-height: 1.7;
    }

    /* Expanders and technical details */
    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #D7E0E8;
        border-radius: 8px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #34485D !important;
    }

    /* Keep disabled controls readable */
    button:disabled {
        opacity: 0.65 !important;
        cursor: not-allowed !important;
    }

    /* Footer */
    .footer-note {
        color: #64748B !important;
        font-size: 0.85rem;
        text-align: center;
        line-height: 1.7;
        padding: 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
