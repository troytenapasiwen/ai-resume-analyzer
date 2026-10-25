import html
import io
import re
from pathlib import Path
import time
import uuid

import requests
import streamlit as st


# ============================================================
# CONFIGURATION (unchanged)
# ============================================================

API_URL = "http://127.0.0.1:8000/chat"
UPLOAD_API_URL = "http://127.0.0.1:8000/upload-resume"
REQUEST_TIMEOUT_SECONDS = 180
PORTFOLIO_PREVIEW = True

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon=str(Path(__file__).parent / "assets" / "analyze-favicon.png"),
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

# UI-only keys (never sent to the backend).
_UI_DEFAULTS = {
    "resume_profile": None,    # facts read locally from the PDF
    "earlier_results": [],     # previous Q&A turns, newest first
    "pending_question": None,  # set by Ask, cleared when the request ends
    "last_error": None,        # {"question", "message"}
    "composer_notice": None,   # e.g. empty-question warning
    "restore_question": None,  # put a failed question back in the box
    "upload_feedback": None,   # one-shot upload messages
    "uploader_nonce": 0,       # resets the file picker after each upload attempt
}
for _key, _value in _UI_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

# A failed question is restored to the input before the widget is created.
if st.session_state.restore_question is not None:
    st.session_state.question_input = st.session_state.restore_question
    st.session_state.restore_question = None


# ============================================================
# STYLING
# ============================================================
# Design system: contemporary brutalist editorial, held in check.
#   Palette  : ink #17191C and paper #F5F4F0 carry the page; white is the
#              "working" surface; concrete #D9D9D5 is for hairlines;
#              cobalt #2457D6 is ONLY for interactive state (hover, focus,
#              selection, links). ok / warn / err appear only as status.
#   Type     : compact heavy sans for display, plain sans for reading,
#              mono for metadata (file info, numbering, processing state).
#              System fonts only, no external requests.
#   Geometry : 0 radius, 1px hairlines, 2px structural rules, 4px accent
#              rules. 4 / 8 / 12 / 16 / 24 / 32 px spacing. 44px touch targets.
# Each logical group is ONE markdown block (class .blk) so its spacing is
# defined here, not by Streamlit's gaps. Regions use the stable `st-key-*`
# classes, and component rules are scoped to them.

st.markdown(
    """
    <style>
    :root {
        --ink: #17191C;
        --paper: #F5F4F0;
        --white: #FFFFFF;
        --cobalt: #2457D6;
        --concrete: #D9D9D5;
        --muted: #555A60;
        --ok: #1B6B3A;
        --warn: #8A5300;
        --err: #B3261E;
        --b1: 1px;
        --b2: 2px;
        --b4: 4px;
        --display: "Helvetica Neue", Helvetica, Arial, "Segoe UI", system-ui, sans-serif;
        --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas,
                "Liberation Mono", monospace;
    }

    .stApp { background: var(--paper); color: var(--ink); color-scheme: light; font-family: var(--sans); }
    /* Streamlit ships its own narrow Source Sans; use the system UI face instead. */
    .stApp :where(p, li, label, summary, button, textarea, input, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"]) { font-family: var(--sans) !important; }
    [data-testid="stMetricLabel"] * { font-family: var(--mono) !important; }
    .block-container { max-width: 1240px; padding-top: 2.75rem; padding-bottom: 1rem; }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer, [data-testid="stAppDeployButton"] { visibility: hidden; }
    [data-testid="stVerticalBlock"] { gap: 0.75rem; }
    /* Streamlit gives every markdown block margin-bottom:-16px to offset paragraph
       margins; our own HTML blocks (root class .blk) have no such paragraph. */
    [data-testid="stMarkdownContainer"]:has(> .blk) { margin-bottom: 0 !important; }
    .stApp p, .stApp label, .stApp li,
    .stApp [data-testid="stMarkdownContainer"] { color: var(--ink); }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted); font-size: 0.84rem; }

    /* Raw-HTML headings: neutralise Streamlit's default sizes, padding, anchors. */
    .blk h1, .blk h2, .blk h3 { margin: 0 !important; padding: 0 !important; border: 0; color: var(--ink); }
    .blk [data-testid="stHeaderActionElements"] { display: none !important; }

    /* Keyboard focus: one consistent cobalt ring everywhere */
    .stApp :is(button, summary, a, textarea, input, [tabindex]):focus-visible {
        outline: 3px solid var(--cobalt) !important; outline-offset: 2px;
    }

    /* ---------- Masthead ---------- */
    .mast { border-top: var(--b4) solid var(--ink); border-bottom: var(--b2) solid var(--ink); padding: 8px 0 10px 0; margin-bottom: 16px; }
    .mast-top { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 4px 24px; }
    .blk .mast-name {
        font-family: var(--display); font-weight: 800;
        font-size: clamp(1.6rem, 3vw, 2.1rem) !important; line-height: 1.05 !important; letter-spacing: -0.02em;
    }
    .mast-meta { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); padding-bottom: 2px; }
    .mast-sub { font-size: 0.97rem; color: var(--muted); margin-top: 10px; max-width: 64ch; line-height: 1.5; }

    /* Section numbering rule used above each region */
    .sec-no {
        display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px 12px; margin-bottom: 10px;
        padding-bottom: 6px; border-bottom: var(--b1) solid var(--ink);
        font-family: var(--mono); font-size: 0.78rem; font-weight: 600; color: var(--ink);
    }
    .sec-no span + span { font-weight: 400; color: var(--muted); }

    /* ---------- Regions ---------- */
    .st-key-resume_col { border-right: var(--b2) solid var(--ink); padding-right: 24px; gap: 0 !important; }
    .st-key-assistant_col { padding-left: 8px; gap: 0.5rem !important; }
    /* The conversation scrolls inside its own region, so the composer and the
       newest answer stay on screen and the whole page does not have to scroll. */
    .st-key-record { max-height: calc(100vh - 385px); min-height: 220px; overflow-y: auto; padding: 2px 8px 2px 2px; scrollbar-gutter: stable; }
    .st-key-resume_col [data-testid="stAlert"] { margin-bottom: 12px; }
    .st-key-resume_col [data-testid="stFileUploader"] { margin-top: 12px; }
    .st-key-resume_col [data-testid="stCaptionContainer"] { margin-top: 8px; }

    /* ---------- Resume dossier ---------- */
    .dossier { background: var(--white); border: var(--b2) solid var(--ink); border-bottom: 0; }
    .dossier-bar {
        display: flex; justify-content: space-between; gap: 12px; padding: 6px 12px;
        background: var(--ink); color: var(--paper); font-family: var(--mono); font-size: 0.76rem;
    }
    .dossier-body { padding: 12px 12px 10px 12px; }
    .blk .doc-name {
        font-family: var(--display); font-weight: 800;
        font-size: 1.45rem !important; line-height: 1.1 !important; letter-spacing: -0.015em; overflow-wrap: anywhere;
    }
    .doc-file { font-family: var(--mono); font-size: 0.76rem; color: var(--muted); margin-top: 6px; overflow-wrap: anywhere; line-height: 1.35; }
    .spec { margin: 10px 0 0 0; border-top: var(--b1) solid var(--ink); }
    .spec-row { display: grid; grid-template-columns: 54px minmax(0, 1fr); gap: 8px; padding: 6px 0; border-bottom: var(--b1) solid var(--concrete); }
    .spec dt { font-family: var(--mono); font-size: 0.76rem; color: var(--muted); margin: 0; padding-top: 2px; }
    .spec dd { margin: 0; font-size: 0.94rem; line-height: 1.45; color: var(--ink); overflow-wrap: anywhere; }
    .spec .ok { color: var(--ok); font-weight: 600; font-size: 0.9rem; }
    .spec-note { font-size: 0.78rem; color: var(--muted); line-height: 1.4; margin: 8px 0 0 0; }

    /* Section index: attaches to the dossier; each row drafts a question */
    .st-key-section_index { gap: 0 !important; background: var(--white); border: var(--b2) solid var(--ink); border-top: var(--b1) solid var(--ink); }
    .idx-head {
        display: flex; justify-content: space-between; gap: 8px; padding: 6px 12px;
        font-family: var(--mono); font-size: 0.74rem; color: var(--muted); border-bottom: var(--b1) solid var(--concrete);
    }

    .side-lead { font-family: var(--display); font-weight: 800; font-size: 1.5rem; letter-spacing: -0.015em; line-height: 1.1; }
    .side-text { font-size: 0.95rem; color: var(--muted); line-height: 1.5; margin-top: 6px; }

    /* ---------- Assistant column ---------- */
    .blk .lead {
        font-family: var(--display); font-weight: 800;
        font-size: clamp(1.4rem, 2.4vw, 1.75rem) !important; line-height: 1.1 !important; letter-spacing: -0.015em;
    }
    .lead-sub { font-size: 0.95rem; color: var(--muted); margin: 8px 0 0 0; }
    .steps { list-style: none; counter-reset: step; margin: 12px 0 0 0; padding: 0; border-top: var(--b2) solid var(--ink); }
    .steps li {
        counter-increment: step; display: flex; gap: 12px; align-items: baseline; padding: 10px 0;
        border-bottom: var(--b1) solid var(--concrete); font-size: 0.97rem; color: var(--ink);
    }
    .steps li::before { content: counter(step, decimal-leading-zero); font-family: var(--mono); font-size: 0.8rem; font-weight: 600; color: var(--muted); }
    .field-label { font-family: var(--mono); font-size: 0.76rem; color: var(--muted); margin: 0; }

    /* Composer */
    .st-key-assistant_col .stTextArea [data-baseweb="textarea"] { border-color: transparent !important; background: transparent !important; border-radius: 0 !important; }
    .st-key-assistant_col .stTextArea textarea {
        background: var(--white) !important; color: var(--ink) !important; font-family: var(--sans);
        border: var(--b2) solid var(--ink) !important; border-radius: 0 !important;
        font-size: 1rem; line-height: 1.5; caret-color: var(--cobalt);
    }
    .st-key-assistant_col .stTextArea textarea::placeholder { color: #666B71 !important; opacity: 1; }
    .st-key-assistant_col .stTextArea textarea:focus { border-color: var(--cobalt) !important; box-shadow: 0 0 0 1px var(--cobalt) !important; }
    .st-key-assistant_col .stTextArea textarea:disabled { background: #EAE9E4 !important; -webkit-text-fill-color: var(--muted); }
    [data-testid="stForm"] { border: 0 !important; padding: 0 !important; background: transparent !important; }

    .st-key-assistant_col [data-testid="stBaseButton-primaryFormSubmit"] {
        background: var(--ink) !important; color: var(--white) !important; border: var(--b2) solid var(--ink) !important;
        border-radius: 0 !important; font-weight: 700 !important; min-height: 44px; padding: 0 12px;
        white-space: nowrap; transition: background-color .1s ease, border-color .1s ease;
    }
    .st-key-assistant_col [data-testid="stBaseButton-primaryFormSubmit"] p { color: var(--white) !important; white-space: nowrap; }
    .st-key-assistant_col [data-testid="stBaseButton-primaryFormSubmit"]:hover:not(:disabled) { background: var(--cobalt) !important; border-color: var(--cobalt) !important; }
    .stApp button:disabled { opacity: 0.45 !important; cursor: not-allowed !important; }

    /* Selectable rows: suggested questions and the section index.
       secondary = available, primary = currently in the question box. */
    :is(.st-key-suggestions, .st-key-section_index) :is([data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"]) {
        width: 100%; justify-content: flex-start !important; text-align: left; min-height: 38px; height: auto;
        padding: 6px 12px 6px 10px; border: 0 !important; border-bottom: var(--b1) solid var(--concrete) !important;
        border-left: var(--b4) solid transparent !important; border-radius: 0 !important;
        background: transparent !important; box-shadow: none !important;
        transition: background-color .1s ease, border-left-color .1s ease;
    }
    :is(.st-key-suggestions, .st-key-section_index) :is([data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"]) > div,
    :is(.st-key-suggestions, .st-key-section_index) :is([data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"]) [data-testid="stMarkdownContainer"] {
        width: 100%; text-align: left; justify-content: flex-start;
    }
    :is(.st-key-suggestions, .st-key-section_index) :is([data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"]) p {
        color: var(--ink) !important; font-size: 0.97rem; font-weight: 500; text-align: left; margin: 0;
    }
    :is(.st-key-suggestions, .st-key-section_index) code {
        font-family: var(--mono); font-size: 0.8rem; background: transparent; color: var(--muted); padding: 0 8px 0 0; border: 0;
    }
    .st-key-suggestions { gap: 0 !important; border-top: var(--b2) solid var(--ink); }
    /* The trailing word states what a click does; always visible on touch */
    :is(.st-key-suggestions, .st-key-section_index) [data-testid="stBaseButton-secondary"]::after {
        content: "Use"; margin-left: auto; padding-left: 12px; font-family: var(--mono); font-size: 0.74rem;
        color: var(--muted); opacity: 1; transition: opacity .1s ease;
    }
    @media (hover: hover) and (pointer: fine) {
        :is(.st-key-suggestions, .st-key-section_index) [data-testid="stBaseButton-secondary"]::after { opacity: 0; }
    }
    :is(.st-key-suggestions, .st-key-section_index) [data-testid="stBaseButton-secondary"]:is(:hover:not(:disabled), :focus-visible)::after { opacity: 1; color: var(--cobalt); }
    .st-key-suggestions [data-testid="stBaseButton-secondary"]:is(:hover:not(:disabled), :focus-visible) { background: var(--white) !important; border-left-color: var(--cobalt) !important; }
    .st-key-section_index [data-testid="stBaseButton-secondary"]:is(:hover:not(:disabled), :focus-visible) { background: var(--paper) !important; border-left-color: var(--cobalt) !important; }
    :is(.st-key-suggestions, .st-key-section_index) [data-testid="stBaseButton-primary"] { background: var(--white) !important; border-left-color: var(--cobalt) !important; }
    :is(.st-key-suggestions, .st-key-section_index) [data-testid="stBaseButton-primary"] p { font-weight: 700; }
    :is(.st-key-suggestions, .st-key-section_index) [data-testid="stBaseButton-primary"]::after {
        content: "In box"; margin-left: auto; padding-left: 12px; font-family: var(--mono); font-size: 0.74rem; font-weight: 600; color: var(--cobalt);
    }

    /* Upload */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--white) !important; border: var(--b2) solid var(--ink) !important; border-radius: 0 !important;
        padding: 14px !important; transition: border-color .1s ease, box-shadow .1s ease;
    }
    [data-testid="stFileUploaderDropzone"]:is(:hover, :focus-within) { border-color: var(--cobalt) !important; box-shadow: 4px 4px 0 var(--concrete); }
    [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
    [data-testid="stFileUploaderDropzone"] small { color: var(--muted) !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--white) !important; border: var(--b2) solid var(--ink) !important; border-radius: 0 !important; min-height: 44px;
    }
    [data-testid="stFileUploaderDropzone"] button:hover { border-color: var(--cobalt) !important; }
    [data-testid="stFileUploaderDropzone"] button p { font-weight: 700; }

    /* ---------- Conversation record ---------- */
    .rec { border-top: var(--b2) solid var(--ink); }
    .rec-q { display: grid; grid-template-columns: 56px minmax(0, 1fr) auto; gap: 6px 12px; align-items: baseline; padding: 10px 0 10px 16px; }
    .rec-no { font-family: var(--mono); font-size: 0.82rem; font-weight: 700; color: var(--ink); }
    .blk .rec .q { font-family: var(--sans); font-size: 1.02rem !important; font-weight: 700; line-height: 1.35 !important; overflow-wrap: anywhere; }
    .verdict {
        display: inline-flex; align-items: center; gap: 6px; padding: 2px 8px; border: var(--b1) solid currentColor;
        font-family: var(--mono); font-size: 0.74rem; font-weight: 700; line-height: 1.5; white-space: nowrap;
    }
    .verdict.ok { color: var(--ok); }
    .verdict.warn { color: var(--warn); }
    .verdict.bad { color: var(--err); }
    .verdict.wait { color: var(--ink); }
    .rec-a {
        display: grid; grid-template-columns: 56px minmax(0, 1fr); gap: 0 12px; background: var(--white);
        border: var(--b1) solid var(--concrete); border-left: var(--b4) solid var(--ink); padding: 16px 16px 14px 12px;
    }
    .rec.earlier .rec-a { border-left-color: #8D9096; }
    .rec-label { font-family: var(--mono); font-size: 0.74rem; color: var(--muted); padding-top: 4px; }
    .a-wrap { min-width: 0; }
    .rec .a { font-size: 1.02rem; line-height: 1.68; color: var(--ink); max-width: 70ch; overflow-wrap: anywhere; }
    .rec .a p { margin: 0 0 0.75em 0 !important; padding: 0 !important; color: var(--ink); line-height: 1.68; }
    .rec .a h4 { margin: 1.1em 0 0.3em 0 !important; padding: 0 !important; font-size: 1rem; font-weight: 800; color: var(--ink); line-height: 1.35; }
    .rec .a h4:first-child { margin-top: 0; }
    .rec .a ul, .rec .a ol { margin: 0 0 0.75em 0 !important; padding-top: 0; padding-bottom: 0; padding-left: 1.3em; }
    .rec .a li { margin: 0 0 0.25em 0 !important; padding: 0; color: var(--ink); line-height: 1.6; }
    .rec .a :is(p, ul, ol):last-child { margin-bottom: 0; }
    .rec-meta { font-family: var(--mono); font-size: 0.76rem; color: var(--muted); margin-top: 12px; padding-top: 8px; border-top: var(--b1) solid var(--concrete); }

    .status { font-size: 0.97rem; color: var(--ink); }
    .status-sub { color: var(--muted); font-size: 0.86rem; margin-top: 8px; }
    .bar { position: relative; height: 4px; margin-top: 10px; background: var(--concrete); overflow: hidden; }
    .bar::after { content: ""; position: absolute; top: 0; bottom: 0; left: -30%; width: 30%; background: var(--ink); animation: slide 1.3s linear infinite; }
    .failure { grid-column: 1 / -1; background: var(--white); border: var(--b2) solid var(--err); padding: 12px 16px; margin-top: 4px; }
    .failure-title { font-weight: 800; color: var(--err); font-size: 0.97rem; }
    .failure-msg { color: var(--ink); font-size: 0.94rem; line-height: 1.55; margin-top: 4px; overflow-wrap: anywhere; }
    .failure-hint { color: var(--muted); font-size: 0.86rem; margin-top: 6px; }

    @keyframes slide { from { left: -30%; } to { left: 100%; } }
    @media (prefers-reduced-motion: reduce) {
        .bar::after { animation: none; left: 0; width: 100%; opacity: 0.35; }
        .stApp * { transition: none !important; }
    }

    /* ---------- Native widgets, made square ---------- */
    [data-testid="stExpander"], [data-testid="stExpander"] details {
        background: transparent !important; border: 0 !important; border-radius: 0 !important; box-shadow: none !important;
    }
    [data-testid="stExpander"] details { border-top: var(--b2) solid var(--ink) !important; }
    [data-testid="stExpander"] summary { padding: 9px 0; min-height: 40px; background: transparent !important; }
    [data-testid="stExpander"] summary:hover { background: var(--white) !important; }
    [data-testid="stExpander"] summary svg { color: var(--ink) !important; fill: var(--ink) !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--ink) !important; font-size: 0.9rem; font-weight: 700; }
    [data-testid="stExpanderDetails"] { background: transparent !important; padding: 4px 0 12px 0; }
    .st-key-resume_col [data-testid="stExpander"] details { background: var(--white) !important; border: var(--b2) solid var(--ink) !important; border-top: var(--b1) solid var(--ink) !important; }
    .st-key-resume_col [data-testid="stExpander"] summary { padding: 9px 12px; }
    .st-key-resume_col [data-testid="stExpanderDetails"] { padding: 4px 12px 12px 12px; }
    [data-testid="stMetric"] { padding: 2px 0; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: var(--muted); font-family: var(--mono); font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }
    /* Alerts: rectangular, kind shown by left rule AND by the message text */
    div[data-testid="stAlert"] {
        background: var(--white) !important; border: var(--b2) solid var(--ink); border-left: var(--b4) solid var(--ink);
        border-radius: 0; padding: 0.55rem 0.75rem;
    }
    div[data-testid="stAlert"], div[data-testid="stAlert"] * { color: var(--ink) !important; }
    div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { border-color: var(--warn); }
    div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { border-color: var(--err); }
    div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { border-color: var(--cobalt); }
    [data-testid="stSpinner"], [data-testid="stSpinner"] * { color: var(--ink) !important; font-family: var(--mono); font-size: 0.84rem; }
    .page-note { color: var(--muted); font-size: 0.82rem; padding-top: 8px; margin: 8px 0 0 8px; border-top: var(--b1) solid var(--concrete); }

    @media (pointer: coarse) {
        :is(.st-key-suggestions, .st-key-section_index) :is([data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"]) { min-height: 44px; }
        [data-testid="stExpander"] summary { min-height: 44px; }
    }

    /* ---------- Responsive ---------- */
    /* Tablet and below: stack the dossier above the assistant at full width.
       Best-effort: it relies on Streamlit's column test ids (Streamlit also
       stacks columns by itself below ~640px). */
    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"]:has(.st-key-resume_col) { flex-wrap: wrap !important; }
        [data-testid="stHorizontalBlock"]:has(.st-key-resume_col) > [data-testid="stColumn"],
        [data-testid="stHorizontalBlock"]:has(.st-key-resume_col) > [data-testid="column"] {
            flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important;
        }
        .st-key-resume_col { border-right: 0; padding-right: 0; border-bottom: var(--b2) solid var(--ink); padding-bottom: 24px; }
        .st-key-assistant_col { padding-left: 0; }
        .st-key-record { max-height: none; overflow: visible; padding: 0; }
        .page-note { margin-left: 0; }
    }
    @media (max-width: 640px) {
        .block-container { padding-top: 2.2rem; }
        .rec-q { grid-template-columns: 44px minmax(0, 1fr); padding-left: 12px; }
        .rec-q .verdict { grid-column: 2; justify-self: start; }
        .rec-a { grid-template-columns: minmax(0, 1fr); padding: 12px; }
        .rec-label { padding: 0 0 6px 0; }
        .spec-row { grid-template-columns: 64px minmax(0, 1fr); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESUME-DERIVED INFO (local only: no API / LLM calls)
# ============================================================
# Everything here is best-effort. Failure never blocks upload or chat.
# BEGIN_LOCAL_PROFILE

SECTION_ALIASES = {
    "projects": {
        "projects", "project", "personal projects", "academic projects",
        "selected projects", "capstone project", "capstone projects",
        "project experience",
    },
    "skills": {
        "skills", "technical skills", "skills summary", "core skills",
        "technologies", "tech stack", "technical proficiencies",
    },
    "experience": {
        "experience", "work experience", "professional experience",
        "internship", "internships", "internship experience", "employment",
        "employment history", "work history", "relevant experience",
    },
    "education": {
        "education", "educational background", "academic background",
        "academic qualifications",
    },
    "certifications": {
        "certifications", "certificates", "licenses and certifications",
    },
}

SECTION_LABELS = {
    "projects": "Projects",
    "skills": "Skills",
    "experience": "Experience",
    "education": "Education",
    "certifications": "Certifications",
}

ALL_HEADERS = set().union(*SECTION_ALIASES.values())

# Terms that are only shown if they literally appear in the resume text.
SKILL_KEYWORDS = {
    "python": "Python", "java": "Java", "javascript": "JavaScript",
    "typescript": "TypeScript", "c++": "C++", "c#": "C#", "php": "PHP",
    "sql": "SQL", "kotlin": "Kotlin", "swift": "Swift", "ruby": "Ruby",
    "golang": "Go", "rust": "Rust", "html": "HTML", "css": "CSS",
    "react": "React", "django": "Django", "flask": "Flask",
    "fastapi": "FastAPI", "docker": "Docker", "git": "Git",
    "pytest": "pytest", "unittest": "unittest", "selenium": "Selenium",
    "jest": "Jest", "cypress": "Cypress", "postman": "Postman",
    "junit": "JUnit", "playwright": "Playwright",
}

NAME_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.'\-]*$")

# Evidence patterns used when a resume has no recognizable section heading.
ROLE_WORDS = re.compile(
    r"\b(engineer|developer|analyst|intern|manager|tester|qa|assistant|"
    r"consultant|designer|administrator|specialist|associate)\b",
    re.I,
)
DATE_RANGE = re.compile(
    r"(19|20)\d{2}\s*(?:-|–|—|to)\s*((19|20)\d{2}|present|current)", re.I
)
EDUCATION_WORDS = re.compile(
    r"\b(university|college|bachelor|bachelor's|degree|diploma|b\.?sc?\.?)\b", re.I
)

# Generic, recruiter-friendly wording. Each question is offered only when
# the uploaded resume contains evidence to support it.
Q_SKILLS = "What are this candidate's strongest technical skills?"
Q_PROJECTS = "What projects has this candidate worked on?"
Q_EXPERIENCE = "What relevant work experience does this candidate have?"
Q_EDUCATION = "What is this candidate's educational background?"
Q_SUMMARY = "Can you summarize this candidate's background?"


def _extract_pdf_text(file_bytes):
    """Best-effort local text extraction using whichever PDF library is
    already installed. Returns "" if none is available or extraction fails."""
    def extract_pages(reader):
        extracted = []
        for page in reader.pages[:6]:
            try:
                page_text = page.extract_text(extraction_mode="layout")
            except TypeError:
                # Older pypdf/PyPDF2 versions may not support layout mode.
                page_text = page.extract_text()
            extracted.append(page_text or "")
        return "\n".join(extracted)

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(file_bytes))
        return extract_pages(reader)
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(file_bytes))
        return extract_pages(reader)
    except Exception:
        pass
    try:
        import fitz  # type: ignore

        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in list(doc)[:6])
    except Exception:
        pass
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages[:6])
    except Exception:
        pass
    return ""


def _normalize_header(line):
    return re.sub(r"[^a-z ]", "", line.lower()).strip()


def _split_sections(lines):
    sections = {}
    current = None
    for line in lines:
        if len(line) <= 40:
            norm = _normalize_header(line)
            matched = next(
                (key for key, names in SECTION_ALIASES.items() if norm in names),
                None,
            )
            if matched:
                current = matched
                sections.setdefault(current, [])
                continue
        if current:
            sections[current].append(line)
    return sections


def _extract_candidate_name(lines):
    for line in lines[:5]:
        if len(line) > 40 or re.search(r"[\d@]|http|www\.", line, re.I):
            continue
        if _normalize_header(line) in ALL_HEADERS:
            continue
        tokens = line.split()
        if not 2 <= len(tokens) <= 4:
            continue
        if not all(NAME_TOKEN.match(t) for t in tokens):
            continue
        if line.isupper():
            return line.title()
        if all(t[0].isupper() for t in tokens):
            return line
    return None


def _find_listed_skills(lines):
    """Return skill keywords that literally appear in the given lines."""
    skills_text = " ".join(lines).lower()
    found = []
    for key, label in SKILL_KEYWORDS.items():
        pattern = rf"(?<![A-Za-z0-9+#]){re.escape(key)}(?![A-Za-z0-9+#])"
        if re.search(pattern, skills_text):
            found.append(label)
    return found[:8]


def _build_suggestions(text, sections):
    """Up to three generic, self-explanatory questions. A question is offered
    when its section was found OR the resume text contains clear evidence for
    it, so unusual headings do not make every suggestion disappear."""
    low = text.lower()

    def section_len(key):
        return len(" ".join(sections.get(key, [])))

    has_skills = (
        section_len("skills") > 20
        or len(_find_listed_skills([text])) >= 3
    )
    has_projects = section_len("projects") > 40 or len(re.findall(r"\bproject", low)) >= 2
    has_experience = (
        section_len("experience") > 60
        or "intern" in low
        or bool(DATE_RANGE.search(text) and ROLE_WORDS.search(text))
    )
    has_education = section_len("education") > 20 or bool(EDUCATION_WORDS.search(text))

    suggestions = []
    if has_skills:
        suggestions.append(Q_SKILLS)
    if has_projects:
        suggestions.append(Q_PROJECTS)
    if has_experience:
        suggestions.append(Q_EXPERIENCE)

    # Fill up to two suggestions so the assistant never looks empty.
    if len(suggestions) < 2 and has_education:
        suggestions.append(Q_EDUCATION)
    if len(suggestions) < 2:
        suggestions.append(Q_SUMMARY)

    return suggestions[:3]


def build_resume_profile(file_bytes, pages):
    """Derive display info and suggested questions from the PDF locally.
    Never raises; falls back to a profile with only the generic question."""
    profile = {
        "name": None,
        "pages": pages,
        "sections": [],
        "skills": [],
        "skills_source": None,
        "suggestions": [Q_SUMMARY],
        "extracted": False,
    }
    try:
        text = _extract_pdf_text(file_bytes)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return profile
        sections = _split_sections(lines)
        profile["extracted"] = True
        profile["name"] = _extract_candidate_name(lines)
        profile["sections"] = [
            SECTION_LABELS[key] for key in SECTION_LABELS
            if sections.get(key)
        ]
        skills = _find_listed_skills(sections.get("skills", []))
        if skills:
            profile["skills"], profile["skills_source"] = skills, "section"
        else:
            mentioned = _find_listed_skills(lines)
            if len(mentioned) >= 2:
                profile["skills"], profile["skills_source"] = mentioned, "text"
        profile["suggestions"] = _build_suggestions("\n".join(lines), sections)
    except Exception:
        pass
    return profile

# END_LOCAL_PROFILE


def use_suggestion(question_text):
    """Button callback: only fills the question box. Callbacks run before the
    script reruns, so this is safe even though the text area uses this key.
    Submission still goes through the single Ask button."""
    st.session_state.question_input = question_text
    st.session_state.composer_notice = None


def submit_question():
    """Ask callback. Validates the question and queues it; the HTTP request
    runs once, at the end of the next script run, while inputs are disabled."""
    if st.session_state.resume_upload_state != "active":
        return
    if st.session_state.pending_question is not None:
        return  # a request is already queued or running
    question = (st.session_state.question_input or "").strip()
    st.session_state.last_error = None
    st.session_state.composer_notice = None
    if not question:
        st.session_state.composer_notice = (
            "Enter a question, or pick a suggestion, before continuing."
        )
        return
    st.session_state.pending_question = question
    st.session_state.question_input = ""


# ============================================================
# BACKEND REQUESTS (unchanged)
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
# UI HELPERS
# ============================================================

def _h(markup):
    """Collapse a multi-line HTML template into one line. Blank lines would
    otherwise end an HTML block in Markdown and break the layout."""
    return "".join(line.strip() for line in markup.splitlines())


def blk(markup):
    """Root wrapper for our own HTML blocks (see the .blk rule in the CSS)."""
    return f'<div class="blk">{markup}</div>'


def esc(value):
    return html.escape(str(value))


def esc_inline(value):
    """Escape text and collapse any line breaks to spaces. Needed because _h()
    joins lines without a separator, which would glue words together."""
    return html.escape(" ".join(str(value).split()))


# A line that starts a bullet ("- ", "* ", a bullet dot) or a number ("1.", "2)").
_LIST_LINE = re.compile(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+(\S.*?)\s*$")
_NUMBERED_LINE = re.compile(r"^\s*\d{1,2}[.)]\s")
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,4}\s+(\S.*?)\s*#*\s*$")


def _answer_inline(text):
    """Escape first, then turn **bold** into <strong>. Because the text is
    already escaped, the only tags that can appear are the ones added here."""
    safe = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)


def answer_to_html(answer):
    """Escape the model's text and give it light structure: headings,
    paragraphs, bullet and numbered lists, and **bold**. Single-line output."""
    text = answer.strip().replace("\r\n", "\n")
    out = []
    for block in re.split(r"\n\s*\n", text):
        groups = []  # [kind, [lines]] where kind is "h", "p", "ul" or "ol"
        for line in block.split("\n"):
            if not line.strip():
                continue
            heading = _HEADING_LINE.match(line)
            if heading:
                groups.append(["h", [heading.group(1)]])
                continue
            match = _LIST_LINE.match(line)
            if match:
                kind = "ol" if _NUMBERED_LINE.match(line) else "ul"
                content = match.group(1)
            else:
                kind, content = "p", line.strip()
            if groups and groups[-1][0] == kind:
                groups[-1][1].append(content)
            else:
                groups.append([kind, [content]])
        for kind, lines in groups:
            if kind == "h":
                out.append(f"<h4>{_answer_inline(lines[0])}</h4>")
            elif kind == "p":
                out.append("<p>" + "<br>".join(_answer_inline(ln) for ln in lines) + "</p>")
            else:
                items = "".join(f"<li>{_answer_inline(ln)}</li>" for ln in lines)
                out.append(f"<{kind}>{items}</{kind}>")
    return "".join(out)


def format_elapsed(seconds):
    if seconds < 1:
        return "under 1 s"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    return f"{seconds // 60} min {seconds % 60} s"


def explain_failure(message, elapsed):
    """Make the read-timeout message accurate and stop it inviting an
    immediate retry. Only the known timeout text is rewritten; every other
    backend message is shown exactly as ask_backend produced it."""
    if message.startswith("The AI backend took too long to respond"):
        return (
            "The AI backend took too long to respond: nothing came back in "
            f"{format_elapsed(elapsed)} (limit {REQUEST_TIMEOUT_SECONDS} s). "
            "It may still be working on this question. Check the FastAPI log "
            "before asking again, because a second request can queue behind the first."
        )
    return message


def pluralize_pages(pages):
    return f"{pages} page" if pages == 1 else f"{pages} pages"


# Each detected section can draft a (generic) question about itself. Only
# sections that build_resume_profile actually found are ever offered.
SECTION_QUESTIONS = {
    "Projects": Q_PROJECTS,
    "Skills": Q_SKILLS,
    "Experience": Q_EXPERIENCE,
    "Education": Q_EDUCATION,
    "Certifications": "What certifications does this candidate hold?",
}


def turn_html(question, body_html="", earlier=False, verdict_html="", tag="Q"):
    """One record in the conversation: numbered question, status, then body."""
    css_class = "rec blk earlier" if earlier else "rec blk"
    return _h(
        f'<div class="{css_class}"><div class="rec-q">'
        f'<span class="rec-no">{esc(tag)}</span>'
        f'<h3 class="q">{esc_inline(question)}</h3>{verdict_html}</div>{body_html}</div>'
    )


def loading_html(question, tag="Q"):
    """Status text + sliding bar. It is drawn BEFORE the blocking HTTP call,
    and the animation is CSS, so it keeps moving while Python waits."""
    return turn_html(
        question,
        '<div class="rec-a"><span class="rec-label">Status</span><div class="a-wrap">'
        '<div class="status" role="status" aria-live="polite">Waiting for the local AI backend...</div>'
        '<div class="bar" aria-hidden="true"></div>'
        '<div class="status-sub">Local models can take a while. Keep this page open; '
        "a new question can be asked only after this one finishes.</div></div></div>",
        verdict_html='<span class="verdict wait">Processing</span>',
        tag=tag,
    )


def error_html(question, message):
    return turn_html(
        question,
        '<div class="rec-a" style="border-left-color:var(--err)"><span class="rec-label">Status</span>'
        '<div class="a-wrap"><div class="failure" role="alert" style="border:0;padding:0;margin:0">'
        '<div class="failure-title">Could not get an answer</div>'
        f'<div class="failure-msg">{esc(message)}</div>'
        '<div class="failure-hint">Your question is back in the box. '
        "Nothing was sent again automatically.</div></div></div></div>",
        verdict_html='<span class="verdict bad"><span aria-hidden="true">\u00d7</span><span>Failed</span></span>',
        tag="ERR",
    )


def resume_dossier_html(filename, profile):
    """The active resume as a dossier: header bar, candidate (when found),
    file line, and a ruled spec table. Only facts the app really has are
    shown, and every value is escaped."""
    pages = profile.get("pages")
    candidate = profile.get("name")
    title = candidate or filename
    bar_right = f"PDF, {pluralize_pages(pages)}" if isinstance(pages, int) else "PDF"
    file_line = f'<div class="doc-file">{esc_inline(filename)}</div>' if candidate else ""

    rows = (
        '<div class="spec-row"><dt>Status</dt>'
        '<dd class="ok"><span aria-hidden="true">\u2713</span> Uploaded and validated</dd></div>'
    )
    if profile.get("skills"):
        label = "Skills" if profile.get("skills_source") == "section" else "Mentions"
        rows += f'<div class="spec-row"><dt>{label}</dt><dd>{esc(", ".join(profile["skills"]))}</dd></div>'

    if candidate or profile.get("skills"):
        note = (
            '<div class="spec-note">Read from the PDF text automatically; '
            "may be incomplete.</div>"
        )
    elif not profile.get("extracted"):
        note = (
            '<div class="spec-note">A quick overview could not be read from this PDF. '
            "You can still ask questions.</div>"
        )
    else:
        note = ""

    return blk(_h(
        '<div class="dossier">'
        f'<div class="dossier-bar"><span>Resume file</span><span>{esc(bar_right)}</span></div>'
        '<div class="dossier-body">'
        f'<h2 class="doc-name">{esc_inline(title)}</h2>{file_line}'
        f'<dl class="spec">{rows}</dl>{note}</div></div>'
    ))


def render_turn(result, show_details=False, number=None):
    data = result["data"]
    answer = data.get("answer", "")
    grounded = data.get("grounded", False)
    elapsed = result.get("elapsed")
    earlier = not show_details  # only the newest, settled turn gets full emphasis
    tag = f"Q{number:02d}" if isinstance(number, int) else "Q"

    timing = f"Answered in {format_elapsed(elapsed)}." if elapsed is not None else ""
    has_answer = bool(answer and answer.strip())

    # Verdict first, so grounding can be scanned before the answer is read.
    if grounded:
        verdict = (
            '<span class="verdict ok"><span aria-hidden="true">\u2713</span>'
            "<span>Grounded</span></span>"
        )
    else:
        verdict = (
            '<span class="verdict warn"><span aria-hidden="true">!</span>'
            "<span>Not grounded</span></span>"
        )

    body = ""
    if has_answer:
        meta = (("Marked as grounded by the backend. " if grounded else "") + timing).strip()
        body = (
            '<div class="rec-a"><span class="rec-label">Answer</span><div class="a-wrap">'
            f'<div class="a">{answer_to_html(answer)}</div>'
            + (f'<div class="rec-meta">{esc(meta)}</div>' if meta else "")
            + "</div></div>"
        )

    st.markdown(
        turn_html(result["question"], body, earlier, verdict, tag), unsafe_allow_html=True
    )

    if not has_answer:
        st.info("The AI did not return an answer.")

    if grounded:
        if not has_answer:
            st.caption(f"Marked as grounded by the backend. {timing}".strip())
    else:
        st.warning(
            "The backend did not mark this response as grounded. "
            "Verify it against the uploaded resume."
        )
        if timing and not has_answer:
            st.caption(timing)

    if show_details:
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
            if elapsed is not None:
                st.write(f"**Response time:** {format_elapsed(elapsed)}")


def handle_upload(uploaded_file):
    """Runs the real upload request as soon as a file is chosen. Every
    outcome ends in a rerun with a fresh file picker, so a file is never
    uploaded twice."""
    with st.spinner("Uploading and validating your document..."):
        upload_data, upload_error, outcome = upload_resume_to_backend(uploaded_file)

    if outcome == "success":
        st.session_state.resume_upload_state = "active"
        st.session_state.active_resume_filename = upload_data["filename"]
        st.session_state.latest_result = None
        st.session_state.earlier_results = []
        st.session_state.last_error = None
        st.session_state.composer_notice = None
        st.session_state.question_input = ""
        st.session_state.resume_profile = build_resume_profile(
            uploaded_file.getvalue(), upload_data["pages"]
        )
        st.session_state.upload_feedback = None

    elif outcome == "rejected":
        feedback = [("error", upload_error)]
        if st.session_state.resume_upload_state == "active":
            feedback.append((
                "info",
                "Your previously active resume remains selected: "
                f"{st.session_state.active_resume_filename}.",
            ))
        else:
            feedback.append(("info", "Upload a valid resume before asking questions."))
        st.session_state.upload_feedback = feedback

    else:
        st.session_state.resume_upload_state = "unknown"
        st.session_state.active_resume_filename = None
        st.session_state.latest_result = None
        st.session_state.earlier_results = []
        st.session_state.resume_profile = None
        st.session_state.upload_feedback = [
            ("error", upload_error),
            (
                "warning",
                "The upload could not be confirmed. Upload the PDF again "
                "before asking questions.",
            ),
        ]

    st.session_state.uploader_nonce += 1
    st.rerun()


# ============================================================
# MASTHEAD
# ============================================================

busy = st.session_state.pending_question is not None
has_active = st.session_state.resume_upload_state == "active"

# The description is only needed before a resume is loaded.
masthead_sub = (
    ""
    if has_active
    else '<div class="mast-sub">'
    "Upload a resume, then ask questions about the candidate's skills, projects and experience."
    "</div>"
)
st.markdown(
    blk(_h(
        '<div class="mast"><div class="mast-top">'
        '<h1 class="mast-name">AI Resume Analyzer</h1>'
        '<div class="mast-meta">Local prototype</div></div>'
        f"{masthead_sub}</div>"
    )),
    unsafe_allow_html=True,
)
left_col, right_col = st.columns([3, 7], gap="medium")


# ============================================================
# LEFT: THE RESUME DOSSIER AND SECTION INDEX
# ============================================================

with left_col:
    with st.container(key="resume_col"):
        feedback = st.session_state.upload_feedback
        st.session_state.upload_feedback = None
        for kind, message in feedback or []:
            getattr(st, kind)(message)

        profile = st.session_state.resume_profile or {}
        uploaded_file = None
        current_draft = (st.session_state.question_input or "").strip()

        st.markdown(
            blk('<div class="sec-no"><span>01 / Resume</span></div>'),
            unsafe_allow_html=True,
        )

        if has_active:
            st.markdown(
                resume_dossier_html(st.session_state.active_resume_filename, profile),
                unsafe_allow_html=True,
            )

            # Only sections the local parser really found. Selecting one fills
            # the question box; it never submits.
            found_sections = [s for s in profile.get("sections", []) if s in SECTION_QUESTIONS]
            if found_sections:
                with st.container(key="section_index"):
                    st.markdown(
                        blk('<div class="idx-head"><span>Sections</span>'
                            "<span>Draft a question</span></div>"),
                        unsafe_allow_html=True,
                    )
                    for number, section in enumerate(found_sections, start=1):
                        section_question = SECTION_QUESTIONS[section]
                        st.button(
                            f"`{number:02d}` {section}",
                            key=f"section_{number}",
                            use_container_width=True,
                            type="primary" if current_draft == section_question else "secondary",
                            on_click=use_suggestion,
                            args=(section_question,),
                            disabled=busy,
                        )

            # Zero-width characters change the expander's identity after each
            # upload attempt, so it comes back collapsed instead of staying open.
            replace_label = "Replace resume" + "\u200b" * (st.session_state.uploader_nonce % 4)
            with st.expander(replace_label):
                uploaded_file = st.file_uploader(
                    "Choose a replacement PDF resume",
                    type=["pdf"],
                    accept_multiple_files=False,
                    label_visibility="collapsed",
                    key=f"resume_file_{st.session_state.uploader_nonce}",
                    disabled=busy or PORTFOLIO_PREVIEW,
                )
                st.caption("PDF only. Use a resume you have permission to analyze.")
        else:
            if st.session_state.resume_upload_state == "unknown":
                st.warning("The active resume could not be confirmed. Please upload it again.")
            st.markdown(
                blk(_h(
                    '<h2 class="side-lead">Add a resume</h2>'
                    '<div class="side-text">Upload a PDF to see what it contains and to ask questions about it.</div>'
                )),
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Select a PDF resume",
                type=["pdf"],
                accept_multiple_files=False,
                label_visibility="collapsed",
                key=f"resume_file_{st.session_state.uploader_nonce}",
                disabled=busy or PORTFOLIO_PREVIEW,
            )
            st.caption("PDF only. Use a resume you have permission to analyze.")

        if uploaded_file is not None:
            handle_upload(uploaded_file)


# ============================================================
# RIGHT: ASK AND READ
# ============================================================

with right_col:
    loader_slot = None

    with st.container(key="assistant_col"):
        if has_active:
            st.markdown(
                blk(_h(
                    '<div class="sec-no"><span>02 / Questions</span>'
                    "<span>Answers are drawn from the uploaded resume</span></div>"
                    '<h2 class="lead">Ask about this candidate</h2>'
                )),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                blk(_h(
                    '<div class="sec-no"><span>02 / Questions</span></div>'
                    '<h2 class="lead">Ask about a candidate</h2>'
                    '<div class="lead-sub">Questions are enabled once a resume has been uploaded.</div>'
                    "<ol class=\"steps\">"
                    "<li>Upload a PDF resume.</li>"
                    "<li>Check the details picked out of the document.</li>"
                    "<li>Ask a question and see whether the answer is marked as grounded.</li>"
                    "</ol>"
                )),
                unsafe_allow_html=True,
            )

        # --- Composer (Ctrl+Enter also submits) ---
        with st.form("ask_form", border=False, clear_on_submit=False):
            input_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
            with input_col:
                st.text_area(
                    "Your question about the candidate",
                    key="question_input",
                    height=68,
                    placeholder=(
                        "For example: What programming languages are mentioned?"
                        if has_active
                        else "Upload a resume to start asking questions."
                    ),
                    label_visibility="collapsed",
                    disabled=PORTFOLIO_PREVIEW or not has_active or busy,
                )
            with button_col:
                st.form_submit_button(
                    "Asking..." if busy else "Ask",
                    type="primary",
                    use_container_width=True,
                    disabled=PORTFOLIO_PREVIEW or not has_active or busy,
                    on_click=submit_question,
                )

        if st.session_state.composer_notice:
            st.warning(st.session_state.composer_notice)

        # --- Suggested questions: after a confirmed upload only. They sit
        # directly under the composer because choosing one fills the box. ---
        suggestions = profile.get("suggestions", []) if has_active else []
        has_history = bool(
            st.session_state.latest_result or st.session_state.last_error or busy
        )
        if suggestions:
            if has_history:
                holder = st.expander("Suggested questions", expanded=False)
            else:
                st.markdown(
                    blk('<div class="field-label">Suggested questions</div>'),
                    unsafe_allow_html=True,
                )
                holder = st.container()
            with holder:
                with st.container(key="suggestions"):
                    for index, suggestion in enumerate(suggestions):
                        # Clicking only fills the question box (via callback).
                        st.button(
                            suggestion,
                            key=f"suggest_{index}",
                            use_container_width=True,
                            type="primary" if current_draft == suggestion else "secondary",
                            on_click=use_suggestion,
                            args=(suggestion,),
                            disabled=busy,
                        )

        # --- Record: newest first. Q numbers count the turns shown here.
        # It scrolls in its own region so the composer stays in view. ---
        if has_active:
            latest = st.session_state.latest_result
            earlier = list(st.session_state.earlier_results)
            turns = ([latest] if latest else []) + earlier
            total = len(turns)

            if has_history:
                with st.container(key="record"):
                    if busy:
                        # Drawn before the HTTP request starts; see the block at
                        # the bottom of this file.
                        loader_slot = st.empty()
                        loader_slot.markdown(
                            loading_html(st.session_state.pending_question, f"Q{total + 1:02d}"),
                            unsafe_allow_html=True,
                        )
                    elif st.session_state.last_error:
                        err = st.session_state.last_error
                        st.markdown(error_html(err["question"], err["message"]), unsafe_allow_html=True)

                    if turns:
                        render_turn(turns[0], show_details=not busy, number=total)
                        rest = turns[1:]
                        if rest:
                            with st.expander(f"Earlier questions ({len(rest)})"):
                                for offset, turn in enumerate(rest, start=1):
                                    render_turn(turn, number=total - offset)

    st.markdown(
        blk('<div class="page-note">Answers come from a local model and can be wrong. Check them against the original resume.</div>'),
        unsafe_allow_html=True,
    )


# ============================================================
# RUN THE QUEUED QUESTION (kept last on purpose)
# ============================================================
# By the time this block runs, the typing indicator, the disabled inputs and
# the rest of the page have already been sent to the browser. The synchronous
# HTTP call blocks Python, but the CSS animation keeps running client-side.
# `finally` always clears the queued question, so the UI cannot stay stuck in
# the loading state. There is no automatic retry.

pending_question = st.session_state.pending_question

if pending_question is not None:
    data, error_message = None, None
    started = time.monotonic()
    try:
        try:
            data, error_message = ask_backend(pending_question)
        except Exception:
            data, error_message = None, "Something went wrong while contacting the AI backend."
        elapsed = time.monotonic() - started

        if error_message:
            st.session_state.last_error = {
                "question": pending_question,
                "message": explain_failure(error_message, elapsed),
            }
            st.session_state.restore_question = pending_question
        else:
            st.session_state.last_error = None
            if st.session_state.latest_result:
                st.session_state.earlier_results = (
                    [st.session_state.latest_result] + st.session_state.earlier_results
                )[:9]
            st.session_state.latest_result = {
                "question": pending_question,
                "data": data,
                "elapsed": elapsed,
            }
    finally:
        st.session_state.pending_question = None
        if loader_slot is not None:
            loader_slot.empty()
    st.rerun()


# ============================================================
# SUBTLE UI PREVIEW NOTICE
# ============================================================
if PORTFOLIO_PREVIEW:
    st.caption(
        "UI preview · Resume uploads and AI answers are unavailable "
        "in this demo because the FastAPI and Ollama backend runs locally."
    )
