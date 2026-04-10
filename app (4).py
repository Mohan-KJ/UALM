# ====================================================
#   UALM – Unified Audio Language Model  (Streamlit)
#   Models:
#     • ASR (transcription)  : openai/whisper-base
#     • Sound classification : MKJ007/ast-speech-nonspeech-finetuned
#     • Emotion analysis     : MKJ007/distilbert-emotion-6class-finetuned
#     • VAD                  : rule-based (energy + ZCR + spectral)
#     • Language Detection   : whisper-base (built-in multilingual)
#     • Speaker Diarization  : MFCC cosine-similarity clustering
#     • Translation          : whisper-base (built-in translate task)
#     • Audio Timeline       : segment-level visual timeline
# ====================================================

import os, warnings, tempfile, sys, re
warnings.filterwarnings("ignore")

sys.modules['torchcodec'] = None

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import streamlit as st

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoTokenizer,
    pipeline,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

st.set_page_config(page_title="UALM · AI", page_icon="🎙️", layout="wide")

# ──────────────────────────────────────────────
# STYLES — Apple / Oracle grade premium
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #06060A !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #F0F0F5 !important;
    -webkit-font-smoothing: antialiased !important;
}
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 1280px !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden !important; display:none !important; }

::-webkit-scrollbar { width: 2px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
    0%   { transform: scale(0.92); opacity: 0.55; }
    70%  { transform: scale(1.18); opacity: 0; }
    100% { transform: scale(0.92); opacity: 0; }
}
@keyframes dot-live {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(48,209,88,0.55); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 7px rgba(48,209,88,0); }
}
@keyframes waveform {
    0%, 100% { height: 5px; }
    50%       { height: 26px; }
}

/* ════ HERO ════ */
.hero {
    padding: 48px 0 40px;
    text-align: center;
    position: relative;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 700px; height: 320px;
    transform: translate(-50%, -50%);
    background: radial-gradient(ellipse, rgba(255,255,255,0.025) 0%, transparent 68%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.26);
    margin-bottom: 18px;
}
.hero-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #30D158;
    animation: dot-live 2.4s ease-in-out infinite;
    flex-shrink: 0;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(120px, 18vw, 220px);
    font-weight: 800;
    letter-spacing: -0.058em;
    line-height: 0.88;
    margin-bottom: 24px;
    background: linear-gradient(
        158deg,
        #FFFFFF        0%,
        rgba(255,255,255,0.86) 38%,
        rgba(195,195,220,0.52) 72%,
        rgba(155,155,205,0.32) 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
}
.hero-sub {
    font-size: 14px;
    font-weight: 400;
    color: rgba(255,255,255,0.3);
    line-height: 1.85;
    max-width: 420px;
    width: 100%;
    margin: 0 auto !important;
    text-align: center !important;
    letter-spacing: 0.01em;
    display: block !important;
}

/* ════ PANEL LABEL ════ */
.panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8.5px;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.17);
    margin-bottom: 9px;
    margin-top: 24px;
}
.panel-label:first-child { margin-top: 0; }

/* ════ INPUT MODE TABS ════ */
[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 0 !important;
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    padding: 3px !important;
    width: 100% !important;
}
[data-testid="stRadio"] label {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 8px 0 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.3) !important;
    cursor: pointer !important;
    transition: all 0.16s ease !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.7) !important;
}
[data-testid="stRadio"] input { display: none !important; }
[data-testid="stRadio"] > label { display: none !important; }

/* ════ FILE UPLOADER ════ */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(255,255,255,0.13) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    color: rgba(255,255,255,0.22) !important;
}
[data-testid="stFileUploader"] small { color: rgba(255,255,255,0.14) !important; }

/* ════ AUDIO INPUT (MIC) ════ */
[data-testid="stAudioInput"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stAudioInput"] > label { display: none !important; }

/* ════ AUDIO PLAYER ════ */
audio {
    width: 100%;
    margin-top: 8px;
    filter: invert(1) hue-rotate(180deg) brightness(0.72);
    border-radius: 6px;
}

/* ════ BUTTON ════ */
div.stButton > button {
    background: rgba(255,255,255,0.055) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: rgba(255,255,255,0.75) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 500 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    margin-top: 10px !important;
}
div.stButton > button:hover {
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,255,255,0.18) !important;
    color: #FFFFFF !important;
}

/* ════ SYS ROWS ════ */
.sys-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.038);
}
.sys-row:last-child { border-bottom: none; }
.sys-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.19);
    letter-spacing: 0.08em;
}
.sys-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.04em;
}
.sys-status {
    width: 4px; height: 4px;
    border-radius: 50%;
    background: #30D158;
    flex-shrink: 0;
}

/* ════ SECTION HEADING ════ */
.section-heading {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8.5px;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.16);
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

/* ════ METRICS ════ */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.055) !important;
    border-radius: 10px !important;
    padding: 16px 15px !important;
    animation: fadeUp 0.32s ease both !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 7.5px !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.2) !important;
    font-weight: 400 !important;
}
[data-testid="stMetricValue"] > div {
    font-family: 'Syne', sans-serif !important;
    font-size: 21px !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: #F0F0F5 !important;
    line-height: 1.15 !important;
    margin-top: 4px !important;
}

/* ════ RESULT CARDS ════ */
.rcard {
    background: rgba(255,255,255,0.018);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 9px;
    animation: fadeUp 0.36s ease both;
}
.rcard-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.2);
    margin-bottom: 11px;
}
.rcard-value {
    font-size: 14.5px;
    font-weight: 400;
    color: rgba(255,255,255,0.7);
    line-height: 1.82;
}
.rcard-italic {
    font-size: 14px;
    font-weight: 300;
    font-style: italic;
    color: rgba(255,255,255,0.5);
    line-height: 1.88;
}

/* ════ LANG BADGE ════ */
.lang-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(76,201,240,0.06);
    border: 1px solid rgba(76,201,240,0.16);
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8.5px;
    letter-spacing: 0.12em;
    color: rgba(76,201,240,0.62);
    margin-bottom: 10px;
}

/* ════ SOUND BARS ════ */
.srow {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.032);
}
.srow:last-child { border-bottom: none; padding-bottom: 0; }
.srow-name {
    font-size: 12.5px;
    color: rgba(255,255,255,0.46);
    width: 150px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.srow-track {
    flex: 1;
    height: 1.5px;
    background: rgba(255,255,255,0.055);
    border-radius: 100px;
    overflow: hidden;
}
.srow-fill {
    height: 100%;
    background: rgba(255,255,255,0.28);
    border-radius: 100px;
}
.srow-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.22);
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

/* ════ TIMELINE ════ */
.timeline-bar-wrap {
    position: relative;
    width: 100%;
    height: 22px;
    background: rgba(255,255,255,0.028);
    border-radius: 5px;
    overflow: hidden;
    margin-bottom: 5px;
}
.timeline-seg { position: absolute; height: 100%; }
.timeline-markers {
    display: flex;
    justify-content: space-between;
    margin-bottom: 14px;
}
.timeline-mark {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8.5px;
    color: rgba(255,255,255,0.15);
}
.timeline-legend {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 14px;
}
.legend-dot { width: 7px; height: 7px; border-radius: 2px; flex-shrink: 0; }
.legend-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: rgba(255,255,255,0.3);
    display: flex;
    align-items: center;
    gap: 5px;
}
.drow {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.032);
}
.drow:last-child { border-bottom: none; }
.drow-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.drow-time {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    color: rgba(255,255,255,0.22);
    width: 115px;
    flex-shrink: 0;
}
.drow-speaker { font-size: 12.5px; color: rgba(255,255,255,0.46); }

/* ════ AWAIT BOX ════ */
.await-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 390px;
    border: 1px solid rgba(255,255,255,0.038);
    border-radius: 14px;
    gap: 22px;
    background: rgba(255,255,255,0.008);
}
.await-rings {
    position: relative;
    width: 58px; height: 58px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.await-rings::before,
.await-rings::after {
    content: '';
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.07);
}
.await-rings::before { width: 58px; height: 58px; animation: pulse-ring 2.9s ease-out infinite; }
.await-rings::after  { width: 38px; height: 38px; animation: pulse-ring 2.9s ease-out 0.65s infinite; }
.await-icon-inner { width: 17px; height: 17px; opacity: 0.17; }
.await-waveform {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 30px;
}
.await-bar {
    width: 2px;
    background: rgba(255,255,255,0.09);
    border-radius: 2px;
    animation: waveform 1.5s ease-in-out infinite;
}
.await-bar:nth-child(1) { animation-delay: 0.00s; }
.await-bar:nth-child(2) { animation-delay: 0.12s; }
.await-bar:nth-child(3) { animation-delay: 0.24s; }
.await-bar:nth-child(4) { animation-delay: 0.36s; }
.await-bar:nth-child(5) { animation-delay: 0.48s; }
.await-bar:nth-child(6) { animation-delay: 0.36s; }
.await-bar:nth-child(7) { animation-delay: 0.24s; }
.await-bar:nth-child(8) { animation-delay: 0.12s; }
.await-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8.5px;
    letter-spacing: 0.34em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.13);
}

/* ════ EXPANDER ════ */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.045) !important;
    border-radius: 10px !important;
    margin-top: 9px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.18) !important;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover { color: rgba(255,255,255,0.42) !important; }

[data-testid="stAlert"] {
    background: rgba(255,159,10,0.04) !important;
    border: 1px solid rgba(255,159,10,0.11) !important;
    border-radius: 8px !important;
    font-size: 12.5px !important;
    color: rgba(255,255,255,0.52) !important;
}
[data-testid="stException"] {
    background: rgba(255,69,58,0.04) !important;
    border: 1px solid rgba(255,69,58,0.11) !important;
    border-radius: 8px !important;
}
[data-testid="stSpinner"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    color: rgba(255,255,255,0.26) !important;
}
[data-testid="stHorizontalBlock"] { gap: 48px !important; align-items: start !important; }
hr { border-color: rgba(255,255,255,0.038) !important; margin: 40px 0 !important; }

.footer {
    text-align: center;
    padding: 34px 0 26px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8.5px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.08);
    border-top: 1px solid rgba(255,255,255,0.038);
    margin-top: 48px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">
        <div class="hero-dot"></div>
        Unified Audio Language Model &nbsp;·&nbsp; vMK.1.0
    </div>
    <span class="hero-title">UALM</span>
    <p class="hero-sub" style="text-align:center;margin:0 auto;display:block;">
        Multilingual speech-to-text, speaker diarization,
        sentiment mapping, translation &amp; acoustic classification.
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SAMPLE_RATE  = 16_000
MAX_DURATION = 30
MAX_SAMPLES  = SAMPLE_RATE * MAX_DURATION
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

WHISPER_LANG_MAP = {
    "en": "English",    "zh": "Chinese",     "de": "German",      "es": "Spanish",
    "ru": "Russian",    "ko": "Korean",      "fr": "French",      "ja": "Japanese",
    "pt": "Portuguese", "tr": "Turkish",     "pl": "Polish",      "ca": "Catalan",
    "nl": "Dutch",      "ar": "Arabic",      "sv": "Swedish",     "it": "Italian",
    "id": "Indonesian", "hi": "Hindi",       "fi": "Finnish",     "vi": "Vietnamese",
    "he": "Hebrew",     "uk": "Ukrainian",   "el": "Greek",       "ms": "Malay",
    "cs": "Czech",      "ro": "Romanian",    "da": "Danish",      "hu": "Hungarian",
    "ta": "Tamil",      "no": "Norwegian",   "th": "Thai",        "ur": "Urdu",
    "hr": "Croatian",   "bg": "Bulgarian",   "lt": "Lithuanian",  "la": "Latin",
    "mi": "Maori",      "ml": "Malayalam",   "cy": "Welsh",       "sk": "Slovak",
    "te": "Telugu",     "fa": "Persian",     "lv": "Latvian",     "bn": "Bengali",
    "sr": "Serbian",    "az": "Azerbaijani", "sl": "Slovenian",   "kn": "Kannada",
    "et": "Estonian",   "mk": "Macedonian",  "br": "Breton",      "eu": "Basque",
    "is": "Icelandic",  "hy": "Armenian",    "ne": "Nepali",      "mn": "Mongolian",
    "bs": "Bosnian",    "kk": "Kazakh",      "sq": "Albanian",    "sw": "Swahili",
    "gl": "Galician",   "mr": "Marathi",     "pa": "Punjabi",     "si": "Sinhala",
    "km": "Khmer",      "sn": "Shona",       "yo": "Yoruba",      "so": "Somali",
    "af": "Afrikaans",  "oc": "Occitan",     "ka": "Georgian",    "be": "Belarusian",
    "tg": "Tajik",      "sd": "Sindhi",      "gu": "Gujarati",    "am": "Amharic",
    "yi": "Yiddish",    "lo": "Lao",         "uz": "Uzbek",       "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto",  "tk": "Turkmen",     "nn": "Nynorsk",
    "mt": "Maltese",    "sa": "Sanskrit",    "lb": "Luxembourgish","my": "Myanmar",
    "bo": "Tibetan",    "tl": "Tagalog",     "mg": "Malagasy",    "as": "Assamese",
    "tt": "Tatar",      "haw": "Hawaiian",   "ln": "Lingala",     "ha": "Hausa",
    "ba": "Bashkir",    "jw": "Javanese",    "su": "Sundanese",
}

SPEAKER_COLORS = {
    "Speaker 1": "#4CC9F0",
    "Speaker 2": "#F72585",
    "Speaker 3": "#7B2FBE",
    "Speaker 4": "#FF9F1C",
    "Silence"  : "rgba(255,255,255,0.04)",
}
SPEAKER_COLOR_DEFAULT = "#A8DADC"

# ──────────────────────────────────────────────
# Keywords used to detect non-speech AST labels
# ──────────────────────────────────────────────
NON_SPEECH_KEYWORDS = [
    "non", "noise", "music", "animal", "cat", "dog", "bird",
    "ambient", "environment", "nature", "sound", "effect",
]


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    whisper_proc  = WhisperProcessor.from_pretrained("openai/whisper-base")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-base"
    ).to(DEVICE)
    whisper_model.eval()

    ast_extractor = AutoFeatureExtractor.from_pretrained(
        "MKJ007/ast-speech-nonspeech-finetuned"
    )
    ast_model = AutoModelForAudioClassification.from_pretrained(
        "MKJ007/ast-speech-nonspeech-finetuned"
    ).to(DEVICE)
    ast_model.eval()

    sentiment_pipe = pipeline(
        "text-classification",
        model  = "MKJ007/distilbert-emotion-6class-finetuned",
        device = 0 if DEVICE == "cuda" else -1,
    )
    return whisper_proc, whisper_model, ast_extractor, ast_model, sentiment_pipe


# ──────────────────────────────────────────────
# Audio utilities
# ──────────────────────────────────────────────
def load_and_resample(path: str) -> tuple:
    audio_np, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
    true_duration = len(audio_np) / SAMPLE_RATE
    if len(audio_np) < MAX_SAMPLES:
        audio_np = np.pad(audio_np, (0, MAX_SAMPLES - len(audio_np)))
    return audio_np.astype(np.float32), true_duration


# ──────────────────────────────────────────────
# VAD
# ──────────────────────────────────────────────
def vad_detect(audio_np: np.ndarray) -> dict:
    frame_len = SAMPLE_RATE // 10
    hop       = frame_len // 2
    speech_windows = 0
    total_windows  = 0
    energies = []

    for start in range(0, len(audio_np) - frame_len, hop):
        frame = audio_np[start : start + frame_len]
        rms   = float(np.sqrt(np.mean(frame ** 2)))
        energies.append(rms)
        zcr   = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2)
        fft   = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1 / SAMPLE_RATE)
        sp_e  = float(np.sum(fft[(freqs >= 85) & (freqs < 3000)]))
        tot_e = float(np.sum(fft)) + 1e-8
        sp_r  = sp_e / tot_e
        hi_r  = float(np.sum(fft[freqs >= 5000])) / tot_e
        if rms > 0.008 and 0.03 < zcr < 0.35 and sp_r > 0.40 and hi_r < 0.40:
            speech_windows += 1
        total_windows += 1

    speech_ratio   = speech_windows / max(total_windows, 1)
    speech_present = speech_ratio > 0.08

    if speech_present and len(energies) > 4:
        e_arr = np.array([e for e in energies if e > 0.008])
        if len(e_arr) > 0:
            cv = e_arr.std() / (e_arr.mean() + 1e-8)
            num_speakers = "2+" if cv > 1.3 else "1"
        else:
            num_speakers = "1"
    elif speech_present:
        num_speakers = "1"
    else:
        num_speakers = "0"

    return {
        "speech_present": speech_present,
        "speech_ratio"  : round(speech_ratio, 3),
        "num_speakers"  : num_speakers,
    }


# ──────────────────────────────────────────────
# AST override: check if top sound label is non-speech
# ──────────────────────────────────────────────
def ast_is_nonspeech(sound_labels: list) -> bool:
    """
    Returns True if the AST classifier's top result indicates
    non-human-speech content (animals, noise, music, etc.).
    The fine-tuned model label for non-speech is checked first;
    the keyword list acts as a broad safety net for edge cases.
    """
    if not sound_labels:
        return False
    top_label = sound_labels[0]["label"].lower()
    return any(kw in top_label for kw in NON_SPEECH_KEYWORDS)


# ──────────────────────────────────────────────
# Language Detection
# ──────────────────────────────────────────────
def detect_language(audio_np: np.ndarray, whisper_proc, whisper_model) -> tuple:
    try:
        clip   = audio_np[:SAMPLE_RATE * 30]
        inputs = whisper_proc(
            clip, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            encoder_out = whisper_model.model.encoder(inputs["input_features"])

        sot_id = whisper_proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        decoder_input = torch.tensor([[sot_id]], device=DEVICE)

        with torch.no_grad():
            dec_out = whisper_model.model.decoder(
                input_ids=decoder_input,
                encoder_hidden_states=encoder_out.last_hidden_state,
            )
            logits = whisper_model.proj_out(dec_out.last_hidden_state)[0, -1]

        lang_scores = {}
        unk_id = whisper_proc.tokenizer.unk_token_id
        for code in WHISPER_LANG_MAP:
            tid = whisper_proc.tokenizer.convert_tokens_to_ids(f"<|{code}|>")
            if tid is not None and tid != unk_id:
                lang_scores[code] = logits[tid].item()

        if not lang_scores:
            return "en", "English"

        best = max(lang_scores, key=lang_scores.get)
        return best, WHISPER_LANG_MAP.get(best, best.upper())

    except Exception:
        try:
            inputs = whisper_proc(
                audio_np[:SAMPLE_RATE * 30],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).to(DEVICE)
            with torch.no_grad():
                gen = whisper_model.generate(inputs["input_features"], max_new_tokens=2)
            raw = whisper_proc.tokenizer.decode(gen[0].tolist())
            hits = re.findall(r'<\|([a-z]{2,4})\|>', raw)
            for h in hits:
                if h in WHISPER_LANG_MAP:
                    return h, WHISPER_LANG_MAP[h]
        except Exception:
            pass
        return "en", "English"


# ──────────────────────────────────────────────
# ASR + Translation
# ──────────────────────────────────────────────
def transcribe_and_translate(
    audio_np, whisper_proc, whisper_model, lang_code="en"
) -> tuple:
    inputs = whisper_proc(
        audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).to(DEVICE)

    try:
        with torch.no_grad():
            ids = whisper_model.generate(
                inputs["input_features"],
                language=lang_code, task="transcribe", max_new_tokens=256,
            )
        transcript = whisper_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception:
        with torch.no_grad():
            ids = whisper_model.generate(
                inputs["input_features"], task="transcribe", max_new_tokens=256,
            )
        transcript = whisper_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    transcript = transcript if transcript else "[inaudible]"

    translation = None
    if lang_code != "en" and transcript != "[inaudible]":
        try:
            with torch.no_grad():
                trans_ids = whisper_model.generate(
                    inputs["input_features"],
                    language=lang_code, task="translate", max_new_tokens=256,
                )
            translation = whisper_proc.batch_decode(
                trans_ids, skip_special_tokens=True
            )[0].strip() or None
        except Exception:
            translation = None

    return transcript, translation


# ──────────────────────────────────────────────
# Speaker Diarization
# ──────────────────────────────────────────────
def diarize_speakers(
    audio_np, true_duration, sample_rate=SAMPLE_RATE,
    segment_sec=2.0, max_speakers=4,
    silence_rms=0.006, similarity_threshold=0.82
) -> tuple:
    segment_len      = int(sample_rate * segment_sec)
    active_audio     = audio_np[:int(true_duration * sample_rate)]
    segments         = []
    speaker_profiles = []

    for i in range(0, len(active_audio) - segment_len + 1, segment_len):
        seg     = active_audio[i : i + segment_len]
        start_t = round(i / sample_rate, 2)
        end_t   = round((i + segment_len) / sample_rate, 2)
        rms     = float(np.sqrt(np.mean(seg ** 2)))

        if rms < silence_rms:
            segments.append({
                "start": start_t, "end": end_t,
                "speaker": "Silence", "is_speech": False,
            })
            continue

        mfcc      = librosa.feature.mfcc(y=seg, sr=sample_rate, n_mfcc=20)
        feat      = np.mean(mfcc, axis=1).astype(np.float32)
        norm      = np.linalg.norm(feat) + 1e-8
        feat_norm = feat / norm

        if not speaker_profiles:
            speaker_profiles.append(feat_norm.copy())
            spk_label = "Speaker 1"
        else:
            sims     = [float(np.dot(feat_norm, sp)) for sp in speaker_profiles]
            best_sim = max(sims)
            best_idx = int(np.argmax(sims))

            if best_sim >= similarity_threshold:
                spk_label = f"Speaker {best_idx + 1}"
                updated   = 0.88 * speaker_profiles[best_idx] + 0.12 * feat_norm
                speaker_profiles[best_idx] = updated / (np.linalg.norm(updated) + 1e-8)
            elif len(speaker_profiles) < max_speakers:
                speaker_profiles.append(feat_norm.copy())
                spk_label = f"Speaker {len(speaker_profiles)}"
            else:
                spk_label = f"Speaker {best_idx + 1}"

        segments.append({
            "start": start_t, "end": end_t,
            "speaker": spk_label, "is_speech": True,
        })

    remainder_start = len(segments) * segment_len
    if remainder_start < len(active_audio):
        seg     = active_audio[remainder_start:]
        start_t = round(remainder_start / sample_rate, 2)
        end_t   = round(true_duration, 2)
        rms     = float(np.sqrt(np.mean(seg ** 2)))
        if rms < silence_rms or len(seg) < sample_rate // 2:
            segments.append({
                "start": start_t, "end": end_t,
                "speaker": "Silence", "is_speech": False,
            })
        elif speaker_profiles:
            segments.append({
                "start": start_t, "end": end_t,
                "speaker": "Speaker 1", "is_speech": True,
            })

    return segments, len(speaker_profiles)


# ──────────────────────────────────────────────
# Sound Classification
# ──────────────────────────────────────────────
def classify_sound(audio_np, ast_extractor, ast_model, top_k=2):
    clip   = audio_np[: SAMPLE_RATE * 10]
    inputs = ast_extractor(
        clip, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    ).to(DEVICE)
    with torch.no_grad():
        logits = ast_model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1)[0]
    top_idx = probs.topk(top_k).indices.tolist()
    return [
        {"label": ast_model.config.id2label[i], "score": round(float(probs[i]), 4)}
        for i in top_idx
    ]


# ──────────────────────────────────────────────
# Sentiment / Emotion
# ──────────────────────────────────────────────
def analyse_sentiment(text: str, sentiment_pipe) -> dict:
    if not text or text == "[inaudible]":
        return {"label": "NEUTRAL", "score": 0.0}

    inputs = sentiment_pipe.tokenizer(
        text[:512], return_tensors="pt", truncation=True, padding=True
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(sentiment_pipe.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = sentiment_pipe.model(**inputs).logits

    probs   = F.softmax(logits, dim=-1)[0]
    pred_id = int(probs.argmax())
    label   = sentiment_pipe.model.config.id2label[pred_id]
    return {"label": label, "score": round(float(probs[pred_id]), 4)}


# ──────────────────────────────────────────────
# Caption
# ──────────────────────────────────────────────
def build_caption(vad_out, sound_labels, transcript, sentiment, lang_name) -> str:
    top_sounds = [s["label"] for s in sound_labels[:3]]
    sound_str  = ", ".join(top_sounds)
    if not vad_out["speech_present"]:
        return f"No human speech detected. Dominant sounds: {sound_str}."
    n      = vad_out["num_speakers"]
    spk    = "one person" if n == "1" else "multiple people"
    sent_l = sentiment["label"].lower()
    lang_s = f" (in {lang_name})" if lang_name != "English" else ""
    if transcript and transcript != "[inaudible]":
        short = (transcript[:120] + "…") if len(transcript) > 120 else transcript
        return (
            f'{spk.capitalize()} speaking{lang_s} with a {sent_l} tone. '
            f'Transcript: "{short}" Background: {sound_str}.'
        )
    return (
        f"{spk.capitalize()} speaking{lang_s} (inaudible) "
        f"with a {sent_l} tone. Background: {sound_str}."
    )


# ──────────────────────────────────────────────
# Timeline HTML
# ──────────────────────────────────────────────
def build_timeline_html(segments: list, total_duration: float) -> str:
    if not segments or total_duration <= 0:
        return ""

    bar = '<div class="timeline-bar-wrap">'
    for seg in segments:
        dur     = seg["end"] - seg["start"]
        left    = (seg["start"] / total_duration) * 100
        w       = (dur / total_duration) * 100
        spk     = seg["speaker"]
        col     = SPEAKER_COLORS.get(spk, SPEAKER_COLOR_DEFAULT)
        opacity = "0.09" if spk == "Silence" else "0.76"
        seg_start = seg["start"]
        seg_end   = seg["end"]
        bar += (
            f'<div class="timeline-seg" '
            f'style="left:{left:.2f}%;width:{w:.2f}%;'
            f'background:{col};opacity:{opacity};" '
            f'title="{spk}: {seg_start:.1f}s – {seg_end:.1f}s"></div>'
        )
    bar += "</div>"

    markers = '<div class="timeline-markers">'
    for i in range(5):
        t = (i / 4) * total_duration
        markers += f'<span class="timeline-mark">{t:.1f}s</span>'
    markers += "</div>"

    seen = list(dict.fromkeys(s["speaker"] for s in segments if s["is_speech"]))
    legend = '<div class="timeline-legend">'
    for spk in seen:
        col = SPEAKER_COLORS.get(spk, SPEAKER_COLOR_DEFAULT)
        legend += (
            f'<span class="legend-label">'
            f'<span class="legend-dot" style="background:{col};"></span>'
            f'{spk}</span>'
        )
    if any(not s["is_speech"] for s in segments):
        legend += (
            '<span class="legend-label">'
            '<span class="legend-dot" style="background:rgba(255,255,255,0.14);"></span>'
            'Silence</span>'
        )
    legend += "</div>"

    rows = ""
    for seg in segments:
        spk     = seg["speaker"]
        col     = SPEAKER_COLORS.get(spk, SPEAKER_COLOR_DEFAULT)
        dot_col = col if seg["is_speech"] else "rgba(255,255,255,0.14)"
        rows += (
            f'<div class="drow">'
            f'<div class="drow-dot" style="background:{dot_col};"></div>'
            f'<span class="drow-time">{seg["start"]:.1f}s → {seg["end"]:.1f}s</span>'
            f'<span class="drow-speaker">{spk}</span>'
            f'</div>'
        )

    return bar + markers + legend + rows


# ──────────────────────────────────────────────
# Master Predict
# ──────────────────────────────────────────────
def predict(path, whisper_proc, whisper_model, ast_extractor, ast_model, sentiment_pipe):
    audio_np, true_duration = load_and_resample(path)

    # ── Step 1: VAD (heuristic baseline) ──
    vad_out = vad_detect(audio_np)

    # ── Step 2: Sound classification (AST) — always runs first ──
    sound_labels = classify_sound(audio_np, ast_extractor, ast_model)

    # ── Step 3: AST overrides VAD ──
    # The fine-tuned AST model is more reliable than energy/ZCR heuristics.
    # If AST flags non-speech (animal, noise, music…), suppress the entire
    # speech pipeline regardless of what VAD estimated.
    if ast_is_nonspeech(sound_labels):
        vad_out["speech_present"] = False
        vad_out["num_speakers"]   = "0"
        vad_out["speech_ratio"]   = 0.0

    # ── Step 4: Language detection — only when speech is confirmed ──
    if vad_out["speech_present"]:
        lang_code, lang_name = detect_language(audio_np, whisper_proc, whisper_model)
    else:
        lang_code, lang_name = "—", "N/A"

    # ── Step 5: Diarization — only when speech is confirmed ──
    if vad_out["speech_present"]:
        diar_segments, num_unique = diarize_speakers(audio_np, true_duration)
    else:
        diar_segments, num_unique = [], 0

    # ── Step 6: Transcription + Emotion — only when speech is confirmed ──
    if vad_out["speech_present"]:
        transcript, translation = transcribe_and_translate(
            audio_np, whisper_proc, whisper_model, lang_code
        )
        sentiment_text = translation if translation else transcript
        sentiment      = analyse_sentiment(sentiment_text, sentiment_pipe)
    else:
        transcript  = ""
        translation = None
        sentiment   = {"label": "N/A", "score": 0.0}

    # ── Step 7: Assemble result ──
    n            = vad_out["num_speakers"]
    speech_label = {"0": "No Speech", "1": "1 Speaker"}.get(n, "Multiple")
    caption      = build_caption(vad_out, sound_labels, transcript, sentiment, lang_name)

    return {
        "speech_label"        : speech_label,
        "speech_ratio"        : vad_out["speech_ratio"],
        "transcript"          : transcript or "—",
        "translation"         : translation,
        "sentiment"           : sentiment["label"],
        "sentiment_score"     : sentiment["score"],
        "lang_code"           : lang_code,
        "lang_name"           : lang_name,
        "top_sound_classes"   : sound_labels,
        "primary_sound"       : sound_labels[0]["label"] if sound_labels else "unknown",
        "diarization_segments": diar_segments,
        "num_unique_speakers" : num_unique,
        "total_duration_sec"  : round(true_duration, 2),
        "caption"             : caption,
    }


# ──────────────────────────────────────────────
# Load Models
# ──────────────────────────────────────────────
whisper_proc, whisper_model, ast_extractor, ast_model, sentiment_pipe = load_models()

# ──────────────────────────────────────────────
# UI — 1:2 split
# ──────────────────────────────────────────────
st.markdown("<div style='padding-top:32px;'>", unsafe_allow_html=True)
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<div class="panel-label">Source</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "", ["Upload File", "Live"],
        horizontal=True, label_visibility="collapsed"
    )

    uploaded = None

    if input_mode == "Upload File":
        uploaded = st.file_uploader(
            "audio",
            type=["mp3", "wav", "flac", "ogg", "m4a", "aac"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.audio(uploaded, format=uploaded.type)
    else:
        try:
            mic = st.audio_input("", label_visibility="collapsed")
            if mic:
                uploaded = mic
                st.audio(mic)
        except AttributeError:
            st.markdown(
                '<p style="font-family:IBM Plex Mono,monospace;font-size:9.5px;'
                'color:rgba(255,255,255,0.28);padding:14px 16px;'
                'border:1px solid rgba(255,255,255,0.07);border-radius:10px;'
                'letter-spacing:0.12em;">Requires Streamlit ≥ 1.33</p>',
                unsafe_allow_html=True
            )

    analyse_btn = st.button("Analyse", use_container_width=True)

    st.markdown("""
    <div class="panel-label" style="margin-top:28px;">Stack</div>
    <div>
        <div class="sys-row">
            <span class="sys-key">ASR</span>
            <span class="sys-val">whisper-base</span>
            <div class="sys-status"></div>
        </div>
        <div class="sys-row">
            <span class="sys-key">Classifier</span>
            <span class="sys-val">ast-finetuned ✦</span>
            <div class="sys-status"></div>
        </div>
        <div class="sys-row">
            <span class="sys-key">Emotion</span>
            <span class="sys-val">distilbert-finetuned ✦</span>
            <div class="sys-status"></div>
        </div>
        <div class="sys-row">
            <span class="sys-key">VAD</span>
            <span class="sys-val">rule-based + AST gated</span>
            <div class="sys-status"></div>
        </div>
        <div class="sys-row">
            <span class="sys-key">Lang Det</span>
            <span class="sys-val">whisper-base</span>
            <div class="sys-status"></div>
        </div>
        <div class="sys-row">
            <span class="sys-key">Diarization</span>
            <span class="sys-val">mfcc-cluster</span>
            <div class="sys-status"></div>
        </div>
        <div class="sys-row">
            <span class="sys-key">Translation</span>
            <span class="sys-val">whisper-base</span>
            <div class="sys-status"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    device_label = "CUDA GPU" if DEVICE == "cuda" else "CPU"
    st.markdown(f"""
    <div class="panel-label" style="margin-top:26px;">Runtime</div>
    <div class="sys-row">
        <span class="sys-key">Device</span>
        <span class="sys-val">{device_label}</span>
        <div class="sys-status"></div>
    </div>
    """, unsafe_allow_html=True)


with right_col:
    if analyse_btn and uploaded:
        with st.spinner("Processing…"):
            name   = getattr(uploaded, "name", "recording.wav")
            suffix = os.path.splitext(name)[-1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                res = predict(
                    tmp_path, whisper_proc, whisper_model,
                    ast_extractor, ast_model, sentiment_pipe
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
            finally:
                os.unlink(tmp_path)

        st.markdown('<div class="section-heading">Analysis Output</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Voice Activity", res["speech_label"])
        m2.metric("Language",       res["lang_name"])
        m3.metric("Speech Ratio",   f"{res['speech_ratio']:.1%}")
        m4.metric("Emotion",        res["sentiment"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Transcript + Diarization — only shown when speech was detected
        if res["speech_label"] != "No Speech":
            st.markdown(
                f'<div class="rcard">'
                f'<div class="rcard-label">Transcription · {res["lang_name"]}</div>'
                f'<div class="rcard-value">{res["transcript"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            if res["translation"]:
                st.markdown(
                    f'<div class="rcard">'
                    f'<div class="rcard-label">Translation → English</div>'
                    f'<div class="lang-badge">{res["lang_code"]} → en</div>'
                    f'<div class="rcard-value">{res["translation"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            diar_html  = build_timeline_html(res["diarization_segments"], res["total_duration_sec"])
            spk_count  = res["num_unique_speakers"]
            dur_str    = f"{res['total_duration_sec']:.1f}s"
            spk_plural = "speaker" if spk_count == 1 else "speakers"
            st.markdown(
                f'<div class="rcard">'
                f'<div class="rcard-label">'
                f'Diarization &amp; Timeline · {spk_count} {spk_plural} · {dur_str}'
                f'</div>'
                f'{diar_html}'
                f'</div>',
                unsafe_allow_html=True
            )

        # Sound classification — always shown regardless of speech/non-speech
        rows_html = ""
        for s in res["top_sound_classes"]:
            pct   = s["score"] * 100
            label = s["label"].title()
            rows_html += (
                f'<div class="srow">'
                f'<span class="srow-name">{label}</span>'
                f'<div class="srow-track">'
                f'<div class="srow-fill" style="width:{pct:.1f}%"></div>'
                f'</div>'
                f'<span class="srow-pct">{pct:.1f}%</span>'
                f'</div>'
            )
        st.markdown(
            f'<div class="rcard">'
            f'<div class="rcard-label">Acoustic Classification</div>'
            f'{rows_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Summary — always shown
        st.markdown(
            f'<div class="rcard">'
            f'<div class="rcard-label">Summary</div>'
            f'<div class="rcard-italic">{res["caption"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        with st.expander("Raw JSON"):
            st.json(res)

    elif analyse_btn and not uploaded:
        st.warning("Upload or record audio first.")

    else:
        st.markdown("""
        <div class="await-box">
            <div class="await-rings">
                <svg class="await-icon-inner" viewBox="0 0 24 24" fill="none"
                     stroke="white" stroke-width="1.5">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    <line x1="12" y1="19" x2="12" y2="23"/>
                    <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
            </div>
            <div class="await-waveform">
                <div class="await-bar"></div>
                <div class="await-bar"></div>
                <div class="await-bar"></div>
                <div class="await-bar"></div>
                <div class="await-bar"></div>
                <div class="await-bar"></div>
                <div class="await-bar"></div>
                <div class="await-bar"></div>
            </div>
            <span class="await-label">Upload or record to begin</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Whisper &nbsp;·&nbsp; AST Finetuned &nbsp;·&nbsp; DistilBERT Finetuned
    &nbsp;·&nbsp; MFCC Diarization &nbsp;·&nbsp; UALM vMK.1.0
</div>
""", unsafe_allow_html=True)