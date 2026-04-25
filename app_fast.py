import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import torch
import tempfile
import whisper
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment


AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"

st.set_page_config(page_title="VoiceGuard AI", layout="centered", page_icon="🛡️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:      #f0f4ff;
    --card:    #ffffff;
    --border:  rgba(99,102,241,0.18);
    --glow:    rgba(99,102,241,0.45);
    --accent:  #6366f1;
    --adim:    rgba(99,102,241,0.1);
    --danger:  #ef4444;
    --warn:    #f59e0b;
    --t1:      #1e1b4b;
    --t2:      #4b5563;
    --t3:      #9ca3af;
    --mono:    'DM Mono', monospace;
    --sans:    'Syne', sans-serif;
}

html, body, [class*="css"] { font-family: var(--sans) !important; }

.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 15% 0%, rgba(165,180,252,0.4) 0%, transparent 55%),
        radial-gradient(ellipse 60% 45% at 85% 100%, rgba(196,181,253,0.35) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(224,242,254,0.5) 0%, transparent 70%) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 800px !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    backdrop-filter: blur(8px) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--t2) !important;
    border-radius: 7px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: var(--adim) !important;
    color: var(--accent) !important;
    border: 1px solid var(--glow) !important;
}

.stButton > button {
    background: white !important;
    border: 1.5px solid var(--glow) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.1) !important;
}
.stButton > button:hover {
    background: var(--adim) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.25) !important;
}

[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.6) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    color: var(--accent) !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    color: var(--t3) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 100px !important;
}

audio {
    width: 100%;
    border-radius: 8px;
    margin: 0.4rem 0;
}

.stAlert { border-radius: 10px !important; font-family: var(--mono) !important; font-size: 0.82rem !important; }
h1, h2, h3 { color: var(--t1) !important; font-family: var(--sans) !important; }
.stPlotlyChart, .stPyplot { border-radius: 12px; overflow: hidden; }
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ── HERO ──
st.markdown("""
<div style="text-align:center;padding:3rem 1rem 2rem;">
    <div style="display:inline-flex;align-items:center;gap:6px;background:rgba(99,102,241,0.1);
                border:1px solid rgba(99,102,241,0.35);color:#6366f1;font-family:'DM Mono',monospace;
                font-size:11px;letter-spacing:0.12em;text-transform:uppercase;padding:5px 14px;
                border-radius:100px;margin-bottom:1.2rem;">
        <span>&#9679;</span> Wav2Vec2 &middot; Whisper &middot; Deep Learning
    </div>
    <h1 style="font-size:clamp(2rem,5vw,3rem);font-weight:800;margin:0 0 0.5rem;
               background:linear-gradient(135deg,#1e1b4b 20%,#6366f1 60%,#8b5cf6 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        VoiceGuard AI
    </h1>
    <p style="color:#4b5563;font-family:'DM Mono',monospace;font-size:0.95rem;margin:0;">
        Real-time fraud detection &amp; deepfake voice authentication
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
for col, val, lbl in [(c1,"92–95%","Accuracy"), (c2,"&lt;1s","Latency"), (c3,"768D","Embedding")]:
    col.markdown(f"""
    <div style="background:rgba(255,255,255,0.75);border:1px solid rgba(99,102,241,0.18);
                border-radius:12px;padding:1rem;text-align:center;margin-bottom:1.5rem;
                box-shadow:0 4px 16px rgba(99,102,241,0.08);backdrop-filter:blur(8px);">
        <div style="font-size:1.4rem;font-weight:700;color:#6366f1;font-family:'DM Mono',monospace;">{val}</div>
        <div style="font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9ca3af;margin-top:4px;font-family:'DM Mono',monospace;">{lbl}</div>
    </div>""", unsafe_allow_html=True)


# ── LOAD MODELS ──
@st.cache_resource
def load_models():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    wav2vec.eval()
    clf = joblib.load("fast_model.pkl")
    whisper_model = whisper.load_model("base")
    return processor, wav2vec, clf, whisper_model

processor, wav2vec, clf, whisper_model = load_models()


def convert_audio(audio_bytes, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        path = f.name
    wav = path.replace(suffix, ".wav")
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav, format="wav")
    audio_np, sr = librosa.load(wav, sr=16000)
    return audio_np, wav, sr


def predict(audio):
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        hidden = wav2vec(inputs.input_values).last_hidden_state
    feat = hidden.mean(dim=1).numpy()
    prob = clf.predict_proba(feat)[0]
    return prob[1], prob[0]


def fraud(text):
    words = [
        "otp", "one time password", "pin", "cvv", "password",
        "verification code", "security code", "login code",
        "bank", "account", "transaction", "payment", "transfer",
        "upi", "net banking", "balance", "credit card", "debit card",
        "kyc", "update kyc", "account blocked", "account suspended",
        "urgent", "immediately", "right now", "asap",
        "last warning", "final notice", "limited time",
        "blocked", "suspended", "legal action", "police case",
        "penalty", "fine", "fraud alert", "unauthorized access",
        "rbi", "reserve bank", "bank officer", "customer care",
        "technical support", "support team", "verification officer",
        "lottery", "prize", "winner", "cash reward",
        "free money", "bonus", "gift", "jackpot",
        "job offer", "work from home", "easy money",
        "loan approval", "instant loan", "credit approved",
        "click link", "download app", "install app",
        "update app", "verify link", "reset password",
        "trust me", "confidential", "do not share",
        "keep it secret", "only you", "special case",
        "call back", "missed call", "helpline",
        "toll free", "customer support"
    ]
    count = sum(w in text.lower() for w in words)
    return min(count / 5, 1), count


def highlight(text):
    words = [
        "otp", "one time password", "pin", "cvv", "password",
        "verification code", "security code", "login code",
        "bank", "account", "transaction", "payment", "transfer",
        "upi", "net banking", "balance", "credit card", "debit card",
        "kyc", "update kyc", "account blocked", "account suspended",
        "urgent", "immediately", "right now", "asap",
        "last warning", "final notice", "limited time",
        "blocked", "suspended", "legal action", "police case",
        "penalty", "fine", "fraud alert", "unauthorized access",
        "rbi", "reserve bank", "bank officer", "customer care",
        "technical support", "support team", "verification officer",
        "lottery", "prize", "winner", "cash reward",
        "free money", "bonus", "gift", "jackpot",
        "job offer", "work from home", "easy money",
        "loan approval", "instant loan", "credit approved",
        "click link", "download app", "install app",
        "update app", "verify link", "reset password",
        "trust me", "confidential", "do not share",
        "keep it secret", "only you", "special case",
        "call back", "missed call", "helpline",
        "toll free", "customer support"
    ]
    found = []
    for w in words:
        if w in text.lower():
            text = text.replace(w,
                f"<span style='color:#ff4d6d;font-weight:bold;"
                f"background:rgba(255,77,109,0.15);padding:1px 5px;"
                f"border-radius:4px;'>{w.upper()}</span>")
            found.append(w.upper())
    return text, list(set(found))


def show_speedometer(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': "Fraud Risk Level", 'font': {'color': '#4b5563', 'size': 13, 'family': 'DM Mono'}},
        number={'font': {'color': '#1e1b4b', 'family': 'DM Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#9ca3af', 'tickfont': {'color': '#9ca3af'}},
            'bar': {'color': "#6366f1"},
            'bgcolor': '#f8faff',
            'bordercolor': '#e0e7ff',
            'steps': [
                {'range': [0, 40],   'color': "rgba(99,102,241,0.08)"},
                {'range': [40, 75],  'color': "rgba(245,158,11,0.1)"},
                {'range': [75, 100], 'color': "rgba(239,68,68,0.1)"},
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='#f8faff', plot_bgcolor='#f8faff',
        font={'color': '#4b5563'},
        margin=dict(t=40, b=20, l=20, r=20), height=260,
    )
    st.plotly_chart(fig, use_container_width=True)


def show_spectrogram(audio, sr):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#f8faff')
    ax.set_facecolor('#f8faff')
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='viridis', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram", color='#4b5563', fontsize=9, fontfamily='monospace')
    ax.tick_params(colors='#9ca3af')
    for spine in ax.spines.values():
        spine.set_edgecolor('#e0e7ff')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def section_label(label):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:1.8rem 0 0.8rem;">
        <span style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.15em;
                     text-transform:uppercase;color:#9ca3af;white-space:nowrap;">{label}</span>
        <div style="flex:1;height:1px;background:rgba(99,102,241,0.15);"></div>
    </div>""", unsafe_allow_html=True)


def card_top(title):
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.85);border:1px solid rgba(99,102,241,0.18);border-radius:18px;
                padding:1.6rem;margin-bottom:0.5rem;position:relative;overflow:hidden;
                box-shadow:0 4px 24px rgba(99,102,241,0.08);backdrop-filter:blur(12px);">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;
                    background:linear-gradient(90deg,transparent,rgba(99,102,241,0.5),transparent);"></div>
        <div style="display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;
                    font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:#4b5563;margin-bottom:1rem;">
            <div style="width:6px;height:6px;border-radius:50%;background:#6366f1;
                        box-shadow:0 0 8px rgba(99,102,241,0.5);"></div>
            {title}
        </div>
    """, unsafe_allow_html=True)


def process(audio_bytes, suffix):
    st.audio(audio_bytes)

    audio_np, path, sr = convert_audio(audio_bytes, suffix)

    section_label("📈 Spectrogram")
    show_spectrogram(audio_np, sr)

    fake, real = predict(audio_np)
    text = whisper_model.transcribe(path)["text"]
    risk_text, keyword_count = fraud(text)
    final = fake * 0.6 + risk_text * 0.4

    section_label("📝 Transcript")
    highlighted_text, words = highlight(text)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.8);border:1px solid rgba(99,102,241,0.15);
                border-radius:10px;padding:1.1rem 1.3rem;font-family:'DM Mono',monospace;
                font-size:0.85rem;color:#4b5563;line-height:1.9;
                box-shadow:0 2px 12px rgba(99,102,241,0.06);">
        {highlighted_text}
    </div>""", unsafe_allow_html=True)

    if words:
        st.warning(f"⚠️ Suspicious keywords: {', '.join(words)}")

    section_label("📊 Analysis")
    col1, col2 = st.columns(2)
    col1.metric("🔴 Suspicious Words", keyword_count)
    col2.metric("⚠️ Risk Score", f"{final * 100:.1f}%")

    if final > 0.75:
        icon, label, sub = "🚨", "HIGH FRAUD RISK DETECTED", f"Risk Score: {final*100:.1f}% · Likely fraudulent or AI-generated"
        color, bg, border = "#ff4d6d", "rgba(255,77,109,0.1)", "rgba(255,77,109,0.3)"
    elif final > 0.45:
        icon, label, sub = "⚠", "SUSPICIOUS — PROCEED WITH CAUTION", f"Risk Score: {final*100:.1f}% · Potentially manipulated content"
        color, bg, border = "#f9ca24", "rgba(249,202,36,0.1)", "rgba(249,202,36,0.3)"
    else:
        icon, label, sub = "✓", "AUDIO APPEARS SAFE", f"Risk Score: {final*100:.1f}% · No significant fraud indicators"
        color, bg, border = "#00dcb4", "rgba(0,220,180,0.1)", "rgba(0,220,180,0.3)"

    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:12px;
                padding:1.2rem 1.5rem;display:flex;align-items:center;gap:14px;margin:1rem 0;">
        <div style="font-size:1.8rem;line-height:1;">{icon}</div>
        <div>
            <strong style="display:block;font-size:1rem;font-weight:700;color:#f0f4ff;">{label}</strong>
            <span style="font-size:0.78rem;font-family:'DM Mono',monospace;color:#8b95b0;">{sub}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.progress(int(final * 100))

    section_label("📊 Fraud Risk Meter")
    show_speedometer(final)


# ── TABS ──
tab1, tab2 = st.tabs(["🎤  Record", "📂  Upload"])

with tab1:
    card_top("Microphone Input")
    audio = mic_recorder(start_prompt="▶ Start Recording", stop_prompt="⏹ Stop & Analyze")
    st.markdown("</div>", unsafe_allow_html=True)
    if audio:
        process(audio["bytes"], ".webm")

with tab2:
    card_top("File Analysis")
    file = st.file_uploader("Drop a WAV, MP3, M4A or WEBM file", type=["wav", "mp3", "m4a", "webm"])
    st.markdown("</div>", unsafe_allow_html=True)
    if file:
        process(file.read(), "." + file.name.split(".")[-1])


# ── AWARENESS ──
section_label("🛡️ Cybercrime Awareness")

st.markdown("""
<div style="background:rgba(255,255,255,0.85);border:1px solid rgba(99,102,241,0.18);border-radius:18px;padding:1.8rem;
            box-shadow:0 4px 24px rgba(99,102,241,0.08);backdrop-filter:blur(12px);">
    <div style="margin-bottom:1.2rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#6366f1;margin-bottom:0.3rem;">&#127470;&#127475; India Scams</div>
        <p style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#4b5563;line-height:1.7;margin:0;">OTP &amp; bank fraud calls &middot; Fake police/RBI calls &middot; Loan &amp; job scams</p>
    </div>
    <div style="margin-bottom:1.2rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#6366f1;margin-bottom:0.3rem;">&#127758; Global Scams</div>
        <p style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#4b5563;line-height:1.7;margin:0;">AI voice cloning &middot; Deepfake impersonation &middot; Crypto scams</p>
    </div>
    <div style="margin-bottom:1.2rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#6366f1;margin-bottom:0.3rem;">&#9888;&#65039; Warning Signs</div>
        <p style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#4b5563;line-height:1.7;margin:0;">Urgency in communication &middot; Asking for OTP or passwords &middot; Unknown callers claiming authority</p>
    </div>
    <div style="margin-bottom:1.2rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#6366f1;margin-bottom:0.3rem;">&#9989; Safety Tips</div>
        <p style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#4b5563;line-height:1.7;margin:0;">Never share your OTP &middot; Always verify via official sources &middot; Report suspected fraud immediately</p>
    </div>
    <div style="margin-bottom:1.2rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#6366f1;margin-bottom:0.3rem;">&#128222; Emergency Contacts</div>
        <p style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#4b5563;margin:0;">
            India Helpline: <strong style="color:#6366f1;">1930</strong> &nbsp;&middot;&nbsp;
            <a href="https://cybercrime.gov.in" target="_blank" style="color:#6366f1;text-decoration:none;">cybercrime.gov.in</a>
        </p>
    </div>
    <div style="margin-bottom:1.2rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#6366f1;margin-bottom:0.3rem;">&#128737; Government Resources</div>
        <p style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#4b5563;margin:0;line-height:2.2;">
            <a href="https://cybercrime.gov.in" target="_blank" style="color:#6366f1;text-decoration:none;">cybercrime.gov.in</a> &nbsp;&middot;&nbsp;
            <a href="https://www.cert-in.org.in" target="_blank" style="color:#6366f1;text-decoration:none;">cert-in.org.in</a> &nbsp;&middot;&nbsp;
            <a href="https://digitalindia.gov.in" target="_blank" style="color:#6366f1;text-decoration:none;">digitalindia.gov.in</a> &nbsp;&middot;&nbsp;
            <a href="https://www.meity.gov.in" target="_blank" style="color:#6366f1;text-decoration:none;">meity.gov.in</a> &nbsp;&middot;&nbsp;
            <a href="https://www.interpol.int" target="_blank" style="color:#6366f1;text-decoration:none;">interpol.int</a> &nbsp;&middot;&nbsp;
            <a href="https://www.europol.europa.eu" target="_blank" style="color:#6366f1;text-decoration:none;">europol.europa.eu</a>
        </p>
    </div>
    <div>
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#6366f1;margin-bottom:0.3rem;">&#128274; Cyber Security Resources</div>
        <p style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#4b5563;margin:0;line-height:2.2;">
            <a href="https://haveibeenpwned.com" target="_blank" style="color:#6366f1;text-decoration:none;">haveibeenpwned.com</a> &nbsp;&middot;&nbsp;
            <a href="https://owasp.org" target="_blank" style="color:#6366f1;text-decoration:none;">owasp.org</a> &nbsp;&middot;&nbsp;
            <a href="https://staysafeonline.org" target="_blank" style="color:#6366f1;text-decoration:none;">staysafeonline.org</a> &nbsp;&middot;&nbsp;
            <a href="https://www.cisa.gov" target="_blank" style="color:#6366f1;text-decoration:none;">cisa.gov</a>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── FOOTER ──
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;border-top:1px solid rgba(99,102,241,0.12);">
    <p style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#9ca3af;letter-spacing:0.12em;">
        VOICEGUARD AI &middot; WAV2VEC2-BASE &middot; WHISPER &middot; FRAUD DETECTION ENGINE
    </p>
</div>
""", unsafe_allow_html=True)