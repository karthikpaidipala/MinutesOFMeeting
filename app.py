import streamlit as st
from transformers import pipeline
import whisper
import tempfile
import os

# -----------------------------
# Load Whisper Model for Audio/Video
# -----------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# -----------------------------
# Hugging Face Summarizer
# -----------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# -----------------------------
# Process Transcript (Text Input)
# -----------------------------
def summarize_text(text, summarizer):
    # Hugging Face BART only handles ~1024 tokens → split text
    max_chunk = 1000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Minutes of Meeting (MoM) Generator – Free Version")

st.sidebar.title("Upload Options")
upload_type = st.sidebar.radio("Choose Input Type:", ["Transcript (.txt)", "Audio/Video"])

summarizer = load_summarizer()
whisper_model = load_whisper()

transcript_text = ""

# --- Case 1: Transcript File Upload ---
if upload_type == "Transcript (.txt)":
    uploaded_file = st.file_uploader("Upload Transcript (.txt)", type=["txt"])
    if uploaded_file:
        transcript_text = uploaded_file.read().decode("utf-8")
        st.subheader("Transcript")
        st.write(transcript_text)

# --- Case 2: Audio / Video Upload ---
else:
    uploaded_file = st.file_uploader("Upload Audio/Video", type=["mp3", "wav", "m4a", "mp4"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.info("Transcribing audio/video... ⏳")
        result = whisper_model.transcribe(tmp_path)
        transcript_text = result["text"]

        st.subheader("Transcript")
        st.write(transcript_text)

# --- Generate MoM ---
if transcript_text:
    if st.button("Generate Minutes of Meeting"):
        st.info("Summarizing transcript... ⏳")
        mom = summarize_text(transcript_text, summarizer)

        st.subheader("📝 Minutes of Meeting")
        st.success(mom)
