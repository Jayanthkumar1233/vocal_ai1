import streamlit as st
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import tempfile
from TTS.api import TTS

# Load the LLM from Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# Load the TTS model from Coqui TTS (pre-trained model)
tts_model_name = TTS.list_models()[0]  # You can change this if needed
tts = TTS(tts_model_name)

# Function to rewrite text with tone
def rewrite_with_tone(input_text, tone):
    prompt = f"Rewrite the following sentence in a {tone} tone: {input_text}"
    result = generator(prompt, max_length=100, do_sample=True, temperature=0.9)[0]['generated_text']
    return result.replace(prompt, '').strip()

# Function to convert text to speech
def generate_audio(text, voice="female"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        tts.tts_to_file(text=text, file_path=f.name)
        return f.name

# Streamlit UI
st.set_page_config(page_title="EchoVerse – AI Audiobook Tool")
st.title("📚 EchoVerse – AI-Powered Audiobook Creation")

option = st.radio("Choose Input Method", ["Type Text", "Upload .txt File"])

input_text = ""
if option == "Type Text":
    input_text = st.text_area("✍️ Enter your text here:")
else:
    uploaded_file = st.file_uploader("📁 Upload a .txt file", type="txt")
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")

# Tone selection
tone = st.selectbox("🎭 Choose Tone", ["Neutral", "Suspenseful", "Inspiring"])

# Voice selection
voice = st.selectbox("🔊 Choose Voice", ["female", "male"])

# Action button
if st.button("🎧 Generate Audiobook"):
    if input_text.strip() == "":
        st.warning("Please provide some text input.")
    else:
        with st.spinner("🔁 Rewriting text in selected tone..."):
            rewritten_text = rewrite_with_tone(input_text, tone.lower())
            st.subheader("📝 Rewritten Text")
            st.write(rewritten_text)

        with st.spinner("🎙️ Generating audio..."):
            audio_file = generate_audio(rewritten_text, voice=voice)
            st.audio(audio_file)
            with open(audio_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Audio",
                    data=f,
                    file_name="audiobook.wav",
                    mime="audio/wav"
                )
