import streamlit as st
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import tempfile
from gtts import gTTS
from pydub import AudioSegment

# Load a lighter model that doesn't require sentencepiece
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# [Rest of your existing code remains exactly the same...]
