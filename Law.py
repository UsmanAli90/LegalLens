import json
import spacy
import whisper
import streamlit as st
import tempfile
import os

whisper_model = whisper.load_model("base")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


try:
    with open("pakistan_laws.json", "r", encoding="utf-8") as f:
        legal_data = json.load(f)
except FileNotFoundError:
    legal_data = []
    st.warning("Warning: 'pakistan_laws.json' not found. Ensure the file exists.")

def extract_keywords(text):
    doc = nlp(text)
    keywords = set()

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]:  
            keywords.add(token.lemma_.lower()) 
            keywords.add(token.text.lower())  
    
    return list(keywords)

def find_relevant_laws(keywords):
    matched_laws = []
    
    for law in legal_data:
        law_text = law.get("description", "").lower()
        for keyword in keywords:
            if keyword in law_text:
                matched_laws.append({
                    "law_name": law.get("law_name", "Unknown Law"),
                    "section": law.get("section", "N/A"),
                    "article": law.get("article", "N/A"),
                    "description": law.get("description", "No description available.")
                })
                break  
    
    return matched_laws

st.title("Legal Advisor AI - Audio to Law Finder")
st.write("Upload an audio file, and the AI will transcribe it, extract keywords, and find relevant laws.")

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    st.write("**Transcribing Audio...** ⏳")
    result = whisper_model.transcribe(temp_audio_path)
    transcribed_text = result.get("text", "")

    keywords = extract_keywords(transcribed_text)

    matched_laws = find_relevant_laws(keywords)

    st.subheader("Transcribed Text")
    st.write(transcribed_text)

    st.subheader("Extracted Keywords")
    st.write(", ".join(keywords))

    st.subheader("⚖️ Matched Laws")
    if matched_laws:
        for law in matched_laws:
            st.write(f"** {law['law_name']}**")
            st.write(f"**Section:** {law['section']} | **Article:** {law['article']}")
            st.write(f" {law['description']}")
            st.write("---")
    else:
        st.write(" No relevant laws found.")

    os.remove(temp_audio_path)
