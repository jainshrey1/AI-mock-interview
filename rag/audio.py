import speech_recognition as sr
import tempfile
import openai 
import requests
import os
import streamlit as st


def transcribe_audio(file_path):
    """Transcribes the given audio file using OpenAI's Whisper API."""
    api_key = os.getenv("OPENAI_API_KEY")  # Ensure API key is set

    with open(file_path, "rb") as audio_file:
        files = {"file": audio_file}
        data = {"model": "whisper-1"}  # Specify the Whisper model
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data
        )

    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        st.error(f"Error transcribing audio: {response.text}")
        return ""

def record_audio():
    """Records user voice input and returns transcribed text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        audio = recognizer.listen(source, timeout=5)
        st.success("Processing...")

    # Save the audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio.get_wav_data())
        temp_audio_path = temp_audio.name

    try:
        # Transcribe the recorded audio
        text = transcribe_audio(temp_audio_path)
        return text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""
