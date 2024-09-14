# import streamlit as st
# import pyaudio
# import wave
# import speech_recognition as sr
# import tempfile
# import os
# import openai
# import requests

# openai.api_key = "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"
# base_url = "https://integrate.api.nvidia.com/v1"

# # Audio recording parameters
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100
# CHUNK = 1024
# # Function to send text to OpenAI API
# def get_response_from_openai(input_text):
#     prompt = f"Assume you are a candidate in a job interview. Answer the following question in single paragraph as if you were the candidate: '{input_text}'"   
#     response = requests.post(
#         f"{base_url}/chat/completions",
#         headers={"Authorization": f"Bearer {openai.api_key}"},
#         json={
#             "model": "meta/llama-3.1-405b-instruct",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.2,
#             "top_p": 0.7,
#             "max_tokens": 1024,
#             "stream": False
#         }
#     )

#     if response.status_code == 200:
#         response_data = response.json()
#         return response_data['choices'][0]['message']['content']
#     else:
#         st.error(f"Failed to get response from OpenAI API. Status code: {response.status_code}")
#         return None
# def record_audio(duration, temp_file_path):
#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#     frames = []

#     st.write("Recording...")

#     for _ in range(0, int(RATE / CHUNK * duration)):
#         data = stream.read(CHUNK)
#         frames.append(data)

#     st.write("Finished recording.")
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     # Save the recorded data to a temporary file
#     with wave.open(temp_file_path, 'wb') as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(audio.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))

# def convert_audio_to_text(audio_file_path):
#     recognizer = sr.Recognizer()
#     try:
#         with sr.AudioFile(audio_file_path) as source:
#             audio_data = recognizer.record(source)
#             try:
#                 text = recognizer.recognize_google(audio_data)
#                 return text
#             except sr.UnknownValueError:
#                 return "Sorry, I could not understand the audio."
#             except sr.RequestError:
#                 return "Sorry, there was an error with the speech recognition service."
#     except ValueError as e:
#         st.error(f"Error reading audio file: {e}")
#         return "Error processing audio file."

# def start_recording(duration):
#     # Create a temporary file for recording
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     temp_file_path = temp_file.name
#     temp_file.close()

#     # Record audio
#     record_audio(duration, temp_file_path)

#     # Save the file path in session state
#     st.session_state.temp_file_path = temp_file_path

# def stop_recording():
#     # Display the recorded audio and convert to text
#     if os.path.exists(st.session_state.temp_file_path):
#         with open(st.session_state.temp_file_path, "rb") as audio_file:
#             st.audio(audio_file, format='audio/wav')
            
#             # Convert and display text
#             text = convert_audio_to_text(st.session_state.temp_file_path)
#             st.write("Converted Text:")
#             st.write(text)
            
#             ai_response = get_response_from_openai(text)
#             st.write('Response')
#             st.write(ai_response)

# def delete_audio_file():
#     # Delete the temporary file
#     if 'temp_file_path' in st.session_state:
#         if os.path.exists(st.session_state.temp_file_path):
#             os.remove(st.session_state.temp_file_path)
#             st.write("Audio file deleted.")
#         else:
#             st.write("No audio file found to delete.")
#         # Clear the temp_file_path from session state
#         del st.session_state.temp_file_path

# st.title("Audio Recorder and Transcriber")

# # Input for recording duration
# duration = st.slider("Select recording duration (seconds)", min_value=1, max_value=30, value=10)

# if st.button("Start Recording"):
#     start_recording(duration)

# if st.button("Stop Recording"):
#     stop_recording()

# if st.button("Delete Audio File"):
#     delete_audio_file()

import openai
import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import speech_recognition as sr
import tempfile
import os
import requests

openai.api_key = "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"
base_url = "https://integrate.api.nvidia.com/v1"

# Audio recording parameters
FORMAT = 'int16'  # Use 'int16' format for WAV files
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Function to send text to OpenAI API
def get_response_from_openai(input_text):
    prompt = f"Assume you are a candidate in a job interview. Answer the following question in single paragraph as if you were the candidate: '{input_text}'"   
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {openai.api_key}"},
        json={
            "model": "meta/llama-3.1-405b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
    )

    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    else:
        st.error(f"Failed to get response from OpenAI API. Status code: {response.status_code}")
        return None
    
    
def record_audio(duration, temp_file_path):
    st.write("Recording...")
    
    # Record audio
    recording = sd.rec(int(duration * RATE), samplerate=RATE, channels=CHANNELS, dtype=FORMAT)
    sd.wait()  # Wait until recording is finished

    st.write("Finished recording.")

    # Save the recorded data to a temporary file
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.iinfo(np.int16).bits // 8)  # Sample width in bytes
        wf.setframerate(RATE)
        wf.writeframes(recording.tobytes())



def convert_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return "Sorry, I could not understand the audio."
            except sr.RequestError:
                return "Sorry, there was an error with the speech recognition service."
    except ValueError as e:
        st.error(f"Error reading audio file: {e}")
        return "Error processing audio file."

def start_recording(duration):
    # Create a temporary file for recording
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file_path = temp_file.name
    temp_file.close()

    # Record audio
    record_audio(duration, temp_file_path)

    # Save the file path in session state
    st.session_state.temp_file_path = temp_file_path

def stop_recording():
    # Display the recorded audio and convert to text
    if os.path.exists(st.session_state.temp_file_path):
        with open(st.session_state.temp_file_path, "rb") as audio_file:
            st.audio(audio_file, format='audio/wav')
            
            # Convert and display text
            text = convert_audio_to_text(st.session_state.temp_file_path)
            st.write("Converted Text:")
            st.write(text)
            
            # Get AI response and display
            ai_response = get_response_from_openai(text)
            if ai_response:
                st.write('Response from AI:')
                st.write(ai_response)
            else:
                st.write("Failed to get a response from AI.")

def delete_audio_file():
    # Delete the temporary file
    if 'temp_file_path' in st.session_state:
        if os.path.exists(st.session_state.temp_file_path):
            os.remove(st.session_state.temp_file_path)
            st.write("Audio file deleted.")
        else:
            st.write("No audio file found to delete.")
        # Clear the temp_file_path from session state
        del st.session_state.temp_file_path

st.title("Audio Recorder and Transcriber")

# Input for recording duration
duration = st.slider("Select recording duration (seconds)", min_value=1, max_value=30, value=10)

if st.button("Start Recording"):
    start_recording(duration)

if st.button("Stop Recording"):
    stop_recording()

if st.button("Delete Audio File"):
    delete_audio_file()
