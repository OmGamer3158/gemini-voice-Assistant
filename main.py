import queue
import sounddevice as sd
import vosk
import pyttsx3  # For text-to-speech
import json
import speech_recognition as sr
import google.generativeai as genai
import os
import time
import threading
from faster_whisper import WhisperModel

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to stop speaking immediately
def stop_speaking():
    engine.stop()

# Global flag to track if the assistant should stop speaking
stop_signal = False

# Function to speak the text in a separate thread
def speak(text):
    global stop_signal
    stop_signal = False
    engine.say(text)
    engine.runAndWait()

# Global variables for Vosk and audio handling
q = queue.Queue()
vosk_model = vosk.Model(r"C:\Users\91635\OneDrive\Desktop\my time pass Work\GEMINI VOICE ASS\-vosk-model-small-en-us-0.15")  # You need to download a Vosk model
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

wake_word = "hey"
listen_for_wake_words = True

# Google Generative AI setup
GOOGLE_API_KEY = 'your api'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro-latest')
convo = model.start_chat()

whisper_size = 'base'
num_coures = os.cpu_count()
WhisperModel = WhisperModel(
    model_size_or_path=whisper_size,  # Use whisper_size here
    device='cpu',
    compute_type='int8',
    cpu_threads=num_coures,
    num_workers=num_coures
)

# Generation and safety settings
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 2,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

modal = genai.GenerativeModel('gemini-1.0-pro-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat()

# Vosk-based audio callback function
def vosk_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# Real-time speech recognition using Vosk
def real_time_recognition():
    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get('text', '')
            if text:
                print("Recognized:", text)
                return text
        else:
            print("Partial:", recognizer.PartialResult())

# Whisper-based transcription
def wav_to_text(audio_path):
    segments, _ = WhisperModel.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

# Function to handle Vosk speech recognition and GPT prompt
def listen_and_respond():
    global stop_signal
    text_input = real_time_recognition()
    if wake_word in text_input.lower().strip():
        print('Wake word detected. Please speak your prompt to Yantra.')
        prompt_text = real_time_recognition()

        if "stop" in prompt_text.lower():
            # Stop speaking if "stop" is detected
            stop_speaking()
            response = "I have stopped. Do you have any other questions?"
            print('Yantra:', response)
            speak(response)
            return  # Stop the current flow and ask for new input

        if "what is your name" in prompt_text.lower():
            response = "My name is Yantra, made by Chirayu."
            print('Yantra:', response)
            speak(response)
        else:
            print('User:', prompt_text)
            convo.send_message(prompt_text)
            output = convo.last.text
            print('Yantra:', output)

            # Run the speaking process in a new thread
            speak_thread = threading.Thread(target=speak, args=(output,))
            speak_thread.start()

            # Listen for the "stop" command during speech
            while speak_thread.is_alive():
                interrupt_input = real_time_recognition()
                if "stop" in interrupt_input.lower():
                    stop_signal = True
                    stop_speaking()
                    speak_thread.join()  # Wait for the thread to finish
                    print('Yantra: I have stopped. Do you have any other questions?')
                    speak("I have stopped. Do you have any other questions?")
                    break

# Start the sounddevice stream for continuous listening
def start_listening_vosk():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=vosk_callback):
        print("Listening... Say 'hey' to wake up.")
        while True:
            listen_and_respond()
            time.sleep(0.5)

if __name__ == '__main__':
    start_listening_vosk()
