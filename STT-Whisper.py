import tkinter as tk
from tkinter import filedialog
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import threading  # To prevent GUI from freezing in Jupyter
from IPython.display import display

# Function to load the Whisper model
def load_whisper_model():
    print("Loading Whisper model...")
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully!")
    return processor, model

# Function to transcribe audio
def transcribe_audio(file_path, processor, model):
    print(f"Transcribing file: {file_path}")
    audio, sr = librosa.load(file_path, sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"Transcription: {transcription}")
    return transcription

# Function to upload and transcribe file
def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.flac")])
    if file_path:
        transcription_label.config(text="Processing...")
        root.update()
        transcription = transcribe_audio(file_path, processor, model)
        transcription_label.config(text=f"Transcription: {transcription}")

# Run the Tkinter GUI in a separate thread
def run_gui():
    global root, transcription_label, processor, model

    # Load model
    processor, model = load_whisper_model()

    # GUI setup
    root = tk.Tk()
    root.title("Speech-to-Text Whisper App")
    root.geometry("500x300")

    frame = tk.Frame(root)
    frame.pack(pady=20)

    upload_button = tk.Button(frame, text="Upload Audio File", command=upload_file, padx=20, pady=10)
    upload_button.pack()

    transcription_label = tk.Label(root, text="Upload an audio file to transcribe", wraplength=450, justify="left")
    transcription_label.pack(pady=20)

    print("Running GUI...")
    root.mainloop()

# Run GUI in a separate thread so Jupyter doesn't freeze
threading.Thread(target=run_gui, daemon=True).start()

# Display a message in Jupyter Notebook
display("GUI started! You can interact with the window.")
