# Speech-to-Text Whisper App 🎤📝

A GUI-based Speech-to-Text application that utilizes OpenAI's Whisper model to transcribe audio files into text. Built with Python (Tkinter, Torch, Transformers, Librosa).

## Features

✅ Upload and transcribe audio files (WAV, MP3, FLAC)
✅ Uses OpenAI Whisper model for accurate transcription
✅ Simple Tkinter GUI for easy interaction

## Installation & Setup

### Prerequisites

* Python 3.8+ (or Anaconda Distribution, as shown in the provided code)
* Torch
* Transformers
* Librosa
* Tkinter (usually included with Python, but may require separate installation on some systems)

### Install Dependencies

To install the required Python packages, follow these steps:

1.  **Open your terminal or command prompt.**
2.  **Ensure you have Python and pip installed.**
3.  **Run the following command:**

    ```bash
    pip install torch transformers librosa
    ```

    **Note:** The original code attempts to install `tkinter` via pip, which is typically incorrect on most systems. Tkinter is usually part of the standard library.

## Run the Application

1.  **Save the provided Python code as `app.py`.**
2.  **Navigate to the directory containing `app.py` in your terminal or command prompt.**
3.  **Execute the following command:**

    ```bash
    python app.py
    ```

## Usage

1.  **The application window will open with an "Upload Audio File" button.**
2.  **Click the "Upload Audio File" button.**
3.  **Select an MP3, WAV, or FLAC audio file from your computer.**
4.  **The application will process the audio, and the transcribed text will be displayed below the button.**

## Troubleshooting

### Tkinter Issues

* Tkinter is usually included with Python, but if you encounter issues, it might require a separate installation on some systems.
* **Linux:**

    ```bash
    sudo apt install python3-tk
    ```

* **macOS:** Tkinter should be included with the standard Python installation. If you used Homebrew to install python you may also need to install tcl-tk:

    ```bash
    brew install tcl-tk
    ```

* **Windows:** Tkinter is typically included with Python on Windows. If you are using Anaconda, it should also be included. If you encounter issues, ensure your Python installation is correct.

### Audio Processing Issues

* If you encounter errors related to audio processing, ensure that your audio file is in a supported format (WAV, MP3, FLAC) and that the file is not corrupted.
* If you encounter librosa related errors, try reinstalling it.
* The code assumes the audio sampling rate is 16000. Ensure your audio is compatible, or modify the code to handle different sampling rates.

## Notes

* The code uses the "openai/whisper-small" model. For better accuracy, you can try using larger models (e.g., "openai/whisper-medium" or "openai/whisper-large"), but they require more computational resources. Change the `model_name` variable in the `load_whisper_model()` function to use a different model.
* This application requires a working internet connection when first loading the whisper model, as it downloads the model from huggingface.

## License

This project is open-source.