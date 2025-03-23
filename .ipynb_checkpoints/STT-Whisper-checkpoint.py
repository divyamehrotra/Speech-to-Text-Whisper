{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5ac16c12-6cb1-40a9-a012-ab4dcafe7611",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: torch in c:\\users\\divya mehrotra\\anaconda3\\lib\\site-packages (2.6.0)\n",
      "Requirement already satisfied: transformers in c:\\users\\divya mehrotra\\anaconda3\\lib\\site-packages (4.50.0)\n",
      "Requirement already satisfied: librosa in c:\\users\\divya mehrotra\\anaconda3\\lib\\site-packages (0.11.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR: Could not find a version that satisfies the requirement tkinter (from versions: none)\n",
      "ERROR: No matching distribution found for tkinter\n"
     ]
    }
   ],
   "source": [
    "pip install torch transformers librosa tkinter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "05ee2af4-5744-4153-80e6-73a013a2a895",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tkinter is installed!\n"
     ]
    }
   ],
   "source": [
    "import tkinter\n",
    "print(\"Tkinter is installed!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "id": "a6d0f16e-47c1-4aa1-abb3-dc988123e6f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tkinter as tk\n",
    "from tkinter import filedialog, Text\n",
    "import torch\n",
    "from transformers import WhisperProcessor, WhisperForConditionalGeneration\n",
    "import librosa\n",
    "\n",
    "def load_whisper_model():\n",
    "    model_name = \"openai/whisper-small\"\n",
    "    processor = WhisperProcessor.from_pretrained(model_name)\n",
    "    model = WhisperForConditionalGeneration.from_pretrained(model_name)\n",
    "    model.eval()\n",
    "    return processor, model\n",
    "\n",
    "def transcribe_audio(file_path, processor, model):\n",
    "    audio, sr = librosa.load(file_path, sr=16000)\n",
    "    input_features = processor(audio, sampling_rate=16000, return_tensors=\"pt\").input_features\n",
    "    with torch.no_grad():\n",
    "        predicted_ids = model.generate(input_features)\n",
    "    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]\n",
    "    return transcription\n",
    "\n",
    "def upload_file():\n",
    "    file_path = filedialog.askopenfilename(filetypes=[(\"Audio Files\", \"*.wav;*.mp3;*.flac\")])\n",
    "    if file_path:\n",
    "        transcription_label.config(text=\"Processing...\")\n",
    "        root.update()\n",
    "        transcription = transcribe_audio(file_path, processor, model)\n",
    "        transcription_label.config(text=f\"Transcription: {transcription}\")\n",
    "\n",
    "# Load model and processor\n",
    "processor, model = load_whisper_model()\n",
    "\n",
    "# GUI setup\n",
    "root = tk.Tk()\n",
    "root.title(\"Speech-to-Text Whisper App\")\n",
    "root.geometry(\"500x300\")\n",
    "\n",
    "frame = tk.Frame(root)\n",
    "frame.pack(pady=20)\n",
    "\n",
    "upload_button = tk.Button(frame, text=\"Upload Audio File\", command=upload_file, padx=20, pady=10)\n",
    "upload_button.pack()\n",
    "\n",
    "transcription_label = tk.Label(root, text=\"Upload an audio file to transcribe\", wraplength=450, justify=\"left\")\n",
    "transcription_label.pack(pady=20)\n",
    "\n",
    "root.mainloop()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
