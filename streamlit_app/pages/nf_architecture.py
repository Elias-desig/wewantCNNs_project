import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import torch
import matplotlib.pyplot as plt
from core.inference import load_nf_model, sample_from_flow
from scipy.io import wavfile
import numpy as np
import io

# Gerät festlegen
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model, input_dim = load_nf_model(device)
    return model, input_dim

model, input_dim = load_model()

st.title("Normalizing Flow Sample Generator")

def sample_to_audio_tensor(sample):
    audio = sample.cpu().numpy().flatten()
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    return audio

if st.button("Generate new sample"):
    sample = sample_from_flow(model, input_dim, device, batch_size=1)
    spec = sample.view(128, 172).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spec, aspect='auto', cmap='magma', origin='lower')
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
    ax.set_title("Generated Spectrogram Sample")
    st.pyplot(fig)
    audio_data = sample_to_audio_tensor(sample[0])

    sample_rate = 16000
    wav_io = io.BytesIO()
    wavfile.write(wav_io, sample_rate, audio_data)
    wav_io.seek(0)

    st.audio(wav_io, format='audio/wav')

    # Download-Button
    st.download_button(
        label="Download generated audio",
        data=wav_io,
        file_name="generated_sample.wav",
        mime="audio/wav"
    )