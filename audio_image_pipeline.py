import librosa
import librosa.display
import numpy as np
import torch
import matplotlib.pyplot as plt

# to convert audio to spectrograms and save them as images.

# Audio to Mel-Spectrogram Tensor
def audio_to_melspectrogram(audio_path, sr=22050, n_mels=128, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return torch.tensor(S_dB, dtype=torch.float32)

# Mel-Spectrogram Tensor to Audio
def melspectrogram_to_audio(S_dB, sr=22050, n_iter=32, hop_length=512):
    S = librosa.db_to_power(S_dB.numpy())
    y = librosa.feature.inverse.mel_to_audio(S, sr=sr, hop_length=hop_length, n_iter=n_iter)
    return y

# Save spectrogram as image
def save_spectrogram_image(S_dB, out_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Load spectrogram image as tensor (optional, if you want to use images)
def load_spectrogram_image(image_path):
    from PIL import Image
    import torchvision.transforms as transforms
    image = Image.open(image_path).convert('L')
    transform = transforms.ToTensor()
    return transform(image).squeeze(0) 