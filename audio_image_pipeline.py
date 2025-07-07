import librosa
import librosa.display
import numpy as np
import torch
import matplotlib.pyplot as plt
import tarfile
import os

# to convert audio to spectrograms and save them as images.

# Audio to Mel-Spectrogram Tensor
def audio_to_melspectrogram(audio_path, sr=22050, n_mels=128, hop_length=512):
    if not os.path.isfile(audio_path):
        raise OSError('Audio path does not exist')
    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
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
# tensor = audio_to_melspectrogram('data/nsynth-valid/audio/bass_electronic_018-022-025.wav', 16000, 128, 512)
# print(tensor.shape)
# plt.imshow(tensor.numpy())
# plt.show()
def unpack_jsonwav_archive(archive_path, output_dir):
    """
    Unpack a .jsonwav.tar.gz archive, extracting only .wav and .json files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with tarfile.open(archive_path, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(('.wav', '.json'))]
        tar.extractall(path=output_dir, members=members)
    print(f"Extracted {len(members)} files from {archive_path} to {output_dir}")
def preprocess_to_hdf5(audio_folder, output_file):
    """Convert audio files to spectrograms and store in HDF5 - minimal version"""
    audio_files = find_audio_files(audio_folder)
    
    # Get dimensions from first file
    sample = audio_to_melspectrogram(audio_files[0])
    n_mels, target_frames = sample.shape
    
    with h5py.File(output_file, 'w') as f:
        # Create dataset without compression for fastest access
        spectrograms = f.create_dataset(
            'spectrograms', 
            shape=(len(audio_files), n_mels, target_frames),
            dtype=np.float32
        )
        
        # Process all files at once
        for i, audio_file in enumerate(tqdm(audio_files)):
            S = audio_to_melspectrogram(audio_file)
            S = (S + 80) / 80  # Normalize to [0,1]
            spectrograms[i] = S.numpy()
    
    print(f"Preprocessed {len(audio_files)} files to {output_file}")

# Nur zum testen!!
if __name__ == "__main__":
    # unpack once after cloning
    unpack_jsonwav_archive(
        'data/nsynth-valid.jsonwav.tar.gz',
        'data/'
    )
    # example downstream usage
    tensor = audio_to_melspectrogram(
        'data/nsynth-valid/audio/bass_electronic_018-022-025.wav',
        sr=16000, n_mels=128, hop_length=512
    )
    print(tensor.shape)
    plt.figure(figsize=(10,4))
    plt.imshow(tensor.numpy(), aspect='auto', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
