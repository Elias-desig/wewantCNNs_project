import torch
from torch.utils.data import Dataset
from audio_image_pipeline import audio_to_melspectrogram, melspectrogram_to_audio, save_spectrogram_image
import numpy as np
import os
import h5py
# Custom Dataset for audio files
def find_audio_files(folder, exts={'.wav', '.mp3', '.flac'}):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if any(f.lower().endswith(ext) for ext in exts):
                files.append(os.path.join(root, f))
    return files

class SpectrogramDataset(Dataset):
    def __init__(self, audio_folder):
        self.audio_files = find_audio_files(audio_folder)
    def __len__(self):
        return len(self.audio_files)
    def __getitem__(self, idx):
        S = audio_to_melspectrogram(self.audio_files[idx])
        S = (S + 80) / 80  # Normalize dB to [0,1] (assuming min -80dB)
        return S
class HDF5SpectrogramDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        # Open file handle that will be reused
        self.f = h5py.File(hdf5_file, 'r')
        self.spectrograms = self.f['spectrograms']
        self.length = len(self.spectrograms)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # HDF5 supports random access - this is very fast!
        spec = torch.from_numpy(self.spectrograms[idx]).float()
        return spec
    
    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()


def collate_fn(batch):
    # Pad to max shape in batch
    max_shape = np.max([b.shape[1] for b in batch]) # Gets largest batch (time dimension)
    padded = [torch.nn.functional.pad(b, (0, max_shape-b.shape[1])) for b in batch] # pads rest with zeros to get equal shapes
    return torch.stack(padded)