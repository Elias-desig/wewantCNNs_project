import torch
from torch.utils.data import Dataset
from audio_image_pipeline import audio_to_melspectrogram, melspectrogram_to_audio, save_spectrogram_image
import numpy as np
import os
import h5py
# Custom Dataset for audio files
def find_audio_files(folder, exts={'.wav', '.mp3', '.flac'}):
    if not os.path.isdir(folder):
        raise OSError('Audio path does not exist')
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if any(f.lower().endswith(ext) for ext in exts):
                files.append(os.path.join(root, f))
    return files

class SpectogramDataset(Dataset):
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
        self.length = None
        self._file_handle = None
        
        # Get length from main process
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['spectrograms'])
    
    def _get_file_handle(self):
        """Lazy loading of file handle per worker"""
        if self._file_handle is None:
            self._file_handle = h5py.File(self.hdf5_file, 'r')
        return self._file_handle
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        f = self._get_file_handle()
        spec = torch.from_numpy(f['spectrograms'][idx]).float()
        return spec
    
    def __del__(self):
        if self._file_handle is not None:
            self._file_handle.close()


def collate_fn(batch):
    # Pad to max shape in batch
    max_shape = np.max([b.shape[1] for b in batch]) # Gets largest batch (time dimension)
    padded = [torch.nn.functional.pad(b, (0, max_shape-b.shape[1])) for b in batch] # pads rest with zeros to get equal shapes
    return torch.stack(padded)