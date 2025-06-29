import os
import torch
from torch.utils.data import Dataset, DataLoader
from VAE_models import VAE
from audio_image_pipeline import audio_to_melspectrogram, melspectrogram_to_audio, save_spectrogram_image
import soundfile as sf
import numpy as np

# to train your VAE on spectrograms.

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

def collate_fn(batch):
    # Pad to max shape in batch
    max_shape = np.max([b.shape[1] for b in batch]) # Gets largest batch (time dimension)
    padded = [torch.nn.functional.pad(b, (0, max_shape-b.shape[1])) for b in batch] # pads rest with zeros to get equal shapes
    return torch.stack(padded)

# Training loop
def train_vae(audio_folder, epochs=10, batch_size=8, latent_dim=32, n_layers=3, lr=1e-3, device='cpu'):
    dataset = SpectrogramDataset(audio_folder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    sample_spec = dataset[0] # Sample spectogram 
    in_dim = sample_spec.numel() # For spectograms = width * height

    vae = VAE(in_dim, latent_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    for epoch in range(epochs):
        vae.train()
        
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            batch_flat = batch.view(batch.size(0), -1) # (B, n_mels * T)
            
            output = vae(batch_flat)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
if __name__ == '__main__':
    # Example usage: train_vae('data/audio_folder')
    pass 