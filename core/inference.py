from pathlib import Path
from .VAE_models import CVAE, VAE, CVAE_Deep
from .nf_model import MLP_Masked
from .audio_image_pipeline import audio_to_melspectrogram, melspectrogram_to_audio
import torch


def load_model(checkpoint_path, device, model_type):
    """Load VAE model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == 'CVAE':
        model = CVAE(
            checkpoint['config']['image_size'],
            checkpoint['config']['latent_dim'],
        ).to(device)
    elif model_type == 'VAE':
        model = VAE(
            checkpoint['config']['in_dim'],
            checkpoint['config']['latent_dim'],
            checkpoint['config']['n_layers']
        ).to(device)
    elif model_type == 'CVAE_Deep':
        model = CVAE_Deep(
            checkpoint['config']['image_size'],
            checkpoint['config']['latent_dim'],
        ).to(device)        
    elif model_type == 'NF':
        model = MLP_Masked(
            checkpoint['config']['input_dim'],
            checkpoint['config']['hidden_dims'],
        ).to(device)
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint


def reconstruction(model, model_type:str, sample_path:str, conv:bool):

    sample = audio_to_melspectrogram(sample_path, even=True)
    original_shape = sample.shape
    # normalize
    min = sample.min()
    max = sample.max()
    sample = (sample - min) / (max - min + 1e-8)

    device = next(model.parameters()).device
    sample = sample.to(device)
    model.eval()

    with torch.no_grad():
        if model_type in ['VAE', 'CVAE', 'CVAE_Deep']:
            if conv:
                sample = sample.unsqueeze(0).unsqueeze(0)
            if not conv:
                
                sample = sample.view(1, -1)
            outputs = model(sample, compute_loss=False)
            recon = outputs.x_recon.cpu()
            z = outputs.z_sample.cpu()
            # Reshape the output 
            if conv:
                recon = recon.squeeze(0).squeeze(0)  # [1, 1, H, W] -> [H, W]
            else:
                recon = recon.view(original_shape)  # [1, H*W] -> [H, W]
            # denormalize
            recon = recon * (max - min) + min

        elif model_type == 'NF':
            pass
        else:
            raise ValueError(f'Provide valid model type!{model_type}')

    recon = torch.clamp(recon, -80.0, 0.0)
    audio = melspectrogram_to_audio(recon)
    return audio, recon, z


def generate(model, model_type: str, latent_sample, output_dim: tuple[str], conv: bool):
    model.eval()
    with torch.no_grad():
        if model_type == 'VAE':
            images = model.decode(latent_sample)
        if not conv:
            images = images.view(images.size(0), output_dim)
        elif model_type == 'NF':
            pass
        else:
            raise NameError('Provide valid model type!')
    return images


def load_nf_model(device):
    checkpoint_path = "/Users/koraygecimli/PycharmProjects/UDL_demo/wewantCNNs_project/models/nf_checkpoint_20250731-021809.pt" #absolute path
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = checkpoint['config']['input_dim']
    hidden_dims = checkpoint['config']['hidden_dims']

    model = MLP_Masked(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, input_dim


def sample_from_flow(model, input_dim, device, batch_size=1):
    z = torch.randn(batch_size, input_dim, device=device)
    with torch.no_grad():
        x_sample = model.inverse(z)
    return x_sample



def select_model(model_type, device=None):
    base_dir = Path(__file__).parent.parent
    checkpoint_dir = base_dir / 'models' / 'inference'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patterns = {
        'VAE':  'vae_checkpoint_*.pt',
        'CVAE': 'c_vae_checkpoint_*.pt',
        'NF':   'nf_checkpoint_*.pt',
        'CVAE_Deep': 'deep_c_vae_checkpoint_*.pt'
    }
    try:
        pat = patterns[model_type]
    except KeyError:
        raise ValueError(f"Unknown model_type {model_type!r}")

    matches = sorted(checkpoint_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No checkpoint file matching `{pat}` in {checkpoint_dir}")

    checkpoint_path = matches[0]
    print(f'loading model from path: {str(checkpoint_path)}, model type: {model_type}')
    model, checkpoint = load_model(str(checkpoint_path), device, model_type)
    return model