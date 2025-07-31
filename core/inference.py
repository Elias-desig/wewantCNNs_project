from .VAE_models import CVAE, VAE
from .nf_model import MLP_Masked
from .audio_image_pipeline import audio_to_melspectrogram, melspectrogram_to_audio
import sys
from pathlib import Path
import torch




def load_model(checkpoint_path, device, model_type):
    """Load VAE model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = None
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
    elif model_type == 'NF':
        model = MLP_Masked(
            checkpoint['config']['input_dim'],
            checkpoint['config']['hidden_dim'],
            checkpoint['config']['conv']
        ).to(device)
    else:
        raise ValueError(f'Not a valid model type:{model_type!r}')
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def reconstruction(model, model_type:str, sample_path:str, conv:bool):
    even = model_type == 'CVAE'
    sample = audio_to_melspectrogram(sample_path, even=even)
    device = next(model.parameters()).device
    sample = sample.to(device)
    model.eval()
    with torch.no_grad():
        if model_type == 'VAE' or model_type == 'CVAE':
            if conv:
                sample = sample.unsqueeze(0).unsqueeze(0)
            if not conv:
                dims = sample.size()
                sample = sample.view(1, -1)
            outputs = model(sample, compute_loss=False)
            recon = outputs.x_recon.cpu()
            z = outputs.z_sample

            # Reshape the output correctly
            if conv:
                recon = recon.squeeze(0).squeeze(0)
            else:
                recon = recon.view(128, 173)
        elif model_type == 'NF':
            pass
        else:
            raise ValueError(f'Provide valid model type!{model_type}')
    audio = melspectrogram_to_audio(recon)

    return audio, recon, z

def generate(model, model_type:str, latent_sample, output_dim:tuple[str], conv:bool):
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
    checkpoint_path = "/Users/koraygecimli/PycharmProjects/UDL_demo/wewantCNNs_project/models/nf_checkpoint_20250731-002053.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = checkpoint['config']['input_dim']
    hidden_dims = checkpoint['config']['hidden_dims']

    model = MLP_Masked(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.eval()


def invert_flow(model: MLP_Masked, z: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    batch_size, dim = z.size()
    x_recon = torch.zeros_like(z).to(device)

    with torch.no_grad():
        for i in range(dim):
            # x_partial enthält bisher rekonstruierte Features, der Rest ist 0
            x_partial = x_recon.clone()

            # Feature i wird jetzt berechnet, also Model auf x_partial forwarden
            out = x_partial
            for layer in model.architecture:
                out = layer(out)
            output = model.output_layer(out)

            s = torch.clamp(output[:, :dim], min=-5, max=5)
            t = output[:, dim:]

            x_recon[:, i] = z[:, i] * torch.exp(s[:, i]) + t[:, i]

    return x_recon



def select_model(model_type, device=None):
    base_dir = Path(__file__).parent.parent
    checkpoint_dir = base_dir / 'models' / 'inference'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patterns = {
        'VAE':  '*vae_checkpoint_*.pt',
        'CVAE': '*c_vae_checkpoint_*.pt',
        'NF':   '*nf_checkpoint_*.pt'
    }
    try:
        pat = patterns[model_type]
    except KeyError:
        raise ValueError(f"Unknown model_type {model_type!r}")

    matches = sorted(checkpoint_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No checkpoint file matching `{pat}` in {checkpoint_dir}")

    checkpoint_path = matches[0]
    model, checkpoint = load_model(str(checkpoint_path), device, model_type)
    return model