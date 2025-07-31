from pathlib import Path
from .VAE_models import CVAE, VAE, CVAE_Deep
from .nf_model import MLP_Masked
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
            checkpoint['config']['in_dim'],
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


def reconstruction(model, model_type: str, samples, conv: bool):
    model.eval()
    with torch.no_grad():
        if model_type == 'VAE':
            if not conv and len(samples.size) > 1:
                dims = samples.size()
                samples = samples.view(dims[0], -1)
            outputs = model(samples, compute_loss=False)
            # image reconstruction
            recon = outputs.x_recon
            # latent space code
            latent_sample = outputs.z_sample
            if not conv and len(samples.size) > 1:
                recon = recon.view(dims)
        if model_type == 'NF':
            pass
        else:
            raise NameError('Provide valid model type!')
    return recon, latent_sample


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
        'VAE':  '*vae_checkpoint_*.pt',
        'CVAE': '*c_vae_checkpoint_*.pt',
        'NF':   '*nf_checkpoint_*.pt',
        'CVAE_Deep': '*deep_c_vae_checkpoint_*.pt'
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