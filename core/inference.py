from VAE_models import load_vae_model, CVAE, VAE
from audio_image_pipeline import 
import sys
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def reconstruction(model, samples, conv:bool):

    model.eval()
    with torch.no_grad():
        if not conv and len(samples.size) > 1:
            dims = samples.size()
            samples = samples.view(dims[0], -1)
        outputs = model(samples, compute_loss=False)
        recon = outputs.x_recon
        latent_sample
        if not conv and len(samples.size) > 1:
            recon = recon.view(dims)
    return recon

def_