from VAE_models import load_vae_model, CVAE, VAE
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

if __name__ == "main":
    model_type = sys.argv[0]
    checkpoint = sys.argv[1]
    if model_type == 'vae'
    model = load_model()