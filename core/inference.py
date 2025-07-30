from VAE_models import load_vae_model, CVAE, VAE
from nf_model import MLP_Masked
#from audio_image_pipeline import
import sys
import torch
from nf_model import MLP_Masked



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
    elif model_type == 'NF':
        model = MLP_Masked(
            checkpoint['config']['input_dim'],
            checkpoint['config']['hidden_dim'],
            checkpoint['config']['conv']
        ).to(device)
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def reconstruction(model, model_type:str, samples, conv:bool):

    model.eval()
    with torch.no_grad():
        if model_type == 'VAE':
            if not conv and len(samples.size) > 1:
                dims = samples.size()
                samples = samples.view(dims[0], -1)
            outputs = model(samples, compute_loss=False)
            recon = outputs.x_recon
            #latent_sample = outputs.
            if not conv and len(samples.size) > 1:
                recon = recon.view(dims)
        if model_type == 'NF':
            pass
        else:
            raise NameError('Provide valid model type!')
    return recon


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