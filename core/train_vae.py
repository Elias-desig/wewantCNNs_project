import torch
from torch import autocast, amp
from audio_image_pipeline import audio_to_melspectrogram
from audio_image_pipeline import melspectrogram_to_audio
from audio_image_pipeline import ave_spectrogram_image
import math
from tqdm import tqdm


def train(model, dataloader, optimizer, prev_updates, config, device, writer=None, conv=False):
    """
    Trains the model on the given data.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    use_amp = True
    model.train()  # Set the model to training mode
    scaler = amp.GradScaler("cuda", enabled=use_amp)


    for batch_idx, data in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx

        data = data.to(device, non_blocking=True)
        if not conv:
            data = data.view(data.size(0), -1) # Flatten data when not using convolutional VAE
        else:
            data = data.view(data.size(0), 1, data.size(1), data.size(2)) # (batch, 1, h, w)
        with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(data)  # Forward pass
            loss = output.loss
        
        scaler.scale(loss).backward()
        if n_upd % 500 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        
            print(f'Step {n_upd:,} (N samples: {n_upd*config.vae.batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)        
        scaler.unscale_(optimizer)
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)        
    return prev_updates + len(dataloader)

def test(model, dataloader, cur_step, config, device, writer=None, conv=False):
    """
    Tests the model on the given data.
    
    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            if not conv:
                data = data.view(data.size(0), -1)  # Flatten the data when not using convolutional VAE
            else:
                data = data.view(data.size(0), 1, data.size(1), data.size(2))
            output = model(data, compute_loss=True)  # Forward pass
            
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()
            
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
    
    H, W = dataloader.dataset[0].shape
    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', output.loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', output.loss_kl.item(), global_step=cur_step)
        
        # Log reconstructions
        writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, H, W), global_step=cur_step)
        writer.add_images('Test/Originals', data.view(-1, 1, H, W), global_step=cur_step)
    
        # Log random samples from the latent space
        z = torch.randn(16, config.vae.latent_dim).to(device)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, H, W), global_step=cur_step)
    return test_loss, test_recon_loss, test_kl_loss

def simple_kl_annealing(epoch):
    beta = torch.sigmoid(torch.tensor(epoch - 10.0)) * 3
    return float(beta.clamp(min=1e-2))