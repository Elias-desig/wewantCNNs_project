import torch
from torch.utils.data import DataLoader
from nf_model import MLP_Masked  # import our wonderful model that hopefully will work perfectly
from datetime import datetime
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda for nivida grafikkarten since cuda is the shit


def save_checkpoint(model, optimizer, epoch, loss, input_dim, hidden_dims, conv=False):
    if not os.path.isdir('/Users/koraygecimli/PycharmProjects/UDL_demo/wewantCNNs_project/models'):
        os.mkdir('/Users/koraygecimli/PycharmProjects/UDL_demo/wewantCNNs_project/models')

    folder = '/Users/koraygecimli/PycharmProjects/UDL_demo/wewantCNNs_project/models/nf_model'
    if not os.path.isdir(folder):
        os.mkdir(folder)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'conv': conv
        },
        'timestamp': timestamp
    }

    torch.save(checkpoint, f'{folder}/nf_checkpoint_{timestamp}.pt')
    print(f"Checkpoint saved at epoch {epoch + 1}")


def train_model(train_loader, input_dim, hidden_dims=[1024, 1024], num_epochs=20, learning_rate=1e-3):
    model = MLP_Masked(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device).float()
            batch = batch.view(batch.size(0), -1)  # Flatten falls nötig

            optimizer.zero_grad()
            output = model(batch, compute_loss=True)
            loss = output.loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, best_loss, input_dim, hidden_dims)

    return model