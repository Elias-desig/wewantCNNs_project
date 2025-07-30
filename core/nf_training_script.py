import torch
from torch.utils.data import DataLoader
from nf_model import MLP_Masked  # dein Model importieren
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
num_epochs = 20
learning_rate = 1e-3
batch_size = 32

model = MLP_Masked(input_dim=22144, hidden_dims=[1024, 1024]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")