# train_gcn.py
from gcn_model import GCN
import torch

def train_gcn(data, epochs=100):
    model = GCN(in_channels=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Unsupervised: dummy loss using node similarity (e.g., contrastive/autoencoding in real cases)
        loss = (out ** 2).sum()  # dummy loss just to let model learn embeddings
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return model
