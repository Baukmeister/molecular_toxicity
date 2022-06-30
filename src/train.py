import torch

from src.model import GCN

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0)
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.

        out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.
