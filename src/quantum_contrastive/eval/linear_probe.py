import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_linear_probe(encoder, dataloader, num_classes, device, epochs=10, lr=1e-3):
    encoder.eval()  # freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    # Assume encoder returns flattened output (we can adapt if not)
    dummy_input = next(iter(dataloader))[0][0].to(device)  # (x, label)
    with torch.no_grad():
        dummy_output = encoder(dummy_input.unsqueeze(0))
    feat_dim = dummy_output.shape[1]

    # Linear classifier
    clf = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    clf.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, labels = x.to(device), labels.to(device)
            with torch.no_grad():
                feats = encoder(x)
                feats = feats.view(
                    feats.size(0), -1
                )  # Reshap from [B, 512, 1, 1] to [B, 512]

            logits = clf(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

    return clf
