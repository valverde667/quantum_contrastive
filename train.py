import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import STL10

from quantum_contrastive.models.contrastive_model import ContrastiveModel
from quantum_contrastive.losses.contrastive import InfoNCELoss

import os
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import torchvision.transforms as transforms


def get_dataloaders(batch_size=128):
    # Base directory relative to this file
    try:
        base = os.path.dirname(__file__)
    except NameError:
        base = os.getcwd()

    data_root = os.path.join(base, "data", "stl10")
    extracted_flag_file = os.path.join(data_root, "stl10_binary", "train_X.bin")

    # Check if data is already downloaded/extracted
    if not os.path.isfile(extracted_flag_file):
        print(f"STL-10 data not found. Downloading to: {data_root}")
        os.makedirs(data_root, exist_ok=True)
        _ = STL10(root=data_root, split="train", download=True)
    else:
        print(f"STL-10 dataset already exists at {data_root}. Skipping download.")

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    train_dataset = STL10(
        root=data_root, split="train", download=False, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return train_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x_i = x
        x_j = x.clone()

        x_i, x_j = x_i.to(device), x_j.to(device)
        _, z_i = model(x_i)
        _, z_j = model(x_j)

        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveModel().to(device)
    criterion = InfoNCELoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loader = get_dataloaders()

    for epoch in range(10):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")


if __name__ == "__main__":
    main()
