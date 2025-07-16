import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import STL10
import matplotlib.pyplot as plt

from quantum_contrastive.models.contrastive_model import ContrastiveModel
from quantum_contrastive.losses.contrastive import InfoNCELoss


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


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    losses = []
    for x, _ in dataloader:
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
        losses.append(loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch Loss: {avg_loss:.4f}")

    return avg_loss, losses


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ContrastiveModel().to(device)
    criterion = InfoNCELoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loader = get_dataloaders()

    all_epoch_losses = []
    for epoch in range(10):
        avg_loss, batch_loss = train_one_epoch(
            model, loader, optimizer, criterion, device
        )
        all_epoch_losses.append(avg_loss)

    # Plot history of losses
    plt.scatter([i for i in range(len(all_epoch_losses))], all_epoch_losses)
    plt.title("Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
