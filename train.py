import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import STL10
import matplotlib.pyplot as plt
import random
import numpy as np

from quantum_contrastive.models.contrastive_model import ContrastiveModel
from quantum_contrastive.losses.contrastive import InfoNCELoss
from quantum_contrastive.eval.linear_probe import train_linear_probe
from quantum_contrastive.eval.knn_eval import knn_evaluate
from quantum_contrastive.visual.plot_format import set_plot_style

set_plot_style()


# Set random seed
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For dataloaders with multiple workers
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


# You’ll need this if you use contrastive views
class ContrastiveTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def get_dataloaders(batch_size=128, for_eval=False):
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

    # Set transform
    base_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    if for_eval:
        eval_transform = transforms.Compose(
            [
                transforms.Resize(96),
                transforms.CenterCrop(96),
                transforms.ToTensor(),
            ]
        )
        dataset = STL10(
            root=data_root, split="test", download=False, transform=eval_transform
        )
    else:
        contrastive_transform = ContrastiveTransform(base_transform)
        dataset = STL10(
            root=data_root,
            split="train",
            download=False,
            transform=contrastive_transform,
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    losses = []
    for x, _ in dataloader:
        x_i, x_j = x
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
    print("---- Beginning contrastive training.")
    for epoch in range(10):
        avg_loss, batch_loss = train_one_epoch(
            model, loader, optimizer, criterion, device
        )
        all_epoch_losses.append(avg_loss)

    # Save model
    torch.save(model.state_dict(), "contrastive_model.pth")

    # Plot history of losses
    plt.scatter([i for i in range(len(all_epoch_losses))], all_epoch_losses)
    plt.title("Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("loss_curve.svg")
    # plt.show()

    # Run linear evaluation
    print("---- Beginning linear probe.")
    encoder = model.encoder
    eval_loader = get_dataloaders(for_eval=True)
    train_linear_probe(encoder, eval_loader, num_classes=10, device=device)

    # Run KNN evaluation
    print("---- Beginning KNN evaluation.")
    test_loader = get_dataloaders(for_eval=True)
    knn_evaluate(encoder, eval_loader, test_loader, device=device, k=5)


if __name__ == "__main__":
    main()
