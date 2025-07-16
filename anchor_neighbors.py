# anchor_neighbors.py

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from quantum_contrastive.models.contrastive_model import ContrastiveModel

# Set up transform
transform = transforms.Compose(
    [
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ]
)


# Load STL-10 test set
def get_test_loader(batch_size=128):
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

    dataset = STL10(root=data_root, split="test", download=False, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False), dataset


# Load trained encoder
def load_encoder(path, device):
    model = ContrastiveModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.encoder.eval()


# Compute cosine similarities
def compute_features(encoder, loader, device):
    all_feats = []
    all_imgs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feats = encoder(imgs)  # shape: (B, 512, 1, 1)
            feats = F.normalize(feats.view(feats.size(0), -1), dim=1)  # (B, 512)
            all_feats.append(feats.cpu())
            all_imgs.append(imgs.cpu())
    return torch.cat(all_feats), torch.cat(all_imgs)


# Select anchor and find top-K nearest neighbors
def get_neighbors(features, images, anchor_idx=0, k=5):
    anchor_feat = features[anchor_idx]
    sims = torch.matmul(features, anchor_feat)  # (N,)
    topk = torch.topk(sims, k=k + 1)  # include anchor itself
    indices = topk.indices[1:]  # skip self
    return images[anchor_idx], images[indices]


# Visualize anchor and neighbors
def visualize(anchor_img, neighbor_imgs, output_path="nearest_neighbors.png"):
    fig, axes = plt.subplots(1, len(neighbor_imgs) + 1, figsize=(15, 3))
    axes[0].imshow(anchor_img.permute(1, 2, 0))
    axes[0].set_title("Anchor")
    axes[0].axis("off")
    for i, img in enumerate(neighbor_imgs):
        axes[i + 1].imshow(img.permute(1, 2, 0))
        axes[i + 1].set_title(f"Neighbor {i+1}")
        axes[i + 1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# Main logic
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder = load_encoder("contrastive_model.pth", device)

    loader, dataset = get_test_loader(batch_size=256)
    features, images = compute_features(encoder, loader, device)

    anchor_img, neighbor_imgs = get_neighbors(features, images, anchor_idx=42 * 4, k=5)
    visualize(anchor_img, neighbor_imgs)


if __name__ == "__main__":
    main()
