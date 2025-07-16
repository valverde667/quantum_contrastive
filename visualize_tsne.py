import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.manifold import TSNE
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader
from quantum_contrastive.models.contrastive_model import ContrastiveModel


def extract_features(encoder, dataloader, device):
    encoder.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            feats = encoder(images)
            features.append(feats.cpu())
            labels.append(targets)

    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def visualize_tsne(features, labels, save_path="tsne_plot.svg"):
    # Ensure features is 2D: [n_samples, n_features]
    features = features.reshape(features.shape[0], -1)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=labels,
        palette=sns.color_palette("hls", 10),
        legend="full",
        alpha=0.7,
    )
    plt.title("t-SNE of STL-10 Representations")
    plt.savefig(save_path)
    plt.show()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load pretrained model (adjust if you saved weights)
    model = ContrastiveModel().to(device)
    model.load_state_dict(torch.load("contrastive_model.pth", map_location=device))
    encoder = model.encoder

    # Eval transform (no augmentation)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
        ]
    )
    dataset = STL10(
        root="data/stl10", split="test", download=False, transform=eval_transform
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # Extract features and labels
    feats, labels = extract_features(encoder, dataloader, device)

    # t-SNE + plot
    visualize_tsne(feats, labels)


if __name__ == "__main__":
    main()
