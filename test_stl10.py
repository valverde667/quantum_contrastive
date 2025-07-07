# test_stl10.py
from torchvision.datasets import STL10
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = STL10(root="./data", split="train", download=True, transform=transform)
print(f"Loaded {len(dataset)} images from STL-10 train split")
