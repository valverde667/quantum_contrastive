import torch
import torch.nn as nn
import torchvision.models as models

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=128, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)

class ContrastiveModel(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        # Load pretrained ResNet18 and remove final classification layer
        base_model = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # remove FC
        self.encoder_out_dim = base_model.fc.in_features

        self.projection_head = ProjectionHead(self.encoder_out_dim, projection_dim)

    def forward(self, x):
        h = self.encoder(x)         # shape: (B, 512, 1, 1)
        h = torch.flatten(h, 1)     # shape: (B, 512)
        z = self.projection_head(h) # shape: (B, 128)
        return h, z                 # return both for later comparison
