# losses/contrastive.py

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, zis, zjs):
        batch_size = zis.size(0)

        # Normalize to unit vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # Concatenate along batch dimension
        z = torch.cat([zis, zjs], dim=0)  # shape: [2N, D]

        # Cosine similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # shape: [2N, 2N]

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Create target indices — positives are at a fixed offset
        targets = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
        ).to(z.device)

        # Cross-entropy expects logits of shape [batch, classes]
        # So we need to treat each row of sim as the logits
        # and the target index in that row is the positive example
        loss = F.cross_entropy(sim, targets)

        return loss
