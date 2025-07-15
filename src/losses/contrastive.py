# losses/contrastive.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, zis, zjs):
        """
        Input:
            zis: tensor of shape (N, D) - projections from view 1
            zjs: tensor of shape (N, D) - projections from view 2
        Output:
            Scalar InfoNCE loss
        """
        batch_size = zis.shape[0]

        # Normalize to unit vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # Concatenate: (2N, D)
        z = torch.cat([zis, zjs], dim=0)

        # Cosine similarity matrix: (2N, 2N)
        similarity_matrix = torch.matmul(z, z.T)

        # Remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        # Labels: positives are at fixed offsets
        positives = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
        ).to(z.device)

        # Similarities of positive pairs
        sim_ij = torch.sum(zis * zjs, dim=-1) / self.temperature  # shape: (N,)
        sim_ji = torch.sum(zjs * zis, dim=-1) / self.temperature  # shape: (N,)
        positives_sim = torch.cat([sim_ij, sim_ji], dim=0)

        # Scale whole matrix by temperature
        logits = similarity_matrix / self.temperature

        # Final loss
        loss = F.cross_entropy(logits, positives)
        return loss
