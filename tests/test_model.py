# tests/test_model.py

import torch
from models.contrastive_model import ContrastiveModel


def test_contrastive_model_output_shapes():
    model = ContrastiveModel(projection_dim=128)
    batch_size = 16
    dummy_input = torch.randn(batch_size, 3, 96, 96)

    h, z = model(dummy_input)

    assert isinstance(h, torch.Tensor)
    assert isinstance(z, torch.Tensor)

    assert h.shape == (batch_size, 512), f"Expected h shape (16, 512), got {h.shape}"
    assert z.shape == (batch_size, 128), f"Expected z shape (16, 128), got {z.shape}"
