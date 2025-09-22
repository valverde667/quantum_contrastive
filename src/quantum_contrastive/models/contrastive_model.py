import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import pennylane as qml


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=128, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        return self.net(x)


class QuantumVQCHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 8,
        n_layers: int = 1,
        dev_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # simple linear map to angles
        self.pre = nn.Linear(in_dim, n_qubits)
        # trainable circuit weights (float32, CPU by default)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        # CPU device; avoid MPS here
        self.dev = qml.device(dev_name, wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, weights):
            # x: shape (n_qubits,) — ensure this upstream
            # Manual amplitude embedding via RY to avoid template/version quirks
            for w in range(n_qubits):
                qml.RY(x[w], wires=w)

            # Variational block
            for l in range(n_layers):
                # single-qubit rotations
                for w in range(n_qubits):
                    qml.RX(weights[l, w, 0], wires=w)
                    qml.RY(weights[l, w, 1], wires=w)
                    qml.RZ(weights[l, w, 2], wires=w)
                # ring entanglement
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def _scale_to_angles(self, a: torch.Tensor) -> torch.Tensor:
        # keep values finite and stable for training
        a = torch.tanh(a) * (torch.pi / 2)  # scale to [-π/2, π/2]
        return torch.clamp(a, -torch.pi, torch.pi)  # optional extra safety clamp

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]

        # Compute angles on SAME device as h, then move to CPU for the circuit
        a = self.pre(h)  # (B, n_qubits)
        a = self._scale_to_angles(a)
        a_cpu = a.to("cpu", dtype=torch.float32)
        w_cpu = self.weights.to("cpu", dtype=torch.float32)

        outs = []
        for i in range(B):
            res = self.circuit(a_cpu[i], w_cpu)
            # Convert list/tuple of scalars to a 1D torch tensor
            if isinstance(res, (list, tuple)):
                zi = torch.stack(
                    [
                        (
                            v
                            if isinstance(v, torch.Tensor)
                            else torch.tensor(v, dtype=torch.float32)
                        )
                        for v in res
                    ],
                    dim=0,
                )
            else:
                zi = (
                    res
                    if isinstance(res, torch.Tensor)
                    else torch.tensor(res, dtype=torch.float32)
                )
            outs.append(zi)

        z_cpu = torch.stack(outs, dim=0)  # (B, n_qubits)
        z = z_cpu.to(h.device, dtype=h.dtype)

        return F.normalize(z, dim=-1)


class ContrastiveModel(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        # Load pretrained ResNet18 and remove final classification layer
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # remove FC
        self.encoder_out_dim = base_model.fc.in_features

        # self.projection_head = ProjectionHead(self.encoder_out_dim, projection_dim)

        # projection_dim will equal the number of qubits / output dims of the quantum head
        q_out = 8  # number of qubits to actually use (fast & feasible)
        self.projection_head = nn.Sequential(
            QuantumVQCHead(in_dim=self.encoder_out_dim, n_qubits=q_out, n_layers=1),
            nn.Linear(q_out, projection_dim),  # lift to 128-D if your code expects it
        )

    def forward(self, x):
        h = self.encoder(x)  # shape: (B, 512, 1, 1)
        h = torch.flatten(h, 1)  # shape: (B, 512)
        z = self.projection_head(h)  # shape: (B, 128)
        return h, z  # return both for later comparison
