import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import pennylane as qml
import numpy as np


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


class BottleneckLinearHead(nn.Module):
    def __init__(self, in_dim=512, bottleneck_dim=8, out_dim=128):
        super().__init__()
        self.pre = nn.Linear(in_dim, bottleneck_dim)  # 512 -> n_q (or 3*n_q)
        self.lift = nn.Linear(bottleneck_dim, out_dim)  # n_q (or 3*n_q) -> 128

    def forward(self, h):
        z = self.lift(self.pre(h))
        return F.normalize(z, dim=-1)


class QuantumFeatureMap:
    """Class to hold the creation of a quantum feature map."""

    def __init__(self, n_qubits=4, n_layers=2, encoding="ry", entangle="ring"):
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.encoding, self.entangle = encoding, entangle
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(angles):
            self._apply_feature_map(angles)
            return qml.state()

        self._circuit = circuit

    def _to_angles(self, x: torch.Tensor) -> torch.Tensor:
        # map features to (-pi, pi) for stable rotations
        x = torch.tanh(x) * np.pi
        need = self.n_qubits * self.n_layers
        if x.numel() < need:
            reps = (need + x.numel() - 1) // x.numel()
            x = x.repeat(reps)[:need]
        else:
            x = x[:need]
        return x.to(dtype=torch.float32)  # Force fp32 into QNode

    def _apply_feature_map(self, angles: torch.Tensor):
        """
        angles: shape [n_layers * n_qubits] or [n_qubits].
        - If [n_layers*n_qubits], we layer-specifically reupload angles[layer, q].
        - If [n_qubits], we reuse the same angles at each layer (classic reuploading).
        """
        nq, L = self.n_qubits, self.n_layers

        # Normalize angles shape
        if angles.numel() == nq * L:
            A = angles.view(L, nq)
        elif angles.numel() == nq:
            A = angles.unsqueeze(0).expand(L, nq)  # same angles each layer
        else:
            raise ValueError(
                f"angles has {angles.numel()} elems; expected {nq} or {nq*L}"
            )

        for l in range(L):
            # --- data reuploading: encode per-qubit angles each layer ---
            for q in range(nq):
                # (Optional) spread info across axes; fall back to your encoding flag
                if getattr(self, "encoding", "ry").lower() == "rx":
                    qml.RX(A[l, q], wires=q)
                elif getattr(self, "encoding", "ry").lower() == "ry":
                    qml.RY(A[l, q], wires=q)
                else:
                    # a slightly richer default if unknown: phase+x
                    qml.RZ(A[l, q], wires=q)
                    qml.RX(A[l, q], wires=q)

            # --- entanglement / interactions ---
            ent = getattr(self, "entangle", "ring").lower()
            if ent == "ring":
                # CZ ring
                for q in range(nq):
                    qml.CZ(wires=[q, (q + 1) % nq])
                # (Optional) add lightweight ZZ-style interactions for more expressivity:
                # for q in range(nq):
                #     q_next = (q + 1) % nq
                #     qml.IsingZZ(A[l, q] * A[l, q_next], wires=[q, q_next])

            elif ent == "linear":
                for q in range(nq - 1):
                    qml.CZ(wires=[q, q + 1])
                # (Optional) extra: qml.IsingZZ(A[l, q] * A[l, q+1], wires=[q, q+1])

            else:
                # default to a simple ring if unknown
                for q in range(nq):
                    qml.CZ(wires=[q, (q + 1) % nq])

    def embed(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Returns a [B, 2**n_qubits] complex tensor of statevectors."""
        return torch.stack([self._circuit(self._to_angles(x)) for x in x_batch])

    def compute_fidelity_matrix(self, x_batch):
        states = self.embed(x_batch)  # [N, 2**nq] complex
        states = states / torch.linalg.vector_norm(states, dim=1, keepdim=True)
        overlaps = states @ states.conj().T  # [N, N] complex
        fids = overlaps.abs() ** 2  # [N, N] in [0,1]
        return fids.to(torch.float32)  # Force fp32


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

            # Return the expectation values of the obervables X, Y, and Z. Here,
            # the return value can be changed to add or subtract other observables.
            return [
                qml.expval(op(i))
                for i in range(n_qubits)
                for op in [qml.PauliX, qml.PauliY, qml.PauliZ]
            ]

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
    def __init__(self, projection_dim=128, use_feature_map=False, qfm_kwargs=None):
        super().__init__()
        # Load pretrained ResNet18 and remove final classification layer
        # Use weights = ResNet18_Weights.DEFAULT or NONE.
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # remove FC
        self.encoder_out_dim = base_model.fc.in_features
        self.use_feature_map = use_feature_map

        # Create switch between using quantum feature map (skip head) or a head projection
        if use_feature_map:
            self.qfm = QuantumFeatureMap(**(qfm_kwargs or {}))
            self.projection_head = None
        else:
            # Projection head selection. Uncomment the section below to select the
            # desired head. The first is the MLP and the second is the Bottlenecklinear.
            # The last option is the VQC where the parameters can be set.
            self.projection_head = ProjectionHead(self.encoder_out_dim, projection_dim)
            # self.projection_head = BottleneckLinearHead()

            # VQC head parameters. n_observables should be checked with the model
            # return from class QuantumVQCHead(nn.Module).
            # q_out = 8  # number of qubits to actually use (fast & feasible)
            # n_observables = 3 # number of observables being measured
            # self.projection_head = nn.Sequential(
            #     QuantumVQCHead(in_dim=self.encoder_out_dim, n_qubits=q_out, n_layers=1),
            #     nn.Linear(n_observables*q_out, projection_dim),  # lift to 128-D if your code expects it
            # )

    def forward(self, x, return_embedding=False):
        h = self.encoder(x)  # shape: (B, 512, 1, 1)
        h = torch.flatten(h, 1)  # shape: (B, 512)

        # Based on selection, use a feature map or projeciton head.
        if self.use_feature_map:
            if return_embedding:
                return h
            else:
                raise RuntimeError(
                    "Quantum Feature Map requires return_embedding=True."
                )
        else:
            z = self.projection_head(h)  # shape: (B, 128)
            return h, z  # return both for later comparison
