import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import STL10
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import argparse

from quantum_contrastive.models.contrastive_model import ContrastiveModel
from quantum_contrastive.losses.contrastive import InfoNCELoss
from quantum_contrastive.eval.linear_probe import train_linear_probe
from quantum_contrastive.eval.knn_eval import knn_evaluate
from quantum_contrastive.visual.plot_format import set_plot_style
import utility

set_plot_style()


# Helper function for synching execution times.
def dev_synchronize(tensor_or_device=None):
    """Synchronize if using CUDA/MPS so timers are accurate."""
    try:
        dev_type = None
        if tensor_or_device is None:
            return
        if isinstance(tensor_or_device, torch.device):
            dev_type = tensor_or_device.type
        elif torch.is_tensor(tensor_or_device):
            dev_type = tensor_or_device.device.type

        if dev_type == "cuda":
            torch.cuda.synchronize()
        elif dev_type == "mps":
            torch.mps.synchronize()
    except Exception:
        # Safe no-op if not available
        pass


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


def get_dataloaders(
    batch_size: int = 128,
    for_eval: bool = False,
    drop_last: bool | None = None,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    device=None,
):
    """
    Returns a DataLoader for STL-10.
    - for_eval=False: contrastive train loader ((x_i, x_j), y), shuffled, drop_last=True by default.
    - for_eval=True:  eval/test loader (x, y), not shuffled, drop_last=False by default.
    """
    # Decide default drop_last based on mode
    if drop_last is None:
        drop_last = not for_eval

    if pin_memory is None:
        # Only useful on CUDA; MPS/CPU: set False to avoid the warning
        pin_memory = device is not None and getattr(device, "type", "") == "cuda"

    # Base directory relative to this file
    try:
        base = os.path.dirname(__file__)
    except NameError:
        base = os.getcwd()

    data_root = os.path.join(base, "data", "stl10")
    extracted_flag_file = os.path.join(data_root, "stl10_binary", "train_X.bin")

    # Ensure dataset is present
    if not os.path.isfile(extracted_flag_file):
        print(f"STL-10 data not found. Downloading to: {data_root}")
        os.makedirs(data_root, exist_ok=True)
        _ = STL10(root=data_root, split="train", download=True)
    else:
        print(f"STL-10 dataset already exists at {data_root}. Skipping download.")

    # Transforms
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
        shuffle = False
    else:
        contrastive_transform = ContrastiveTransform(base_transform)
        dataset = STL10(
            root=data_root,
            split="train",
            download=False,
            transform=contrastive_transform,
        )
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    return loader


# ---- Functions particularl to training  with a qfm
@torch.no_grad()
def _build_targets(B: int, device):
    """For a 2N batch (two views), positives are (i, i+N) and (i+N, i)."""
    assert B % 2 == 0, "Batch must be even: contains two views concatenated."
    N = B // 2
    t = torch.arange(B, device=device)
    t[:N] += N
    t[N:] -= N
    return t


def _info_nce_from_similarity(sim: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    InfoNCE on a [2N, 2N] similarity matrix where sim[i,j] is higher for positives.
    We mask the diagonal and use cross-entropy row-wise.
    """
    B = sim.shape[0]
    device = sim.device
    targets = _build_targets(B, device)

    # Numeric hygiene: clamp to avoid -inf/NaN when using as logits
    eps = 1e-6
    sim = sim.clamp(min=eps, max=1 - eps)

    # mask self-similarity
    sim = sim.clone()
    sim.fill_diagonal_(-float("inf"))

    logits = sim / tau
    return F.cross_entropy(logits, targets, reduction="mean")


def fidelity_kernel_matrix(states: torch.Tensor) -> torch.Tensor:
    """
    states: [N, D] complex, row-normalized |psi>.
    returns K in [0,1], K_ij = |<psi_i|psi_j>|^2
    """
    inner = states @ torch.conj(states).t()  # [N,N] complex
    K = (inner.abs() ** 2).real
    return K.clamp_(0.0, 1.0)


def rbf_kernel_matrix(X: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    """
    X: [N, D] real. If gamma is None, use median heuristic.
    returns K_ij = exp(-gamma * ||x_i - x_j||^2)
    """
    # pairwise squared distances
    XX = (X * X).sum(dim=1, keepdim=True)
    D2 = XX + XX.t() - 2 * (X @ X.t())
    D2 = D2.clamp_min_(0.0)

    if gamma is None:
        # median heuristic (exclude diagonal)
        tri = D2[~torch.eye(D2.size(0), dtype=torch.bool, device=D2.device)]
        med = torch.median(tri)
        gamma = (1.0 / (2.0 * med.clamp_min(1e-6))).item()
    return torch.exp(-gamma * D2)


def mmd2_unbiased(
    Kxx: torch.Tensor, Kyy: torch.Tensor, Kxy: torch.Tensor
) -> torch.Tensor:
    """
    Unbiased MMD^2:
      E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]
    with diagonals removed from Kxx, Kyy.
    """
    n = Kxx.size(0)
    m = Kyy.size(0)
    assert Kxx.shape == (n, n) and Kyy.shape == (m, m) and Kxy.shape == (n, m)

    # remove diagonals
    Kxx_no_diag = Kxx - torch.diag(torch.diag(Kxx))
    Kyy_no_diag = Kyy - torch.diag(torch.diag(Kyy))

    term_xx = Kxx_no_diag.sum() / (n * (n - 1) + 1e-9)
    term_yy = Kyy_no_diag.sum() / (m * (m - 1) + 1e-9)
    term_xy = Kxy.sum() / (n * m + 1e-9)

    return term_xx + term_yy - 2.0 * term_xy


def mmd_alignment_loss(
    states_or_feats_1, states_or_feats_2, kernel="quantum"
) -> torch.Tensor:
    """
    Minimizes MMD^2 between the two view-distributions.
    kernel: "quantum" => fidelity on complex states; "rbf" => RBF on real vectors
    """
    if kernel == "quantum":
        Kxx = fidelity_kernel_matrix(states_or_feats_1)
        Kyy = fidelity_kernel_matrix(states_or_feats_2)
        Kxy = (states_or_feats_1 @ torch.conj(states_or_feats_2).t()).abs() ** 2
    elif kernel == "rbf":
        Kxx = rbf_kernel_matrix(states_or_feats_1)
        Kyy = rbf_kernel_matrix(states_or_feats_2)
        # cross kernel
        X, Y = states_or_feats_1, states_or_feats_2
        XX = (X * X).sum(dim=1, keepdim=True)
        YY = (Y * Y).sum(dim=1, keepdim=True)
        D2 = XX + YY.t() - 2 * (X @ Y.t())
        D2 = D2.clamp_min_(0.0)
        gamma = None  # median heuristic per batch
        if gamma is None:
            med = torch.median(D2.view(-1))
            gamma = (1.0 / (2.0 * med.clamp_min(1e-6))).item()
        Kxy = torch.exp(-gamma * D2)
    else:
        raise ValueError("kernel must be 'quantum' or 'rbf'.")

    return mmd2_unbiased(Kxx, Kyy, Kxy)


def train_one_epoch_qfm(
    model,
    dataloader,
    optimizer,
    device,
    tau: float = 0.1,
    loss_type: str = "infonce",
):
    """
    Train one epoch using the Quantum Feature Map path.
    Expects dataloader to yield ((x_i, x_j), _) just like your classical loop.
    """
    model.train()
    total_loss = 0.0
    losses = []

    epoch_t0 = time.perf_counter()
    batch_s_list = []

    # Initialize metrics
    tau_used, sigma2_used, off_mu, off_sd, delta = None, None, None, None, None

    for x, _ in dataloader:
        t0 = time.perf_counter()
        x_i, x_j = x
        x_i, x_j = x_i.to(device), x_j.to(device)

        # 1) Encode to features h (no projection head)
        h_i = model(x_i, return_embedding=True)  # [B, D]
        h_j = model(x_j, return_embedding=True)  # [B, D]
        h = torch.cat([h_i, h_j], dim=0)  # [2B, D]

        # 2) Compute pairwise fidelities with the QFM
        S = model.qfm.compute_fidelity_matrix(h).clamp(0.0, 1.0)  # [2B, 2B], in [0,1]

        # 3) Choose loss
        if loss_type == "infonce":
            tau_used = tau

            # Diagnostics
            off_mu, off_sd = utility.offdiag_stats(S.detach())
            delta = utility.pos_neg_gap_from_full(S.detach())

            # Standard NT-Xent on similarities
            N = S.size(0)  # 2B
            S = S.masked_fill(torch.eye(N, device=S.device, dtype=torch.bool), 0.0)
            logits = S / tau
            targets = (torch.arange(N, device=S.device) + N // 2) % N
            loss = F.cross_entropy(logits, targets)

        elif loss_type == "mmd":
            # Unbiased MMD^2 between the two view distributions
            B = S.size(0) // 2
            Kxx = S[:B, :B].clone()  # view-1 vs view-1
            Kyy = S[B:, B:].clone()  # view-2 vs view-2
            Kxy = S[:B, B:].clone()  # view-1 vs view-2 (cross)
            # (Optional) clamp for numerical safety
            Kxx.clamp_(0.0, 1.0)
            Kyy.clamp_(0.0, 1.0)
            Kxy.clamp_(0.0, 1.0)

            # Unbiased MMD^2 (diagonals removed inside)
            loss = mmd2_unbiased(Kxx, Kyy, Kxy)

            # --- Metrics (detach so they don't grow the graph) ---
            # Off-diagonal stats (full matrix) OR use Kxy only — choose one convention
            off_mu, off_sd = utility.offdiag_stats(S.detach())  # full matrix version
            # off_mu, off_sd = utility.offdiag_stats(Kxy_det) # cross-block-only version

            # Δ gap from cross block (positives = diag(Kxy), negatives = off-diag(Kxy))
            delta = utility.pos_neg_gap_from_cross(Kxy.detach())

            # (Log/store) tau is not used for MMD; report None or "-"
            tau_used = None

        else:
            raise ValueError(f"Unknown loss_type={loss_type}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())

        batch_s_list.append(time.perf_counter() - t0)

    epoch_time = time.perf_counter() - epoch_t0
    batch_s = sum(batch_s_list) / max(1, len(batch_s_list))
    avg_loss = total_loss / len(dataloader)

    print(f"[QFM] Average batch time: {batch_s:.4f} s")
    print(f"[QFM] Epoch Time: {epoch_time:.4f} s")
    print(f"[QFM] Epoch Loss: {avg_loss:.4f}")

    metrics = {
        "loss_type": loss_type,
        "tau": tau_used,  # float or None
        "sigma2": sigma2_used,  # float or None
        "offdiag_mu": off_mu,  # float
        "offdiag_sd": off_sd,  # float
        "delta": delta,  # float
        "s_per_batch": batch_s,  # float
        "epoch_time_s": epoch_time,  # float
        "avg_loss": avg_loss,  # float
    }
    # one clean printable string
    metrics["pretty"] = (
        f"[{loss_type}] "
        f"loss={avg_loss:.4f}  s/batch={batch_s:.1f}  "
        f"tau={tau_used if tau_used is not None else '-'}  "
        f"sigma2={sigma2_used:.3e}"
        if sigma2_used is not None
        else f"[{loss_type}] loss={avg_loss:.4f}  s/batch={batch_s:.1f}  "
    ) + (
        f"  off_mu={off_mu:.3e}  off_sd={off_sd:.3e}  Δ={delta:.3e}"
        if (off_mu is not None and off_sd is not None and delta is not None)
        else ""
    )

    print(metrics["pretty"])

    return avg_loss, losses


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    losses = []

    # Setup variables for collecting execution time metrics.
    epoch_t0 = time.perf_counter()
    batch_ms_list = []
    for x, _ in dataloader:
        t0 = time.perf_counter()
        x_i, x_j = x
        x_i, x_j = x_i.to(device), x_j.to(device)

        _, z_i = model(x_i)
        _, z_j = model(x_j)

        # # === DIAGNOSTIC: check variance of z ===
        # # We'll check L2 norm and per-dimension variance
        # z_all = torch.cat([z_i, z_j], dim=0)  # shape [2B, D]
        # z_var = z_all.var(dim=0)              # variance per feature dim
        # z_std = z_all.std()                   # overall std across all z
        # z_mean_norm = z_all.norm(dim=1).mean()  # avg L2 norm per sample
        # print(f"z std: {z_std.item():.4f}, mean L2 norm: {z_mean_norm.item():.4f}")
        # # Optionally log these to WandB or file

        loss = criterion(z_i, z_j)
        optimizer.zero_grad()
        loss.backward()

        # === VQC Gradient Diagnostics ===
        # print("")``
        # print("*****DEBUG****")
        # for name, param in model.named_parameters():
        #     if name.startswith("projection_head.0"):
        #         if param.grad is not None:
        #             print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
        #         else:
        #             print(f"{name}: grad is None")

        # print("")
        optimizer.step()

        total_loss += loss.item()
        losses.append(loss.item())

        batch_ms_list.append((time.perf_counter() - t0))

    epoch_time = time.perf_counter() - epoch_t0
    imgs_per_sec = len(dataloader) / epoch_time

    batch_ms = sum(batch_ms_list) / max(1, len(batch_ms_list))

    print(f"Average batch time: {batch_ms:.4f} s")
    print(f"Epoch Time: {epoch_time:.4f} s")
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch Loss: {avg_loss:.4f}")

    return avg_loss, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_feature_map",
        action="store_true",
        help="Use quantum feature map (fidelity) instead of a projection head.",
    )
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--encoding", type=str, default="ry", choices=["ry", "rx"])
    parser.add_argument(
        "--entangle", type=str, default="ring", choices=["ring", "linear"]
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--loss", type=str, default="infonce", choices=["infonce", "mmd"]
    )
    args = parser.parse_args()

    # Device: keep QFM on CPU for stability with PennyLane simulators.
    if args.use_feature_map:
        device = torch.device("cpu")
        print("[QFM] Using CPU for encoder + QNode (recommended for stability).")
        qfm_kwargs = dict(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            encoding=args.encoding,
            entangle=args.entangle,
        )
        model = ContrastiveModel(use_feature_map=True, qfm_kwargs=qfm_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Ensure your dataloader yields ((x_i, x_j), _) and drop_last=True for 2N batches
        loader = get_dataloaders(drop_last=True)
    else:
        # Classical projection-head path (MPS if available, else CPU)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = ContrastiveModel(projection_dim=128, use_feature_map=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loader = get_dataloaders(drop_last=True)
        criterion = InfoNCELoss(temperature=args.temperature)

    all_epoch_losses = []
    print("---- Beginning contrastive training.")
    for epoch in range(args.epochs):
        print(f"---- Epoch {epoch}")
        if args.use_feature_map:
            avg_loss, batch_loss = train_one_epoch_qfm(
                model,
                loader,
                optimizer,
                device,
                tau=args.temperature,
                loss_type=args.loss,
            )
        else:
            avg_loss, batch_loss = train_one_epoch(
                model, loader, optimizer, criterion, device
            )
        all_epoch_losses.append(avg_loss)
        print("")

    # (optional) save losses, checkpoint, etc.
    torch.save(model.state_dict(), "contrastive_model.pth")
    # torch.save({"model": model.state_dict(), "losses": all_epoch_losses}, "ckpt.pt")

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
