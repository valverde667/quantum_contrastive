import torch


def offdiag_stats(K: torch.Tensor):
    """
    Mean/std of off-diagonal entries of a square kernel matrix K (any size).
    Returns floats (Python) for easy logging.
    """
    N = K.size(0)
    mask_off = ~torch.eye(N, dtype=torch.bool, device=K.device)
    vals = K[mask_off]
    mu = vals.mean().item()
    sd = vals.std(unbiased=False).item()
    return mu, sd


def pos_neg_gap_from_full(K: torch.Tensor):
    """
    Δ = mean(positives) - mean(negatives) for the standard 2B x 2B matrix
    used in InfoNCE with two views. Positives are at column i+B (mod 2B).
    """
    N = K.size(0)  # = 2B
    B = N // 2
    idx = torch.arange(N, device=K.device)
    pos_col = (idx + B) % N

    pos = K[idx, pos_col]
    # negatives: everything except diagonal and the positive index
    neg_mask = torch.ones_like(K, dtype=torch.bool)
    neg_mask[idx, idx] = False
    neg_mask[idx, pos_col] = False
    neg = K[neg_mask]

    pos_mean = pos.mean().item()
    neg_mean = neg.mean().item()
    return pos_mean - neg_mean


def pos_neg_gap_from_cross(Kxy: torch.Tensor) -> float:
    """
    For two-view kernels: Kxy is [B, B] (view1 vs view2).
    Positives = diag(Kxy); negatives = off-diagonals of Kxy.
    Returns Δ = mean(pos) - mean(neg).
    """
    B = Kxy.size(0)
    pos = torch.diag(Kxy)
    neg = Kxy[~torch.eye(B, dtype=torch.bool, device=Kxy.device)]
    return (pos.mean() - neg.mean()).item()


def fs_rbf_from_fidelity(K_fid: torch.Tensor):
    """
    Convert fidelity kernel K_fid (in [0,1]) to an RBF kernel on
    Fubini–Study distance with median-bandwidth heuristic.
    Returns (K_rbf, sigma2).
    """
    K = K_fid.clamp_(0.0, 1.0)
    fs = torch.acos(K.sqrt())  # d_FS in [0, pi/2]
    # median heuristic on off-diagonals
    N = fs.size(0)
    off_mask = ~torch.eye(N, dtype=torch.bool, device=fs.device)
    d = fs[off_mask]
    sigma2 = (torch.median(d) ** 2).clamp_min(1e-6)
    K_rbf = torch.exp(-(fs**2) / sigma2)
    return K_rbf, sigma2.item()


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between two sets of features.
    Args:
        X: [N, D] tensor (e.g., from model 1)
        Y: [N, D] tensor (e.g., from model 2)
    Returns:
        scalar CKA value (float)
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    dot_product_similarity = (X.T @ Y).norm("fro") ** 2
    normalization = (X.T @ X).norm("fro") * (Y.T @ Y).norm("fro")

    return (dot_product_similarity / normalization).item()


def kernel_CKA(K: torch.Tensor, L: torch.Tensor) -> float:
    """
    Kernel CKA between two kernel matrices K and L.
    Assumes K and L are [N, N] symmetric similarity matrices (e.g., cosine kernel).
    Args:
        K: [N, N] kernel matrix (e.g., cosine similarity of features from model 1)
        L: [N, N] kernel matrix (e.g., cosine similarity of features from model 2)
    Returns:
        scalar CKA value (float)
    """

    def center(K):
        n = K.size(0)
        I = torch.eye(n, device=K.device)
        one_n = torch.ones((n, n), device=K.device) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    K_c = center(K)
    L_c = center(L)

    numerator = (K_c * L_c).sum()
    denom = torch.sqrt((K_c * K_c).sum() * (L_c * L_c).sum())

    return (numerator / denom).item()
