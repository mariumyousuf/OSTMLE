# ================================================================
# Two-phase OST-MLE (prune-then-refit) with hard mask
# Phase 1: one-step NLL + L1 on off-diagonal W, diag forced to 0
# Threshold -> sparse mask (W>=tau kept)
# Phase 2: one-step NLL only, using fixed sparse mask (pruned weights stay 0)
# ================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#--------------- helpers ----------------
def device_auto(prefer_cuda=True):
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")

def spin_zero(N):  # (-1,...,-1)
    return -np.ones(N, dtype=np.int8)

@torch.no_grad()
def simulate_paths_factorized(beta, W, K, T, s0, seed=123):
    rng = np.random.default_rng(seed)
    N = beta.shape[0]
    paths = np.empty((K, T + 1, N), dtype=np.int8)
    paths[:, 0, :] = s0[None, :]
    for k in range(K):
        s = s0.copy()
        for t in range(T):
            logits = beta + (W @ s)          # (N,)
            p = 1.0 / (1.0 + np.exp(-logits))
            s = np.where(rng.random(N) < p, 1, -1).astype(np.int8)
            paths[k, t + 1, :] = s
    return paths

def make_one_step_pairs(paths):
    K, Tp1, N = paths.shape
    T = Tp1 - 1
    s_t   = paths[:, :-1, :].reshape(-1, N) #all time steps except the last for every path → shape (K, T, N)
    s_tp1 = paths[:,  1:, :].reshape(-1, N) #all time steps except the first for every path → shape (K, T, N)
    return s_t, s_tp1

# --------------- model -------------------
class FactorizedLogisticPrior(nn.Module):
    """
    p(s_{t+1,n}=+1 | s_t) = sigmoid(beta_n + sum_m W_{n,m} * s_t[m]).
    Forward always applies a fixed mask (1=active, 0=pruned). Diagonal forced 0.
    """
    def __init__(self, N, mask=None):
        super().__init__()
        self.N = N
        self.beta = nn.Parameter(torch.zeros(N))
        self.W = nn.Parameter(torch.zeros(N, N))
        if mask is None:
            m = torch.ones(N, N)
        else:
            m = torch.as_tensor(mask, dtype=torch.float32).clone()
        m.fill_diagonal_(0.0)   # forbid self-edges
        self.register_buffer("mask", m)

    def logits(self, s_t):  # s_t: (B,N)
        # forward uses masked W only
        return self.beta + s_t @ ( (self.W * self.mask).T )

# --------------- loss --------------------
def one_step_nll_from_logits(logits, s_tp1):
    y = (s_tp1 > 0).float()
    return F.binary_cross_entropy_with_logits(logits, y, reduction='none').sum(dim=1).mean()

def one_step_nll(model, s_t, s_tp1, batch_idx):
    logits = model.logits(s_t[batch_idx])
    return one_step_nll_from_logits(logits, s_tp1[batch_idx])

# --------------- training (one phase) ----
def train_one_phase(model, s_t, s_tp1, *, steps=1000, batch_size=1024, lr=5e-3,
                    l1_weight=0.0, hard_mask_freeze=False, seed=0):
    """
    Minimizes one-step NLL + (optional) L1 on masked W.
    Always enforces W*=mask and diag=0 after each step.
    If hard_mask_freeze=True, also zeros gradients for pruned entries each iter.
    """
    device = model.beta.device
    B_all = s_t.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    hist = []
    for it in range(steps):
        opt.zero_grad()
        idx = rng.integers(0, B_all, size=min(batch_size, B_all))
        loss_data = one_step_nll(model, s_t, s_tp1, idx)
        if l1_weight > 0.0:
            # L1 only on active (masked) entries
            loss = loss_data + l1_weight * torch.abs(model.W * model.mask).sum()
        else:
            loss = loss_data
        loss.backward()

        # Optional: zero gradients on pruned entries so they never update
        if hard_mask_freeze and model.W.grad is not None:
            model.W.grad *= model.mask

        opt.step()

        # Enforce mask & zero diag after step (keeps pruned weights at 0)
        with torch.no_grad():
            model.W.data *= model.mask
            model.W.data.fill_diagonal_(0.0)

        hist.append(float(loss.item()))
    return hist

def OST_MLE_NEURON(
    N,paths,
    # Phase 1
    steps1=2000, lr1=5e-3, batch_size1=1024, lambda_l1=1e-3,
    # Threshold: W_1.mean+(alpha*W_1.std)
    alpha=0.5,
    # Phase 2
    steps2=1500, lr2=5e-3, batch_size2=1024,
    prefer_cuda=True, seed=123
    ):
    """
    alpha determines the threshold intensity above the non-zero mean of first-phase estimation
    """
    
    np.random.seed(seed); torch.manual_seed(seed)
    device = device_auto(prefer_cuda)
    
    s_t_np, s_tp1_np = make_one_step_pairs(paths)
    s_t   = torch.as_tensor(s_t_np,   dtype=torch.float32, device=device)
    s_tp1 = torch.as_tensor(s_tp1_np, dtype=torch.float32, device=device)

    # -------- Phase 1: dense off-diagonal mask, with L1 ----------
    mask_offdiag = np.ones((N, N), dtype=np.float32); np.fill_diagonal(mask_offdiag, 0.0)
    model1 = FactorizedLogisticPrior(N, mask=mask_offdiag).to(device)

    _ = train_one_phase(
        model1, s_t, s_tp1,
        steps=steps1, batch_size=batch_size1, lr=lr1,
        l1_weight=lambda_l1, hard_mask_freeze=False, seed=seed+1
    )

    with torch.no_grad():
        beta_1 = model1.beta.detach().cpu().numpy()
        W_1_full = model1.W.detach().cpu().numpy()
        W_1 = (model1.W * model1.mask).detach().cpu().numpy()  # masked (diag=0)

    # ---- Threshold to build sparse mask (absolute value) ----
    nonzero_mean = W_1[W_1 != 0].mean()
    nonzero_std = W_1[W_1 != 0].std()
    tau = nonzero_mean+(alpha*nonzero_std) 
    edge_mask = (W_1 >= float(tau)).astype(np.float32)
    np.fill_diagonal(edge_mask, 0.0)
    kept = int(edge_mask.sum())

    # -------- Phase 2: fixed sparse mask, HARD-enforced ----------
    model2 = FactorizedLogisticPrior(N, mask=edge_mask).to(device)
    with torch.no_grad():
        model2.beta.copy_(torch.tensor(beta_1, dtype=torch.float32, device=device))
        # initialize W with masked W_1 (pruned entries exactly 0)
        model2.W.copy_(torch.tensor(W_1 * edge_mask, dtype=torch.float32, device=device))

    _ = train_one_phase(
        model2, s_t, s_tp1,
        steps=steps2, batch_size=batch_size2, lr=lr2,
        l1_weight=0.0, hard_mask_freeze=True, seed=seed+2
    )

    with torch.no_grad():
        beta_2 = model2.beta.detach().cpu().numpy()
        W_2 = (model2.W * model2.mask).detach().cpu().numpy()

    return beta_1, W_1, beta_2, W_2, tau

def OST_MLE_NEURON_2(
    N,paths,
    # Phase 1
    steps1=2000, lr1=5e-3, batch_size1=1024, lambda_l1=1e-3,
    # Threshold: 
    tau=0.1,
    # Phase 2
    steps2=1500, lr2=5e-3, batch_size2=1024,
    prefer_cuda=True, seed=123
    ):
    """
    Allows direct passing-in of tau (the threshold on first-phase estimation)
    """
    
    np.random.seed(seed); torch.manual_seed(seed)
    device = device_auto(prefer_cuda)
    
    s_t_np, s_tp1_np = make_one_step_pairs(paths)
    s_t   = torch.as_tensor(s_t_np,   dtype=torch.float32, device=device)
    s_tp1 = torch.as_tensor(s_tp1_np, dtype=torch.float32, device=device)

    # -------- Phase 1: dense off-diagonal mask, with L1 ----------
    mask_offdiag = np.ones((N, N), dtype=np.float32); np.fill_diagonal(mask_offdiag, 0.0)
    model1 = FactorizedLogisticPrior(N, mask=mask_offdiag).to(device)

    _ = train_one_phase(
        model1, s_t, s_tp1,
        steps=steps1, batch_size=batch_size1, lr=lr1,
        l1_weight=lambda_l1, hard_mask_freeze=False, seed=seed+1
    )

    with torch.no_grad():
        beta_1 = model1.beta.detach().cpu().numpy()
        W_1_full = model1.W.detach().cpu().numpy()
        W_1 = (model1.W * model1.mask).detach().cpu().numpy()  # masked (diag=0)

    # ---- Threshold to build sparse mask (absolute value) ----
    edge_mask = (W_1 >= float(tau)).astype(np.float32)
    np.fill_diagonal(edge_mask, 0.0)
    kept = int(edge_mask.sum())

    # -------- Phase 2: fixed sparse mask, HARD-enforced ----------
    model2 = FactorizedLogisticPrior(N, mask=edge_mask).to(device)
    with torch.no_grad():
        model2.beta.copy_(torch.tensor(beta_1, dtype=torch.float32, device=device))
        # initialize W with masked W_1 (pruned entries exactly 0)
        model2.W.copy_(torch.tensor(W_1 * edge_mask, dtype=torch.float32, device=device))

    _ = train_one_phase(
        model2, s_t, s_tp1,
        steps=steps2, batch_size=batch_size2, lr=lr2,
        l1_weight=0.0, hard_mask_freeze=True, seed=seed+2
    )

    with torch.no_grad():
        beta_2 = model2.beta.detach().cpu().numpy()
        W_2 = (model2.W * model2.mask).detach().cpu().numpy()

    return beta_1, W_1, beta_2, W_2, tau