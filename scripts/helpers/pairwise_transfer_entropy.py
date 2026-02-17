import numpy as np
import time

def transfer_entropy_binary(source, target, max_lag=1):
    """
    Compute transfer entropy TE(source -> target) for binary time series
    with history length up to max_lag (lagged TE).

    Parameters
    ----------
    source : array-like, shape (T,)
        Binary time series of source neuron
    target : array-like, shape (T,)
        Binary time series of target neuron
    max_lag : int
        Number of past bins to include for each neuron

    Returns
    -------
    te : float
        Transfer entropy in bits
    """
    T = len(source)
    eps = 1e-12
    te = 0.0

    counts_xyz = {}
    counts_xy = {}
    counts_xhist = {}

    for t in range(max_lag, T-1):
        x_hist = tuple(target[t - l] for l in range(max_lag))
        y_hist = tuple(source[t - l] for l in range(max_lag))
        x_next = target[t + 1]

        key_xyz = (x_next, x_hist, y_hist)
        key_xy = (x_hist, y_hist)
        key_xhist = (x_next, x_hist)

        counts_xyz[key_xyz] = counts_xyz.get(key_xyz, 0) + 1
        counts_xy[key_xy] = counts_xy.get(key_xy, 0) + 1
        counts_xhist[key_xhist] = counts_xhist.get(key_xhist, 0) + 1

    total = T - 1 - max_lag
    for key_xyz, n_xyz in counts_xyz.items():
        x_next, x_hist, y_hist = key_xyz
        p_xyz = n_xyz / total

        n_xy = counts_xy[(x_hist, y_hist)]
        n_xhist = sum(counts_xhist.get((x_next_, x_hist), 0) for x_next_ in (0,1))

        p_x_given_xy = n_xyz / (n_xy + eps)
        p_x_given_xhist = counts_xhist[(x_next, x_hist)] / (n_xhist + eps)

        te += p_xyz * np.log2((p_x_given_xy + eps) / (p_x_given_xhist + eps))

    return te


def pairwise_transfer_entropy(spikes, max_lag=1):
    """
    Compute N x N matrix of pairwise transfer entropy for binary spikes.

    Parameters
    ----------
    spikes : ndarray, shape (N, T)
        Binary spike trains (N neurons, T time bins)
    max_lag : int
        Number of past bins to include

    Returns
    -------
    TE : ndarray, shape (N, N)
        TE[i, j] = transfer entropy i -> j
    """
    N, T = spikes.shape
    TE = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                TE[i, j] = transfer_entropy_binary(spikes[j], spikes[i], max_lag=max_lag)
    return TE
