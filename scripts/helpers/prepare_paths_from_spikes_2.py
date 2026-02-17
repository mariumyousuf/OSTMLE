import numpy as np

def prepare_paths_from_spikes_2(data, T):
    """
    Convert a binary spike matrix (N x L) into paths (K, T+1, N)
    for one-step logistic model training.

    Args:
        data : np.ndarray of shape (N, L), binary {0,1} spikes.
        T    : int, number of time steps per trajectory (excluding initial state).

    Returns:
        paths : np.ndarray of shape (K, T+1, N), spins {-1,+1}.
    """
    N, L = data.shape
    spins = 2 * data - 1   # convert to {-1, +1}
    K = L - T
    if K <= 0:
        raise ValueError(f"T={T} too long for data length {L}")
    paths = np.empty((K, T+1, N), dtype=np.int8)
    for k in range(K):
        paths[k] = spins[:, k : k + T + 1].T
    return paths