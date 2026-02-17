import numpy as np

def generate_W_random_CA3(
    N,
    connection_frac,
    w_mean,
    w_std,
    seed=42,
):
    """
    Generate a random sparse CA3-style recurrent weight matrix.

    A fraction (connection_frac) of all possible directed connections
    (excluding self-loops) are assigned nonzero weights drawn from a
    lognormal distribution. Diagonal entries are zero.

    Parameters:
        N (int): number of neurons
        connection_frac (float): fraction of possible connections (0–1)
        w_mean (float): mean of synaptic weights
        w_std (float): std dev of synaptic weights
        seed (int or None): random seed

    Returns:
        W (N x N np.array): weighted adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize weight matrix
    W = np.zeros((N, N))

    # Total possible directed edges (no self-loops)
    possible_edges = [(i, j) for i in range(N) for j in range(N) if i != j]
    n_edges = int(connection_frac * len(possible_edges))

    # Randomly select edges
    selected_edges = np.random.choice(
        len(possible_edges),
        size=n_edges,
        replace=False
    )

    # Lognormal distribution parameters
    sigma = np.sqrt(np.log(1 + (w_std**2) / (w_mean**2)))
    mu = np.log(w_mean) - 0.5 * sigma**2

    # Assign weights to selected edges
    for idx in selected_edges:
        i, j = possible_edges[idx]
        W[i, j] = np.random.lognormal(mu, sigma)

    # Enforce no self-loops
    np.fill_diagonal(W, 0.0)

    return W
