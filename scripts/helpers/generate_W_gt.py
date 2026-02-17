import numpy as np

from .generate_graph import generate_graph

def generate_W_gt(
    N,
    motifs,
    w_mean,
    w_std,
    seed=42,
):
    """
    Generate a ground-truth CA3-style network for identifiability testing.

    Motif edges use the specified w_mean/w_std. 
    Background (non-motif) edges are weaker using non_motif_scale and non_motif_std_scale.

    Parameters:
        N (int): number of neurons
        motifs (list of dict): each dict has 'type' and 'nodes'
        w_mean (float): mean weight for motif edges
        w_std (float): std dev for motif edges
        seed (int or None): random seed

    Returns:
        W_gt (N x N np.array): ground-truth weighted adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Initialize adjacency matrix
    # ------------------------------------------------------------------
    W = np.zeros((N, N))

    # Identify silent neurons
    silent_nodes = set()
    for motif in motifs:
        if motif["type"] == "empty":
            silent_nodes.update(motif["nodes"])

    # ------------------------------------------------------------------
    # Insert motifs
    # ------------------------------------------------------------------
    for motif in motifs:
        nodes = np.asarray(motif["nodes"])
        mtype = motif["type"]

        if mtype == "empty":
            continue

        sigma_m = np.sqrt(np.log(1 + (w_std**2) / (w_mean**2)))
        mu_m = np.log(w_mean) - 0.5 * sigma_m**2
        
        # Generate motif topology (binary)
        motif_topology = generate_graph(len(nodes), mtype, w_val=1.0)

        # Fill weights for motif edges
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if motif_topology[i, j] != 0:
                    W[nodes[i], nodes[j]] = np.random.lognormal(mu_m, sigma_m)

    # ------------------------------------------------------------------
    # Insert sparse background edges
    # ------------------------------------------------------------------
    candidates = [
        (i, j)
        for i in range(N)
        for j in range(N)
        if i != j and W[i, j] == 0 and i not in silent_nodes and j not in silent_nodes
    ]

    # ------------------------------------------------------------------
    # Enforce silent neurons and remove self-loops
    # ------------------------------------------------------------------
    if silent_nodes:
        silent_nodes = list(silent_nodes)
        W[silent_nodes, :] = 0.0
        W[:, silent_nodes] = 0.0

    np.fill_diagonal(W, 0.0)

    return W
