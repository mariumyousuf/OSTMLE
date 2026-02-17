import numpy as np

def generate_graph(N, graph_type, w_val):
    np.random.seed(42)
    """
    Generate an adjacency matrix for a specific motif type.

    Parameters:
        N (int): number of nodes in the motif
        graph_type (str): type of motif
        w_val (float): placeholder weight (will be overwritten later)
    
    Returns:
        W (N x N np.array): adjacency matrix (0/1)
    """
    W = np.zeros((N, N))

    if graph_type == "chain":
        # Linear feedforward chain: 0->1->2->...
        for i in range(1, N):
            W[i, i-1] = w_val

    # elif graph_type == "chain_branch":
    #     # Linear chain with one branching edge from node 1 to last node
    #     for i in range(1, N):
    #         W[i, i-1] = w_val
    #     if N >= 3:
    #         W[1, N-1] = w_val  # branch from 2nd node to last node

    elif graph_type == "chain_branch":
        # Chain with sparse branching
        for i in range(1, N):
            W[i, i-1] = w_val

        if N >= 3:
            num_branches = max(1, int(np.sqrt(N)))
            branch_sources = np.random.choice(range(N - 2), size=num_branches, replace=False)
            for src in branch_sources:
                W[N - 1, src] = w_val

    elif graph_type == "feedforward_loop":
        # Cascading local feedforward loops
        if N < 3:
            raise ValueError("Feedforward loop requires at least 3 nodes")

        # backbone chain
        for i in range(1, N):
            W[i, i - 1] = w_val

        # local feedforward shortcuts (i -> i+2)
        for i in range(N - 2):
            if np.random.rand() < 0.3:  # sparse reciprocity
                W[i + 2, i] = w_val

    elif graph_type == "cycle":
        for i in range(N):
            src = (i+1) % N
            W[src, i] = w_val

    elif graph_type == "cycle_chord":
        if N > 3:
            for i in range(N):
                src = (i+1) % N
                W[src, i] = w_val
            chord_from = N // 2
            chord_to = 0
            W[chord_from, chord_to] = w_val
        else:
            raise ValueError("Cycle + chord requires N > 3")

    # elif graph_type == "reciprocal_cycle":
    #     # 3–4 node cycle with bidirectional edges
    #     if N < 3:
    #         raise ValueError("Reciprocal cycle requires at least 3 nodes")
    #     for i in range(N):
    #         src = (i+1) % N
    #         W[src, i] = w_val     # forward edge
    #         W[i, src] = w_val     # reciprocal edge

    elif graph_type == "reciprocal_cycle":
        if N < 3:
            raise ValueError("Reciprocal cycle requires at least 3 nodes")

        for i in range(N):
            j = (i + 1) % N
            W[j, i] = w_val      # forward edge
            # W[i, j] = w_val      # backward edge
            if np.random.rand() < 0.3:  # sparse reciprocity
                W[i, j] = w_val # backward edge


    elif graph_type == "inward_fork":
        center = N // 2
        for i in range(N):
            if i != center:
                W[center, i] = w_val

    elif graph_type == "inward_fork_half":
        center = N // 2
        first_half_indices = [i for i in range(center)]
        for i in first_half_indices:
            W[center, i] = w_val

    elif graph_type == "inward_fork_14":
        center = N // 4
        first_half_indices = [i for i in range(center)]
        for i in first_half_indices:
            W[center, i] = w_val

    elif graph_type == "inward_fork_alt":
        center = N // 2
        alt_indices = list(range(0, N, 2))
        for i in alt_indices:
            W[center, i] = w_val

    elif graph_type == "single_edge":
        if N >= 2:
            W[1, 0] = w_val

    elif graph_type == "outward_fork":
        center = N // 2
        for i in range(N):
            if i != center:
                W[i, center] = w_val

    elif graph_type == "empty":
        pass  # all zeros

    elif graph_type == "dag":
        # simple forward DAG
        for i in range(N-1):
            W[i+1, i] = w_val  # chain: 0->1->2->...
        
        # optional: add extra forward edges (skip connections) while preserving acyclicity
        for i in range(N-2):
            for j in range(i+2, N):
                W[j, i] = w_val  # edge from node i -> node j

    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return W