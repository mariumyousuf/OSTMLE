import numpy as np

def generate_motifs(N, motif_probs=None, seed=42):
    if seed is not None:
        np.random.seed(seed)

    if motif_probs is None:
        motif_probs = {
            "chain": 0.08,
            "chain_branch": 0.08,
            "feedforward_loop": 0.12,
            "cycle": 0.20,
            "cycle_chord": 0.10,
            "reciprocal_cycle": 0.08,
            "inward_fork": 0.30,
            "outward_fork": 0.25,
            "empty": 0.4,
        }

    # Minimum nodes required per motif type
    motif_min_size = {
        "chain": 3,
        "chain_branch": 3,
        "feedforward_loop": 3,
        "cycle": 3,
        "cycle_chord": 4,
        "reciprocal_cycle": 3,
        "inward_fork": 3,
        "outward_fork": 3,
        "empty": 1,
    }
    
    max_motif_size=int(0.6*N)

    motifs = []
    used_nodes = set()

    # 1. Place empty/silent neurons first
    num_empty = int(motif_probs.get("empty", 0) * N)
    if num_empty > 0:
        empty_nodes = np.random.choice(
            [i for i in range(N) if i not in used_nodes],
            size=num_empty,
            replace=False
        )
        motifs.append({"type": "empty", "nodes": empty_nodes})
        used_nodes.update(empty_nodes)

    # 2. Place other motifs
    total_motifs = max(1, N // 2)
    motif_types = [m for m in motif_probs.keys() if m != "empty"]

    motif_type_list = []
    for m in motif_types:
        n_motifs = max(1, int(motif_probs[m] * total_motifs))
        motif_type_list.extend([m] * n_motifs)
    np.random.shuffle(motif_type_list)

    for mtype in motif_type_list:
        min_size = motif_min_size[mtype]
        size = np.random.randint(min_size, max(min_size+1, max_motif_size+1))

        available_nodes = [i for i in range(N) if i not in used_nodes]

        if len(available_nodes) >= size:
            nodes = np.random.choice(available_nodes, size=size, replace=False)
        else:
            # Not enough free nodes: allow overlaps to satisfy minimum size
            nodes = np.random.choice(range(N), size=size, replace=False)

        motifs.append({"type": mtype, "nodes": [int(n) for n in nodes]})
        used_nodes.update(nodes)
    
    return motifs