import numpy as np

def getAvgFiringRateSourceTarget(W_gt, num_spikes, duration_in_s):
    """
    Compute average firing rates separately for source and target neurons.

    Parameters:
    - W_gt : np.ndarray, shape (N, N)
        Connection matrix. W_gt[i,j] is weight from neuron j -> neuron i
    - num_spikes : list or np.ndarray, shape (N,)
        Number of spikes for each neuron
    - duration_in_s : float
        Duration of recording in seconds

    Returns:
    - fr : np.ndarray, shape (N,)
        Firing rates for all neurons
    - avg_fr_source : float
        Average firing rate of source neurons (neurons with outgoing connections)
    - avg_fr_target : float
        Average firing rate of target neurons (neurons with incoming connections)
    - source_mask : np.ndarray, shape (N,), bool
        Mask indicating source neurons
    - target_mask : np.ndarray, shape (N,), bool
        Mask indicating target neurons
    """

    N = np.shape(W_gt)[0]
    
    fr = np.asarray(num_spikes) / duration_in_s

    # Source neurons: have at least one outgoing connection
    source_mask = (W_gt.sum(axis=0) > 0)
    num_source_per_N = np.sum(source_mask)/N

    # Target neurons: have at least one incoming connection
    target_mask = (W_gt.sum(axis=1) > 0)
    num_target_per_N = np.sum(target_mask)/N

    # Compute averages
    avg_fr_source = fr[source_mask].mean() if source_mask.any() else np.nan
    avg_fr_target = fr[target_mask].mean() if target_mask.any() else np.nan

    return fr, avg_fr_source, avg_fr_target


def getAvgFiringRate(W_gt, num_spikes, duration_in_s):
    """
    Average firing rates for connected vs disconnected neurons.
    Connected = has at least one incoming OR outgoing edge.
    """

    fr = np.asarray(num_spikes) / duration_in_s

    A = (W_gt != 0).astype(int)

    # connected if any incoming or outgoing edge
    conn_mask = (A.sum(axis=0) + A.sum(axis=1)) > 0
    disconn_mask = ~conn_mask

    avg_fr_conn = fr[conn_mask].mean() if conn_mask.any() else np.nan
    avg_fr_disconn = fr[disconn_mask].mean() if disconn_mask.any() else np.nan

    return fr, avg_fr_conn, avg_fr_disconn, conn_mask, disconn_mask 