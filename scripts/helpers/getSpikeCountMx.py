import numpy as np

def getSpikeCountMx(h, N, spikeTimes):
    """
    Computes a spike count matrix based on the number of spikes in one spike train 
    that occur within a time window h after spikes in another spike train.

    The function calculates, for each pair of neurons (i, j), how many spikes from neuron `j`
    occurred within h time units *before* each spike of neuron i. The diagonal elements
    (self-comparisons) are excluded and set to zero.

    Parameters:
    -----------
    h : float
        The time window (in the same units as spike times) within which to count spikes.
        Only spike pairs where the time difference falls in [0, h] are counted.
    
    N : int
        The number of neurons or spike trains.

    spikeTimes : list of 1D numpy arrays
        A list where each element is a sorted array of spike times for a neuron.

    Returns:
    --------
    X : 2D numpy array of shape (N, N)
        A matrix where element X[i, j] represents the number of spikes in neuron j
        that occurred within h time units before each spike in neuron i.
        Diagonal elements X[i, i] are zero.
    """
    X = np.zeros((N, N), dtype=int)
    for i in range(N):
        t_times = np.array(spikeTimes[i])
        for j in range(N):
            if i == j:
                continue
            s_times = np.array(spikeTimes[j])
            # Calculate all pairwise differences
            diffs = t_times[:, None] - s_times[None, :]
            # Count the number of differences within the range [0, h]
            count = np.sum((diffs >= 0) & (diffs <= h))
            X[i, j] = count
    return X