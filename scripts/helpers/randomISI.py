import numpy as np

def randomISI(spikeTimes):
    """
    Randomizes the inter-spike intervals (ISIs) within each spike train while preserving
    the overall number of spikes and approximate timing structure.

    The function destroys any precise temporal correlations or spike patterns in the original
    data by shuffling the ISIs (differences between consecutive spike times), to preserve
    the distribution of ISIs and remove their temporal ordering.

    Parameters:
    -----------
    spikeTimes : list of 1D numpy arrays
        Each element in the list represents the spike times of a single neuron (or trial),
        as a sorted array of time values (in milliseconds).

    Returns:
    --------
    newSpikeTrains : list of 1D numpy arrays
        A list of spike trains where each train has the same ISIs as the original,
        but in a randomized order. The first spike time preserved.
    """
    np.random.seed(17)
    newSpikeTrains = []
    for spike in spikeTimes:
        arr = np.diff(spike)
        np.random.shuffle(arr)  # Shuffle differences in place
        newTrains = np.zeros(len(arr) + 1)  # Initialize with zeros, one more than the shuffled array
        newTrains[0] = spike[0]
        # Compute new spike times
        for l in range(1, len(arr)+1):
            newTrains[l] = newTrains[l-1]+arr[l-1]
        newSpikeTrains.append(newTrains)
    # Optionally shuffle the list of spike trains if required
    np.random.shuffle(newSpikeTrains)
    return newSpikeTrains