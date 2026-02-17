import numpy as np

# def getDiscretizedSpikeMatrix(spike_times, tstop, N, window, binarize=True):
#     """
#     Converts spike times into a binarized/binned spike matrix.

#     Parameters:
#         spike_times (list of lists): Each sublist contains spike times (in ms) for one neuron.
#         tstop (float): Total simulation time in seconds.
#         N (int): Number of neurons.
#         window (int): Time bin size in milliseconds.

#     Returns:
#         np.ndarray: Binary/Binned matrix of shape (N, num_bins), where 1 indicates at least one spike
#                     occurred in the corresponding time bin.
#     """
#     timeline = tstop*1000 # tstop is in s and timeline is in ms
#     discList = np.arange(0, timeline+1, window) 
#     T=np.zeros((N,len(discList)))
#     for n, spikes in enumerate(spike_times):
#         for s in spikes:
#             i = bisect.bisect_left(discList, s)
#             if binarize == True:
#                 T[n][i-1]=1 # multiple fires within the window represented with a 1 
#             else:
#                 T[n][i-1]+=1 # count of fires within the window 
#     return np.array(T)

def getDiscretizedSpikeMatrix(spike_times, tstop, N, window, binarize=True):
    """
    Converts spike times into a binarized or binned spike matrix using vectorized NumPy operations.

    Parameters:
        spike_times (list of lists): Each sublist contains spike times (in ms) for one neuron.
        tstop (float): Total simulation time in seconds.
        N (int): Number of neurons.
        window (int): Time bin size in milliseconds.
        binarize (bool): If True, output is binary (0/1). If False, output is spike counts.

    Returns:
        np.ndarray: Matrix of shape (N, num_bins), where each entry is either 1 (spike occurred) 
                    or the count of spikes in that bin.
    """
    timeline = tstop * 1000  # Convert tstop to milliseconds
    num_bins = int(np.ceil(timeline / window))
    T = np.zeros((N, num_bins), dtype=int)

    for n, spikes in enumerate(spike_times):
        if len(spikes) == 0:
            continue
        spikes = np.array(spikes)
        # Compute bin indices for each spike
        bins = (spikes // window).astype(int)
        # Clip bins in case a spike is exactly at tstop
        bins = np.clip(bins, 0, num_bins - 1)
        if binarize:
            T[n, bins] = 1
        else:
            # Count spikes per bin
            counts = np.bincount(bins, minlength=num_bins)
            T[n] += counts

    return T
