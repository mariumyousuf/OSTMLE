import numpy as np
from scipy.signal import correlate, correlation_lags

# def getCorrDirect(data, N, h):
#     """
#     Computes a directional cross-correlation matrix based on peak correlation lag.

#     Parameters:
#         data (list or array-like): List of N time series (1D arrays), one per neuron or signal.
#         N (int): Number of time series (neurons).
#         h (int): Maximum allowed lag (in time steps) to consider a valid correlation.

#     Returns:
#         np.ndarray: Asymmetric NxN matrix where entry (i, j) is nonzero if the peak 
#                     cross-correlation between neuron i and j occurs with a lag ≤ h. 
#                     Direction is encoded such that the leading neuron (earlier spike) 
#                     points to the lagging neuron.
#     """

#     C_MX = np.zeros((N, N))
#     for i in range(N):
#         cell1 = data[i]
#         for j in range(i+1, N):
#             if i != j:
#                 cell2 = data[j]
#                 corr = correlate(cell1, cell2)
#                 lags = correlation_lags(len(cell1), len(cell2))
#                 max_lag = lags[np.argmax(corr)]
#                 if abs(max_lag) <= h and abs(max_lag)!=0:
#                     if max_lag > 0:
#                         C_MX[i][j] = abs(max_lag)
#                     else:
#                         C_MX[j][i] = abs(max_lag)
    
#     return C_MX


def getCorrDirect(data, N, h):
    """
    Computes a directional cross-correlation matrix based on peak correlation magnitude
    within a maximum lag window. Direction is determined by which neuron leads.

    Parameters:
        data (list or array-like): List of N time series (1D arrays), one per neuron/signal.
        N (int): Number of time series (neurons).
        h (int): Maximum allowed lag (in time steps) to consider a valid directional connection.

    Returns:
        np.ndarray: Asymmetric NxN matrix where entry (i, j) represents the strength
                    of influence from neuron i to neuron j based on peak cross-correlation.
                    Entries are zero if no valid connection is detected.
    """
    C_MX = np.zeros((N, N))

    for i in range(N):
        x = data[i]
        for j in range(i + 1, N):
            if i == j:
                continue
            y = data[j]
            
            if np.var(x) == 0 or np.var(y) == 0:
                continue
            
            # Full cross-correlation
            corr = correlate(x, y)
            lags = correlation_lags(len(x), len(y))
            
            # Peak correlation and its lag
            idx_max = np.argmax(np.abs(corr))   # peak magnitude
            peak_corr = corr[idx_max]
            max_lag = lags[idx_max]
            
            # Only consider connections within allowed lag window (exclude zero lag)
            if abs(max_lag) <= h and max_lag != 0:
                if max_lag > 0:
                    # i leads j (i -> j)
                    C_MX[i, j] = abs(peak_corr)
                else:
                    # j leads i (j -> i)
                    C_MX[j, i] = abs(peak_corr)

    return C_MX