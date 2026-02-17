import io
import contextlib
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import statsmodels.api as sm

def silent_granger_test(data, maxlag, addconst=True, **kwargs):
    """
    Runs the Granger causality test silently (without console output)
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return grangercausalitytests(data, maxlag, addconst=addconst, **kwargs)

def compute_granger_matrix(data, max_lag, test='lrtest'):
    """
    Computes a Granger causality matrix between all pairs of binary time series.

    Parameters:
        data (array-like): 2D array of shape (N, T), where N is the number of binary variables 
                           (e.g., neurons) and T is the number of time steps.
        max_lag (int): Maximum lag to use in Granger causality testing.
        test (str): Test statistic to extract from the Granger test result. 
                    For binary data, 'lrtest' (likelihood ratio test) is recommended 
                    due to its suitability for discrete outcomes.

    Returns:
        np.ndarray: Normalized N x N matrix of Granger causality test statistics.
                    Entry (i, j) represents the strength with which time series j 
                    Granger-causes i.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    N = np.shape(data)[0]
    df = pd.DataFrame(data).T
    gc_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                result = silent_granger_test(df[[i, j]], [max_lag])#grangercausalitytests(df[[i, j]], [max_lag], verbose=False)
                val = result[max_lag][0][test][0]#result[max_lag][0][test][1]
                gc_matrix[i, j] = val
    
#     if sig_threshold is not None:
#         gc_matrix = (gc_matrix < sig_threshold).astype(int)
        
    return (gc_matrix - np.min(gc_matrix)) / (np.max(gc_matrix) - np.min(gc_matrix))

# def point_process_granger(spikes, history_lags=1):
#     """
#     Reference: 
#     @article{kim2011granger,
#       title={A Granger causality measure for point process models of ensemble neural spiking activity},
#       author={Kim, Sanggyun and Putrino, David and Ghosh, Soumya and Brown, Emery N},
#       journal={PLoS computational biology},
#       volume={7},
#       number={3},
#       pages={e1001110},
#       year={2011},
#       publisher={Public Library of Science San Francisco, USA}
#     }
    
#     Compute N x N directed influence (Granger causality) matrix from binary spike trains.

#     Parameters
#     ----------
#     spikes : np.ndarray
#         Binary spike train array of shape (N, T), N neurons, T time bins.
#     history_lags : int
#         Number of past bins to include as history covariates.

#     Returns
#     -------
#     GC : np.ndarray
#         Directed influence matrix of shape (N, N). GC[i, j] indicates
#         influence from neuron j (source) to neuron i (target).
#     """

#     N, T = spikes.shape

#     # Build history design matrix: T x (N*history_lags)
#     X_hist = []
#     for lag in range(1, history_lags + 1):
#         X_hist.append(np.roll(spikes, lag, axis=1).T)  # T x N
#     X_hist = np.hstack(X_hist)  # T x (N*history_lags)

#     # Because rolling wraps around, zero out first `history_lags` rows
#     X_hist[:history_lags, :] = 0

#     GC = np.zeros((N, N))

#     for target in range(N):
#         y = spikes[target, :]  # target spike train

#         # Fit full model
#         X_full = sm.add_constant(X_hist)
#         full_model = sm.GLM(y, X_full, family=sm.families.Poisson())
#         full_result = full_model.fit()
#         L_full = full_result.llf

#         # Fit reduced models by removing each source neuron
#         for source in range(N):
#             # Indices of covariates excluding source neuron history
#             cols_to_keep = [
#                 k for k in range(N * history_lags) if (k % N) != source
#             ]
#             X_red = sm.add_constant(X_hist[:, cols_to_keep])
#             red_model = sm.GLM(y, X_red, family=sm.families.Poisson())
#             red_result = red_model.fit()
#             L_red = red_result.llf

#             # Log-likelihood ratio (Granger causality)
#             GC[target, source] = 2 * (L_full - L_red)

#     return GC

import numpy as np
import statsmodels.api as sm

def point_process_granger(spikes, history_lags=1):
    """
    Scalable Poisson-GLM Granger causality for point processes.
    Conceptually equivalent to likelihood-ratio GC, but uses Wald tests.

    Parameters
    ----------
    spikes : np.ndarray, shape (N, T)
        Binary spike trains
    history_lags : int
        Number of history lags

    Returns
    -------
    GC : np.ndarray, shape (N, N)
        GC[i, j] = influence from neuron j to neuron i
    """

    N, T = spikes.shape
    L = history_lags

    # ----- Build history design matrix -----
    X_hist = np.zeros((T, N * L))
    for lag in range(1, L + 1):
        X_hist[:, (lag-1)*N : lag*N] = np.roll(spikes, lag, axis=1).T

    X_hist[:L, :] = 0  # fix wraparound
    X = sm.add_constant(X_hist)

    GC = np.zeros((N, N))

    # ----- One GLM per target neuron -----
    for target in range(N):
        y = spikes[target]

        model = sm.GLM(y, X, family=sm.families.Poisson())
        res = model.fit()

        beta = res.params[1:]        # drop intercept
        cov = res.cov_params()[1:, 1:]

        # ----- Wald tests per source neuron -----
        for source in range(N):
            idx = np.arange(source, N * L, N)  # all lags of this source

            b = beta[idx]
            C = cov[np.ix_(idx, idx)]

            # Numerical safety
            if np.linalg.matrix_rank(C) < L:
                GC[target, source] = 0.0
                continue

            GC[target, source] = b.T @ np.linalg.inv(C) @ b

    return GC
