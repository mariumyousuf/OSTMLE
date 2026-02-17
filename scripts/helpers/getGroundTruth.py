import numpy as np
import pandas as pd

def getGroundTruth(fn):
    """
    Loads and processes ground truth spike data from a text file.

    Assumes the file contains space-separated spike trains as strings,
    one per line. Skips the first two lines and binarizes the result.

    Parameters:
    -----------
    fn : str
        Path to the ground truth file.

    Returns:
    --------
    gt : 2D numpy array
    """
    data = pd.read_table(fn, header=None).to_numpy()
    gt = []
    for i in range(np.shape(data)[0]):
        string = data[i][0]
        d = np.fromstring(string, dtype=float, sep=' ')
        gt.append(d)
    gt = np.array(gt[2:][:])
    return gt