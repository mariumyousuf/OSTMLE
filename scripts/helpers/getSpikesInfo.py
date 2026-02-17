import numpy as np
import pandas as pd

def getSpikesInfo(fn):
    """
    Reads the spike train data saved from NEURON
    Takes in the filename (fn) and reads it as a multi-size numpy array
    
    Returns:
    --------
        the number of neurons (int), 
        the number of spikes per neuron (list of int),
        the spike times for each neuron (list of lists of floats)
    """
    data = pd.read_table(fn, header=None).to_numpy()
    numSpikes = []
    spikeTimes = []
    for i in range(np.shape(data)[0]):
        string = data[i][0]
        d = np.fromstring(string, dtype=float, sep=' ')
        numSpikes.append(int(d[0]))
        spikeTimes.append(d[1:])
    spikeTimes = spikeTimes[1:]
    numNeurons = numSpikes[0]
    numSpikes = numSpikes[1:]
    
    return numSpikes, spikeTimes