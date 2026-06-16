# OSTMLE
OST-MLE: Python implementation of One-Step Transition Maximum Likelihood Estimation method to infer effective connectivity graph motifs from neural replay spike trains. 

The repo currently includes demos for 5-neuron toy examples:
1. demo_notebooks/demo_paramRecov_NEURON.ipynb demos OST-MLE on 
 with data generated from NEURON. The estimated influence matrix is valided against the ground truth connectivity matrix used to generate the raster data. See "../NEURON/NEURON_Demo.ipynb" for how this raster plot was simulated. Figure shown below compares connectivity matrix against the estimated influence matrix in both Phase 1 and Phase 2.

2. demo_notebooks/demo_paramRecov_model.ipynb demos OST-MLE on 
 with data generated from model then used as input into the model. The estimated parameters are then valided against the ground truth. Figure shown below.

3. NEURON/NEURON_Demo.ipynb saves a raster plot for 
 neurons. This raster plot is used as input OST-MLE in the demo notebook saved as "../demo_notebooks/demo_paramRecov_NEURON.ipynb"