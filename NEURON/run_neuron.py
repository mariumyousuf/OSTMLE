import sys
import os
from neuron import h, gui

# --- Read arguments ---
N = int(sys.argv[1])
duration_in_s = float(sys.argv[2])
gt_path = sys.argv[3]
rasterplot_path = sys.argv[4]
numSamples = int(sys.argv[5])

# --- Load NEURON mechanisms ---
h.nrn_load_dll("modfiles/arm64/libnrnmech.dylib")

# --- Load parameters ---
h.load_file("params.hoc")

# Override HOC globals
h.numneurons = N
h.tstop = duration_in_s * 1000

print("Running NEURON with:")
print(f"  N = {h.numneurons}")
print(f"  tstop = {h.tstop}")
print(f"  numSamples = {numSamples}")

# --- Build network ---
h.load_file("network.hoc")

# --- Save output ---
h.saveSpikeData(gt_path, rasterplot_path, h.numneurons, numSamples)

# --- Run simulation ---
h.go()
