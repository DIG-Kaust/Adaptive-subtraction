import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

def adaptive_subtraction_qc(data, primaries, gather_num):
    """Plot correlograms of total data gathers and primaries gathers panels to compare periodicity of signals.

               Parameters
               ----------
               data : :obj:`np.ndarray`
                   Total data gathers stored in a 3d array
               primaries : :obj:`np.ndarray`
                   Primaries estimated via adaptive subtraction stored in a 3d array
               gather_num : :obj:`int`
                   Number of common channel gather to do qc on

               Note
               -------
               This function does not return data, just plots the correlograms and stacked traces auto-correlations.

               """
    ns = data.shape[0] # number of shots
    nr, nt = 180, 1950 # number of receivers
    # Reshape total data to be the same as primaries
    data = data[:, :nr, :nt]

    # Create arrays to store correlated traces
    data_corr_traces = np.zeros((ns, nt*2 -1))
    prim_corr_traces = np.zeros((ns, nt*2 -1))

    for i in range(ns):
        data_corr_traces[i] = correlate(data[i,gather_num],data[i,gather_num])
        prim_corr_traces[i] = correlate(primaries[i,gather_num],primaries[i,gather_num])

    # Create a stacked correlated trace
    data_stack = sum(data_corr_traces)/ns
    prim_stack = sum(prim_corr_traces)/ns

    vmax = 0.006 * np.amax(data_corr_traces) ; vmin = -vmax
    xmin = 627; xmax = 1025 ; ymin = 0 ; ymax = nt*2 -1
    fig, axs = plt.subplots(1, 2, figsize=(10,6),sharey=True)
    axs[0].imshow(data_corr_traces.T, aspect='auto',vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax], cmap='gray')
    axs[0].set_title(f'Total data correlogram for rx={gather_num}', fontsize=12)
    axs[0].set_xlabel('Sources', fontsize=12)
    axs[1].imshow(prim_corr_traces.T, aspect='auto', vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax], cmap='gray')
    axs[1].set_title(f'Primaries correlogram for rx={gather_num}', fontsize=12)
    axs[1].set_xlabel('Sources', fontsize=12)

    ymin = np.min(data_stack) ; ymax = np.max(data_stack)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(data_stack)
    axs[0].set_title(f"Auto-correlated stacked trace for data with rx={gather_num}", fontsize=10)
    axs[0].set_ylim(0.1 * ymin, 0.08 * ymax)

    ymin = np.min(prim_stack) ; ymax = np.max(prim_stack)
    axs[1].plot(prim_stack)
    axs[1].set_title(f"Auto-correlated stacked trace for primaries with rx={gather_num}", fontsize=10)
    axs[1].set_ylim(0.1 * ymin, 0.08 * ymax)
    plt.tight_layout()


