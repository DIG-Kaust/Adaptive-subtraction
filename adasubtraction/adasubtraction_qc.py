import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

def adasubtraction_qc(data, primaries, gather_num, dt=0.004, max_amp=0.25):
    """Estimate amplitude of multiples present in total data vs residual multiples in corrected primaries.

               Parameters
               ----------
               data : :obj:`np.ndarray`
                   Total data CCG
               primaries : :obj:`np.ndarray`
                   Estimated primaries CCG  
               gather_num : :obj:`int`
                   Number of gather
               dt : :obj:`float`
                   Time sampling in seconds
               max_amp : :obj:`float`
                   Maximum amplitude to define first arrival
               Note
               -------
               This function plots the correlograms and stacked traces auto-correlations showing the multiples windows.

               """
    ns, nt = data.shape[0], data.shape[1] # shots, receivers, time samples
    
    # Create multiples array to locate multiples arrivals
    multiples = data - primaries

    # Create arrays to store correlated traces
    data_corr_traces = np.zeros((ns, nt*2 -1))
    prim_corr_traces = np.zeros((ns, nt*2 -1))

    for i in range(ns):
        data_corr_traces[i] = correlate(data[i],data[i])
        prim_corr_traces[i] = correlate(primaries[i],primaries[i])

    # Create a stacked correlated trace
    data_stack = sum(data_corr_traces)/ns
    prim_stack = sum(prim_corr_traces)/ns
    
    # Find first primary index over the ccg
    first_prim = abs(primaries[:])>np.max(abs(primaries[:]))*max_amp # First arrival is stronger
    inxs_p = np.zeros(first_prim.shape[0])                          # than 25% of the absolute max
    for i in range(first_prim.shape[0]):
        for j in range(first_prim.shape[1]):
            if first_prim[i,j]==True:
                inxs_p[i] = j
                if j == first_prim.shape[1] - 1:
                    inxs_p[i] = np.nan
                break                
    inxs_avg_p = int(np.nanmean(inxs_p))
    time_prim = inxs_avg_p*dt
    
    # Find first multiple index over the ccg
    first_mult = abs(multiples[:])>np.max(abs(multiples[:]))*max_amp # Boolean array 
    inxs_m = np.full(first_mult.shape[0], np.nan)
    for i in range(first_mult.shape[0]):  # iterate over traces
        for j in range(first_mult.shape[1]): # iterate over samples
            if first_mult[i,j]==True:
                inxs_m[i] = j 
                if j == first_mult.shape[1] - 1:
                    inxs_m[i] = np.nan
                break
    inxs_avg_m = int(np.nanmean(inxs_m))
    time_mult = inxs_avg_m*dt
   
    delta = 0.8 # time window witdh (s)
    shift = delta/2 # only for the second multiple window, so the center of the window doubles (s)
    
    diff = time_mult - time_prim
    diff_2 = diff*2 + shift 
    idx_diff = (inxs_avg_m - inxs_avg_p) + data_stack.shape[0]//2
    idx_diff_2 = (inxs_avg_m - inxs_avg_p)*2 + data_stack.shape[0]//2 + int(shift/dt)

    mult_avg = np.mean(abs(data_stack[idx_diff: int(idx_diff + delta//dt)])   # First output
                       + abs(data_stack[idx_diff_2: int(idx_diff_2 + delta//dt)]))
    prim_avg = np.mean(abs(prim_stack[idx_diff: int(idx_diff + delta//dt)])   # Second output
                       + abs(prim_stack[idx_diff_2: int(idx_diff_2 + delta//dt)]))
    

    # Plot correlogram of total data and primaries
    vmax = 0.006 * np.amax(data_corr_traces) ; vmin = -vmax
    xmin = 627; xmax = 1025 ; ymin = -(nt-1)*dt ; ymax = (nt-1)*dt
    fig1, axs1 = plt.subplots(1, 2, figsize=(10,6),sharey=True)
    axs1[0].imshow(data_corr_traces.T, aspect='auto',vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax], cmap='gray')
    axs1[0].set_ylabel("Time (s)", fontsize=12)
    axs1[0].set_title(f'Total data correlogram for rx={gather_num}', fontsize=12)
    axs1[0].set_xlabel('Sources', fontsize=12)
    axs1[1].imshow(prim_corr_traces.T, aspect='auto', vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax], cmap='gray')
    axs1[1].set_title(f'Primaries correlogram for rx={gather_num}', fontsize=12)
    axs1[1].set_xlabel('Sources', fontsize=12)

    # Find 2 window extents for the total data
    data_max = max(data_stack[idx_diff: int(idx_diff + delta//dt)])
    data_max_2 = max(data_stack[idx_diff_2: int(idx_diff_2 + delta//dt)])
    data_min = min(data_stack[idx_diff: int(idx_diff + delta//dt)])
    data_min_2 = min(data_stack[idx_diff_2: int(idx_diff_2 + delta//dt)])

    xs = [diff, diff, diff+delta, diff+delta, diff]
    xs_2 = [diff_2, diff_2, diff_2+delta, diff_2+delta, diff_2]
    ys = [data_min*1.1, data_max*1.1, data_max*1.1, data_min*1.1, data_min*1.1]
    ys_2 = [data_min_2*1.1, data_max_2*1.1, data_max_2*1.1, data_min_2*1.1, data_min_2*1.1]
 
    # Plot auto-correlation of stacked correlated data
    t = np.linspace(-data_stack.shape[0]//2, data_stack.shape[0]//2, data_stack.shape[0])*dt
    ymin = np.min(data_stack) ; ymax = np.max(data_stack)
    
    fig2, axs2 = plt.subplots(2, 1, figsize=(8, 6))
    axs2[0].plot(t, data_stack)
    axs2[0].plot(xs, ys, color='red', linewidth=1.5)
    axs2[0].plot(xs_2, ys_2, color='red', linewidth=1.5)
    axs2[0].set_title(f"Auto-correlated stacked trace for data with rx={gather_num}", fontsize=10)
    axs2[0].set_xlabel("Time (s)", fontsize=12)
    axs2[0].set_ylim(0.1 * ymin, 0.08 * ymax)

    # Find 2 window extents for the primaries data
    prim_max = max(prim_stack[idx_diff: int(idx_diff + delta//dt)])
    prim_max_2 = max(prim_stack[idx_diff_2: int(idx_diff_2 + delta//dt)])
    prim_min = min(prim_stack[idx_diff: int(idx_diff + delta//dt)])
    prim_min_2 = min(prim_stack[idx_diff_2: int(idx_diff_2 + delta//dt)])

    ys = [prim_min*1.1, prim_max*1.1, prim_max*1.1, prim_min*1.1, prim_min*1.1]
    ys_2 = [prim_min_2*1.1, prim_max_2*1.1, prim_max_2*1.1, prim_min_2*1.1, prim_min_2*1.1]

    # Plot auto-correlation of stacked primaries data
    t = np.linspace(-prim_stack.shape[0]//2, prim_stack.shape[0]//2, prim_stack.shape[0])*dt
    ymin = np.min(prim_stack) ; ymax = np.max(prim_stack)
    axs2[1].plot(t, prim_stack)
    axs2[1].plot(xs, ys, color='red', linewidth=1.5)
    axs2[1].plot(xs_2, ys_2, color='red', linewidth=1.5)
    axs2[1].set_title(f"Auto-correlated stacked trace for primaries with rx={gather_num}", fontsize=10)
    axs2[1].set_xlabel("Time (s)", fontsize=12)
    axs2[1].set_ylim(0.1 * ymin, 0.08 * ymax)
       
    plt.suptitle(f'\nTotal data absolute average: {round(mult_avg,2)}\n\nPrimaries absolute average: {round(prim_avg,2)}', c='r')
   
    plt.tight_layout()