import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

import random
import time
import warnings
import pylops
warnings.filterwarnings('ignore')

from pylops.optimization.solver import lsqr
from pylops.utils.backend import get_array_module
from mpl_toolkits.axes_grid1 import make_axes_locatable

# some_file.py
import sys
# # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../adasubtraction')

from ADMM2 import ADMM
from read_gathers import read_gathers
from adaptive_subtraction_2d import adaptive_subtraction_2d
from adaptive_subtraction_3d import adaptive_subtraction_3d
from adaptive_subtraction_2d_parallel_2 import adaptive_subtraction_2d_parallel
from adaptive_subtraction_3d_parallel import adaptive_subtraction_3d_parallel
from adaptive_subtraction_qc import adaptive_subtraction_qc

dt = 0.004
dx = 12.5

def rm_upper_noise(data, trim, amp_scale=0.03):
    
    if data.ndim == 2: 
        data = data[:, int(trim/dt):]  #trim in seconds
        max_amp = np.max(np.abs(data))

        for i in range(data.shape[0]):
            next_trace = 0
            for j in range(data.shape[1]):
                if next_trace == 0:
                    if np.abs(data[i, j]) <= amp_scale*max_amp:
                        data[i, j] = 0
                    else:
                        next_trace = 1
                else:
                    break
        data_clean = np.hstack((np.zeros((data.shape[0], int(trim/dt))), data))
        
    elif data.ndim == 3:        

        data = data[:, :, int(trim/dt):]  #trim in seconds
        for s in range(data.shape[0]):
            max_amp = np.max(np.abs(data[s]))
            for i in range(data.shape[1]):
                next_trace = 0
                for j in range(data.shape[2]):
                    if next_trace == 0:
                        if np.abs(data[s, i, j]) <= amp_scale*max_amp:
                            data[s, i, j] = 0
                        else:
                            next_trace = 1
                    else:
                        break
        data_clean = np.vstack((np.zeros((int(trim/dt), data.shape[1], data.shape[0])), data.T)).T
        
    return data_clean

def phase_shift(x, phase_shift):
    
    from scipy.signal import hilbert
        
    x_shift = np.cos(phase_shift)*x + np.sin(phase_shift)*hilbert(x).imag
    
    return x_shift

data = np.load('/home/bermanu/Documents/Devito/output/fs_data.npz')['arr_0']

primaries = np.load('/home/bermanu/Documents/Devito/output2/primaries.npz')['arr_0']

multiples = np.load('/home/bermanu/Documents/Devito/output2/multiples.npz')['arr_0']

data_128 = data[172]

prim_128 = primaries[172]

mult_128 = multiples[172]

data_128 = rm_upper_noise(data_128, trim=1.65, amp_scale=0.03)

prim_128 = rm_upper_noise(prim_128, trim=1.65, amp_scale=0.03)

mult_128 = rm_upper_noise(mult_128, trim=3.2, amp_scale=0.008)

corr_mult_128 = np.zeros_like(mult_128)

for i in range(mult_128.shape[0]):
    
#     scale = random.uniform(0.5, 1.5)
    scale = 1.3
    corr_mult_128[i] = scale*phase_shift(mult_128[i], np.pi/2)
    
corr_mult_128 = rm_upper_noise(corr_mult_128, trim=3.2, amp_scale=0.008)

nwin = (100, 200)

nfilt = 35

eps = 0

admm_diff = []

for rho in np.logspace(-3,2,10):
    
        _, _, _, diff = adasubtraction_parallel(data_128, corr_mult_128, solver='ADMM', nfilt=nfilt,
                                                                    nwin=nwin,
                                                                    solver_dict={'rho':rho,
                                                         			    'nouter':200,                                                                                                                    												'ninner':5,
                                                                                 'eps':eps})

        admm_diff.append((rho, diff))

admm_diff = np.array(admm_diff)
np.savez('admm_diff_2.npz', admm_diff)
    

