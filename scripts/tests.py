import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

import random
import time
import warnings
import pylops
import json
warnings.filterwarnings('ignore')

from pylops.optimization.solver import lsqr
from pylops.utils.backend import get_array_module
from mpl_toolkits.axes_grid1 import make_axes_locatable

# some_file.py
import sys
# # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../adasubtraction')

from ADMM import ADMM
from read_gathers import read_gathers
from adaptive_subtraction_2d import adaptive_subtraction_2d
from adaptive_subtraction_2d_parallel import adaptive_subtraction_2d_parallel
from adaptive_subtraction_2d_parallel_2 import adaptive_subtraction_2d_parallel_2
from adaptive_subtraction_3d import adaptive_subtraction_3d
from adaptive_subtraction_3d_parallel import adaptive_subtraction_3d_parallel
from adaptive_subtraction_qc import adaptive_subtraction_qc

dt = 0.004
dx = 12.5



data = np.load('../data/data_cube.npz')['arr_0']

multiples = np.load('../data/srme_multiples.npz')['arr_0']
multiples = -1*multiples/np.amax(abs(multiples)) 

true_multiples = np.load('../data/true_multiples.npz')['arr_0']


nwin = (100, 200)

nover = (nwin[0] // 2, nwin[1] // 2)
    
# reshape input data so it fits the patching
nr = (nover[0]) * (data.shape[1] // nover[0])
nt = (nover[1]) * (data.shape[2] // nover[1])

data_300 = data[300, :nr, :nt]

mult_300 = multiples[300, :nr, :nt]

true_mult_300 = true_multiples[300, :nr, :nt]

# mults = np.array([multiples[299, :nr, :nt], multiples[300, :nr, :nt], multiples[301, :nr, :nt]])

admm_errors = {}

start_time = time.time()/60

nfilt = 55

for eps in [0.001, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 2, 10]:

    for rho in [0.001, 0.01, 0.1, 0.5, 1.5, 2, 2.5, 3, 10]:

            _, admm_mult, _ = adasubtraction_parallel(data_300, mult_300, solver='ADMM', nfilt=nfilt,
                                                                        nwin=nwin,
                                                                        solver_dict={'rho':rho,
                                                                                     'nouter':200,
                                                                                     'ninner':5,
                                                                                     'eps':eps})

            r_error = round((np.linalg.norm(true_mult_300-admm_mult)/np.linalg.norm(true_mult_300))*100, 3)

            admm_errors[f'eps{eps}_rho{rho}'] = r_error

            print(f'eps{eps}_rho{rho} error = {r_error}')

print("--- %s minutes ---" % round((time.time()/60 - start_time), 3))

with open("admm_errors_4_1shot.txt", "w") as file:
    json.dump(admm_errors, file)  # encode dict into JSON
print("Done writing dict into admm_errors_4.txt file")


    
