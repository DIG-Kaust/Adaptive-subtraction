import numpy as np
import pylops
from pylops.optimization.solver import lsqr
from pylops.utils.backend import get_array_module
from ADMM import ADMM
def adaptive_subtraction_3d(data, multiples, nfilt, solver, solver_dict, nwin=(30,150), clipping=False):
    """Applies adaptive subtraction to all seismic gathers of a cube.

            Parameters
            ----------
            data : :obj:`np.ndarray`
                Total data gathers stored in a 3d array
            multiples : :obj:`np.ndarray`
                Initial estimated multiples stored in a 3d array
            nfilt : :obj:`int`
                Size of the filter
            solver :{“lsqr”, “ADMM”}
                Optimizer to find best filter
            solver_dict :obj:`dict`
                Dictionary with solver parameters
            nwin : :obj:`tuple`, optional
                Number of samples of window for patching data
            clipping : :obj:`boolean`
                Clip filters that exceed 10 to 1

            Returns
            -------
            primaries_cube : :obj:`np.ndarray`
                3d np.array with estimated primaries for each gather
            multiples_cube : :obj:`np.ndarray`
                3d np.array with corrected multiples for each gather

            """

    ncp = get_array_module(data)
    # Create output cubes full of zeros+
    nwin = nwin  # number of samples of window
    nover = (nwin[0] // 2, nwin[1] // 2)  # number of samples of overlapping part of window

    # reshape input data so it fits the patching
    ns = data.shape[0]  # shots, receivers, time samples
    nr = (nover[0]) * (data.shape[0] // nover[0])
    nt = (nover[1]) * (data.shape[1] // nover[1])

    primaries_cube = ncp.zeros((ns, nr, nt))
    multiples_cube = ncp.zeros((ns, nr, nt))

    num_patches = (nr // (nwin[0] // 2) - 1) * (nt // (nwin[1] // 2) - 1)
    filt_cube = ncp.zeros((ns, num_patches, nfilt))

    for shot_num in range(ns):
        primary_est, multiple_est, filts_est = adaptive_subtraction_2d_parallel(data[shot_num], multiples[shot_num], nfilt, solver,
                                                                     solver_dict, nwin=nwin, clipping=clipping)

        primaries_cube[shot_num] = primary_est
        multiples_cube[shot_num] = multiple_est
        filt_cube[shot_num] = filts_est
    return primaries_cube, multiples_cube, filt_cube