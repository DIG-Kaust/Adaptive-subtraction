import numpy as np
import pylops
import cupy as cp
import multiprocessing as mp
from pylops.utils.backend import get_array_module
from pylops.optimization.solver import lsqr
from pylops import LinearOperator
from adasubtraction.ADMM import ADMM

def prox_data(x, rho):
    return np.maximum( np.abs( x ) - rho, 0. ) * np.sign(x)

def adaptive_subtraction(data_patched, multiples_patched, nwin, solver, nfilt, solver_dict, clipping=False):
    ncp = get_array_module(multiples_patched)
    data_patch_n = ncp.reshape(data_patched, (nwin[0], nwin[1]))
    multiple_patch_n = ncp.reshape(multiples_patched, (nwin[0], nwin[1]))
    CopStack = []

    # construct the convolutional operator
    for j in range(nwin[0]):
        C = pylops.utils.signalprocessing.convmtx(ncp.asarray(multiple_patch_n[j]), nfilt)
        Cop = pylops.basicoperators.MatrixMult(C[nfilt // 2:-(nfilt // 2)])
        CopStack.append(Cop)
    CopStack = pylops.VStack(CopStack)
    dataStack = data_patch_n.ravel()

    # if no multiples present put filter to 1
    no_multiples = multiple_patch_n == 0
    if no_multiples.all():
        filt_est = ncp.ones(nfilt)
    else:
        # solve for the filter
        if solver == 'lsqr':
            filt_est = lsqr(CopStack, dataStack, x0=ncp.asarray(solver_dict['x0']), niter=solver_dict['niter'],
                            damp=solver_dict['damp'])[0]
        elif solver == 'ADMM':
            filt_est = ADMM(CopStack, dataStack, rho=solver_dict['rho'], nouter=solver_dict['nouter'],
                            ninner=solver_dict['ninner'], eps=solver_dict['eps'])

    # clip filter if neccesary
    if clipping and max(abs(filt_est)) > 10:
        filt_est = ncp.ones(nfilt)
    multiple_est = (CopStack * filt_est).T  # Make sure it's in (nr, nt) format before patching back!
    primary_est = (dataStack - multiple_est.T).T

    return primary_est, multiple_est, filt_est

def adasubtraction_parallel(data, multiples, nfilt, solver, solver_dict, nwin, cpu_num, clipping=False):
    """Applies adaptive subtraction to all seismic gathers of a cube.

            Parameters
            ----------
            data : :obj:`np.ndarray`
                Total data shot gather
            multiples_init : :obj:`np.ndarray`
                Initial estimated multiples
            nfilt : :obj:`int`
                Size of the filter
            solver :{“lsqr”, “ADMM”}
                Optimizer to find best filter
            solver_dict :obj:`dict`
                Dictionary with solver parameters
            nwin : :obj:`tuple`
                Number of samples of window for patching data. Must be even numbers.
            cpu_num : :obj:`int`
                Total number of cpu cores employed
            clipping : :obj:`boolean`. Default to False
                Clip filters that exceed 10 to 1

            Returns
            -------
            primaries_est : :obj:`np.ndarray`
                2d np.array with estimated primaries
            multiples_est : :obj:`np.ndarray`
                2d np.array with estimated multiples
            filts_est : :obj:`np.ndarray`
                2d np.array with estimated filters
            Note
            -------
            Processes are performed in parallel with different the CPU cores.
            """

    ncp = get_array_module(data)

    nwin = nwin  # number of samples of window
    nover = (nwin[0] // 2, nwin[1] // 2)  # number of samples of overlapping part of window

    # reshape input data so it fits the patching
    nr = (nover[0]) * (data.shape[0] // nover[0])
    nt = (nover[1]) * (data.shape[1] // nover[1])

    data = data[:nr, :nt]
    multiples = multiples[:nr, :nt]

    # Create patching operator
    dimsd = (nr, nt)  # shape of 2-dimensional data
    nop = nwin  # size of model in the transformed domain
    nwins = (nr // (nwin[0] // 2) - 1, nt // (nwin[1] // 2) - 1)
    dims = (nwins[0] * nop[0],  # shape of 2-dimensional model
            nwins[1] * nop[1])
    I = pylops.Identity(nwin[0] * nwin[1])

    # Get the patching operators
    PatchOpH = pylops.signalprocessing.Patch2D(I, dims, dimsd, nwin, nover, nop, tapertype='none', design=False)
    PatchOp = pylops.signalprocessing.Patch2D(I, dims, dimsd, nwin, nover, nop, tapertype='hanning', design=False)

    # Patch the data
    data_patched = PatchOpH.H * data.ravel()
    multiples_patched = PatchOpH.H * multiples.ravel()

    # Reorder the data so that every row is a single patch
    num_patches = nwins[0] * nwins[1]
    data_patched = ncp.reshape(data_patched, (num_patches, nwin[0] * nwin[1]))
    multiples_patched = ncp.reshape(multiples_patched, (num_patches, nwin[0] * nwin[1]))

    nproc = cpu_num
    pool = mp.Pool(processes=nproc)
    out = pool.starmap(adaptive_subtraction, [(data_patched[n], multiples_patched[n], nwin, solver, nfilt,
                                               solver_dict, clipping) for n in range(num_patches)])
    primary_est = [out[n][0] for n in range(num_patches)]
    primary_est = np.array(primary_est)
    multiple_est = [out[n][1] for n in range(num_patches)]
    multiple_est = np.array(multiple_est)
    filts_est = [out[n][2] for n in range(num_patches)]
    filts_est = np.array(filts_est)

    # Glue the patches back together
    # first put the arrays back on the CPU
    multiple_est = cp.asnumpy(multiple_est)
    primary_est = cp.asnumpy(primary_est)
    primaries_est = PatchOp * primary_est.ravel()
    multiples_est = PatchOp * multiple_est.ravel()
    primaries_est = np.reshape(primaries_est, (nr, nt))
    multiples_est = np.reshape(multiples_est, (nr, nt))

    return primaries_est, multiples_est, filts_est


































