import numpy as np
import pylops
import cupy as cp
from pylops.utils.backend import get_array_module
from pylops.optimization.solver import lsqr
from ADMM import ADMM

def adaptive_subtraction_2d(data, multiples_init, nfilt, solver, solver_dict, nwin=(30,150), clipping=False):
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
            nwin : :obj:`tuple`, optional
                Number of samples of window for patching data
            clipping : :obj:`boolean`
                Clip filters that exceed 10 to 1

            Returns
            -------
            primary_est : :obj:`np.ndarray`
                2d np.ndarray with estimated primaries
            multiple_est : :obj:`np.ndarray`
                2d np.ndarray with estimated multiples

            """

    nr, nt = data.shape[0],  data.shape[1] #receivers, time samples
    ncp = get_array_module(multiples_init)

    # Create patching operator
    dimsd = (nr, nt)  # shape of 2-dimensional data
    nwin = (nwin[0], nwin[1])  # number of samples of window
    nover = (nwin[0] // 2, nwin[1] // 2)  # number of samples of overlapping part of window
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
    multiples_patched = PatchOpH.H * multiples_init.ravel()

    # Reorder the data so that every row is a single patch
    num_patches = nwins[0] * nwins[1]
    data_patched = ncp.reshape(data_patched, (num_patches, nwin[0] * nwin[1]))
    multiples_patched = ncp.reshape(multiples_patched, (num_patches, nwin[0] * nwin[1]))
    primary_est = ncp.zeros_like(data_patched)
    multiple_est = ncp.zeros_like(multiples_patched)

    for i in range(num_patches):
        data_patch_i = ncp.reshape(data_patched[i], (nwin[0], nwin[1]))
        multiple_patch_i = ncp.reshape(multiples_patched[i], (nwin[0], nwin[1]))
        CopStack = []

        # construct the convolutional operator
        for j in range(nwin[0]):
            C = pylops.utils.signalprocessing.convmtx(multiple_patch_i[j], nfilt)
            Cop = pylops.basicoperators.MatrixMult(C[nfilt // 2:-(nfilt // 2)])
            CopStack.append(Cop)
        CopStack = pylops.VStack(CopStack)
        
        dataStack = data_patch_i.ravel()
        # solve for the filter
        if solver=='lsqr':
            filt_est = lsqr(CopStack, dataStack, x0=cp.asarray(solver_dict['x0']), niter=solver_dict['niter'], damp=solver_dict['damp'])[0]
        elif solver=='ADMM':
            filt_est = ADMM(CopStack, dataStack, rho=solver_dict['rho'], nouter=solver_dict['nouter'], ninner=solver_dict['ninner'], eps=solver_dict['eps'])
        # clip filter if neccesary
        if clipping and max(abs(filt_est)) > 10:
            filt_est = ncp.ones(nfilt)
        multiple_est[i] = (CopStack * filt_est).T  # Make sure it's in (nr, nt) format before patching back!
        primary_est[i] = (dataStack - multiple_est[i].T).T
    # Glue the patches back together
    # first put the arrays back on the CPU
    multiple_est = cp.asnumpy(multiple_est) 
    primary_est = cp.asnumpy(primary_est)
    primary_est = PatchOp * primary_est.ravel()
    multiple_est = PatchOp * multiple_est.ravel()
    primary_est = np.reshape(primary_est, (nr, nt))
    multiple_est = np.reshape(multiple_est, (nr, nt))

    return primary_est, multiple_est
