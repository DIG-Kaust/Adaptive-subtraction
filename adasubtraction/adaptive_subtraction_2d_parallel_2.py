import numpy as np
import pylops
import cupy as cp
from pylops.utils.backend import get_array_module
from pylops.optimization.solver import lsqr
from ADMM2 import ADMM
import multiprocessing as mp

def adaptive_subtraction(data_patched, multiples_patched, nwin, solver, nfilt, solver_dict, clipping):
    ncp = get_array_module(multiples_patched)
    data_patch_i = ncp.reshape(data_patched, (nwin[0], nwin[1]))

    CopStack = []
    if multiples_patched.ndim == 3:
        CopMultStack = []
        multiple_patch_i = ncp.reshape(multiples_patched, (multiples_patched.shape[0], nwin[0], nwin[1]))
        filt_est = np.zeros(multiple_patch_i.shape[0], nfilt)
        for i in range(multiple_patch_i):
            # if no multiples present put filter to 1
            no_multiples = multiple_patch_i == 0
            if no_multiples.all():
                filt_est[i] = ncp.ones(nfilt)
            # construct the convolutional operator
            else:
                for j in range(nwin[0]):
                    C = pylops.utils.signalprocessing.convmtx(ncp.asarray(multiple_patch_i[i, j]), nfilt)
                    Cop = pylops.basicoperators.MatrixMult(C[nfilt // 2:-(nfilt // 2)])
                    CopStack.append(Cop)
                CopStack = pylops.VStack(CopStack)
                CopMultStack.append(CopStack)
        CopMultStack = pylops.HStack(CopMultStack)
        # solve for the filter
        if solver == 'lsqr':
            filters = lsqr(CopMultStack, dataStack, x0=ncp.asarray(solver_dict['x0']), niter=solver_dict['niter'],
                            damp=solver_dict['damp'])[0]
        elif solver == 'ADMM':
            # print('zeroes:' , dataStack.size - np.count_nonzero(dataStack))
            filters = ADMM(CopMultStack, dataStack, rho=solver_dict['rho'], nouter=solver_dict['nouter'],
                            ninner=solver_dict['ninner'], eps=solver_dict['eps'])

        if (filt_est == 0).all():
            filt_est = filters
        else:
            count = 0
            for i in range(filt_est):
                if (filt_est[i] == 0).all():
                    filt_est[i] = filters[0]
                    count += 1
        # clip filter if neccesary
        if clipping:
            for i in range(filt_est.shape[0]):
                if max(abs(filt_est[i])) > 10:
                    filt_est[i] = ncp.ones(nfilt)

        multiple_est = (CopStack * filt_est).T  # Make sure it's in (nr, nt) format before patching back!


    elif multiples_patched.ndim == 2:
        multiple_patch_i = ncp.reshape(multiples_patched, (nwin[0], nwin[1]))
        # construct the convolutional operator
        for j in range(nwin[0]):
            C = pylops.utils.signalprocessing.convmtx(ncp.asarray(multiple_patch_i[j]), nfilt)
            Cop = pylops.basicoperators.MatrixMult(C[nfilt // 2:-(nfilt // 2)])
            CopStack.append(Cop)
        CopStack = pylops.VStack(CopStack)

        # if no multiples present put filter to 1
        no_multiples = multiple_patch_i == 0
        if no_multiples.all():
            filt_est = ncp.ones(nfilt)
        else:
            # solve for the filter
            if solver == 'lsqr':
                filt_est = lsqr(CopStack, dataStack, x0=ncp.asarray(solver_dict['x0']), niter=solver_dict['niter'],
                                damp=solver_dict['damp'])[0]
            elif solver == 'ADMM':
                # print('zeroes:' , dataStack.size - np.count_nonzero(dataStack))
                filt_est = ADMM(CopStack, dataStack, rho=solver_dict['rho'], nouter=solver_dict['nouter'],
                                ninner=solver_dict['ninner'], eps=solver_dict['eps'])

        # clip filter if neccesary
        if clipping and max(abs(filt_est)) > 10:
            filt_est = ncp.ones(nfilt)

        multiple_est = (CopStack * filt_est).T

    dataStack = data_patch_i.ravel()

    primary_est = (dataStack - multiple_est.T).T

    return primary_est, multiple_est, filt_est

def adaptive_subtraction_2d_parallel(data, multiples, nfilt, solver, solver_dict, nwin=(30, 150), clipping=False):
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
                Number of samples of window for patching data. Must be even numbers.
            clipping : :obj:`boolean`
                Clip filters that exceed 10 to 1

            Returns
            -------
            primary_est : :obj:`np.ndarray`
                2d np.array with estimated primaries
            multiple_est : :obj:`np.ndarray`
                2d np.array with estimated multiples
            filt_est : :obj:`np.ndarray`
                2d np.array with estimated filters
            Note
            -------
            Processes are performed in parallel in the CPU.
            """

    ncp = get_array_module(data)

    nwin = nwin  # number of samples of window
    nover = (nwin[0] // 2, nwin[1] // 2)  # number of samples of overlapping part of window

    # reshape input data so it fits the patching
    nr = (nover[0]) * (data.shape[0] // nover[0])
    nt = (nover[1]) * (data.shape[1] // nover[1])
    data = data[:nr, :nt]

    if multiples.ndim == 3:
        multiples = multiples[:, nr, :nt]
    elif multiples.ndim == 2:
        multiples = multiples[nr, :nt]

    # Create patching operator
    dimsd = (nr, nt)  # shape of 2-dimensional data
    nop = nwin  # size of model in the transformed domain
    nwins = (nr // (nwin[0] // 2) - 1, nt // (nwin[1] // 2) - 1)
    dims = (nwins[0] * nop[0],  # shape of 2-dimensional model
            nwins[1] * nop[1])
    I = pylops.Identity(nwin[0] * nwin[1])

    # Calculate number of patches
    num_patches = nwins[0] * nwins[1]

    # Get the patching operators
    PatchOpH = pylops.signalprocessing.Patch2D(I, dims, dimsd, nwin, nover, nop, tapertype='none', design=False)
    PatchOp = pylops.signalprocessing.Patch2D(I, dims, dimsd, nwin, nover, nop, tapertype='hanning', design=False)

    # Patch the data
    data_patched = PatchOpH.H * data.ravel()

    if multiples.ndim == 3:
        multiples_patched = []
        for i in range(multiples.shape[0]):
            multiples_patched.append(PatchOpH.H * multiples[i].ravel())
        multiples_patched = np.array(multiples_patched)
        multiples_patched = ncp.reshape(multiples_patched, (multiples_patched.shape[0], num_patches, nwin[0] * nwin[1]))
    elif multiples.ndim == 2:
        multiples_patched = PatchOpH.H * multiples.ravel()
        multiples_patched = ncp.reshape(multiples_patched, (num_patches, nwin[0] * nwin[1]))

    data_patched = ncp.reshape(data_patched, (num_patches, nwin[0] * nwin[1]))

    nproc = 15
    pool = mp.Pool(processes=nproc)
    if multiples.ndim == 3:
        out = pool.starmap(adaptive_subtraction, [(data_patched[i], multiples_patched[:,i], nwin, solver, nfilt,
                                                  solver_dict, clipping) for i in range(num_patches)])
    elif multiples.ndim == 2:
        out = pool.starmap(adaptive_subtraction, [(data_patched[i], multiples_patched[i], nwin, solver, nfilt,
                                                   solver_dict, clipping) for i in range(num_patches)])
    primary_est = [out[i][0] for i in range(num_patches)]
    primary_est = np.array(primary_est)
    multiple_est = [out[i][1] for i in range(num_patches)]
    multiple_est = np.array(multiple_est)
    filts_est = [out[i][2] for i in range(num_patches)]
    filts_est = np.array(filts_est)

    # Glue the patches back together
    # first put the arrays back on the CPU
    multiple_est = cp.asnumpy(multiple_est)
    primary_est = cp.asnumpy(primary_est)
    primary_est = PatchOp * primary_est.ravel()
    multiple_est = PatchOp * multiple_est.ravel()
    primary_est = np.reshape(primary_est, (nr, nt))
    multiple_est = np.reshape(multiple_est, (nr, nt))

    return primary_est, multiple_est, filts_est
















