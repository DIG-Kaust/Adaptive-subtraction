import numpy as np
import pylops
from pylops.optimization.solver import lsqr
from ADMM import ADMM
def adaptive_subtraction_3d(data, multiples, nfilt, solver, nwin=(30,150)):
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
            nwin : :obj:`tuple`, optional
                Number of samples of window for patching data

            Returns
            -------
            primaries_cube : :obj:`np.ndarray`
                3d np.array with estimated primaries for each gather
            multiples_cube : :obj:`np.ndarray`
                3d np.array with corrected multiples for each gather

            """

    ns = data.shape[0]
    nr, nt = 180, 1950  # number of receivers and time samples necessary for patching the gathers

    # Reshape the input data to be able to patch it
    data = data[:, :nr, :nt]
    multiples = multiples[:, :nr, :nt]

    # Create output cubes full of zeros+
    primaries_cube = np.zeros((ns, nr, nt))
    multiples_cube = np.zeros((ns, nr, nt))

    for shot_num in range(ns):

        total_shot = data[shot_num]
        mult_shot = multiples[shot_num]

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
        data_patched = PatchOpH.H * total_shot.ravel()
        multiples_patched = PatchOpH.H * mult_shot.ravel()

        # Reorder the data so that every row is a single patch
        num_patches = nwins[0] * nwins[1]
        data_patched = np.reshape(data_patched, (num_patches, nwin[0] * nwin[1]))
        multiples_patched = np.reshape(multiples_patched, (num_patches, nwin[0] * nwin[1]))
        primary_est = np.zeros_like(data_patched)
        multiple_est = np.zeros_like(multiples_patched)

        for i in range(num_patches):
            data_patch_i = np.reshape(data_patched[i], (nwin[0], nwin[1]))
            multiple_patch_i = np.reshape(multiples_patched[i], (nwin[0], nwin[1]))
            CopStack = []

            # construct the convolutional operator
            for j in range(nwin[0]):
                C = pylops.utils.signalprocessing.convmtx(multiple_patch_i[j], nfilt)
                Cop = pylops.basicoperators.MatrixMult(C[nfilt // 2:-(nfilt // 2)])
                CopStack.append(Cop)
            dataStack = data_patch_i.ravel()
            CopStack = pylops.VStack(CopStack)
            # solve for the filter
            if solver=='lsqr':
                filt_est = lsqr(CopStack, dataStack, x0=np.zeros(nfilt), niter=15)[0]
            elif solver=='ADMM':
                filt_est = ADMM(CopStack, dataStack, rho=1e0, nouter=2000, ninner=5, eps=0e0)
            multiple_est[i] = (CopStack * filt_est).T  # Make sure it's in (nr, nt) format before patching back!
            primary_est[i] = (dataStack - multiple_est[i].T).T
        # Glue the patches back together
        primary_est = PatchOp * primary_est.ravel()
        multiple_est = PatchOp * multiple_est.ravel()
        primary_est = np.reshape(primary_est, (nr, nt))
        multiple_est = np.reshape(multiple_est, (nr, nt))

        primaries_cube[shot_num] = primary_est
        multiples_cube[shot_num] = multiple_est

    return primaries_cube, multiples_cube