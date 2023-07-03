import numpy as np
import pylops
import cupy as cp
import multiprocessing as mp
from pylops.utils.backend import get_array_module
from pylops.optimization.solver import lsqr
from adasubtraction.ADMM import ADMM


def adaptive_subtraction(data_patched, multiples_patched, nwin, nwins, solver, nfilt, solver_dict, smooth_x, smooth_z, smooth_mults, clipping):
    ncp = get_array_module(multiples_patched)

    multiple_est = ncp.zeros_like(data_patched)

    Mult_Diag_Stack = []
    Data_Stack_NC = [] # NC stands for Non-Complete

    non_zero_patches = [] # Create a list to store the indexes of the patches with non_zero multiples

    num_patches = nwins[0]*nwins[1]
    
    if multiples_patched.ndim == 3:

        m_num = multiples_patched.shape[0]

        filt_est = ncp.ones((m_num * num_patches * nfilt))
        
        for n in range(num_patches):

            multiple_patch_n = ncp.reshape(multiples_patched[:, n], (m_num, nwin[0], nwin[1]))  # Reshape flat patches to 2D
            CopMultStack = []

            if not (multiple_patch_n==0).all():
            # construct the convolutional operator
                for m in range(m_num):
                    CopStack = []
                    for j in range(nwin[0]):
                        C = pylops.utils.signalprocessing.convmtx(ncp.asarray(multiple_patch_n[m, j]), nfilt)
                        Cop = pylops.basicoperators.MatrixMult(C[nfilt // 2:-(nfilt // 2)])
                        CopStack.append(Cop)
                    CopStack = pylops.VStack(CopStack)
                    CopMultStack.append(CopStack)

                # if no multiples present put filter to 1
                CopMultStack = pylops.HStack(CopMultStack) 
                Mult_Diag_Stack.append(CopMultStack)
                Data_Stack_NC.append(data_patched[n*(nwin[0]*nwin[1]): (n+1)*(nwin[0]*nwin[1])]) # NC stands for Non-Complete
                non_zero_patches.append(n)

        Data_Stack_NC = np.array(Data_Stack_NC).ravel()
        
        Mult_Diag_Stack = pylops.BlockDiag(Mult_Diag_Stack)  # construct Diagonal Matrix with the multi convolutional operators

        if smooth_mults:   # Make filters of the same patch similar 
        
            if smooth_x and smooth_z:
                Lop = pylops.Laplacian(dims=(m_num, nwins[0], nwins[1], nfilt), dirs=(0,1,2), weights=(1,1,1), sampling=(1,1,1)) # Laplacian operator
                Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop]) # Stack the convolutional operators and the laplacian operator vertically
                Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0]))) # Pad data with 0

            elif smooth_x:
                Lop = pylops.Laplacian(dims=(m_num, nwins[0], nwins[1], nfilt), dirs=(0,1)) 
                Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop]) 
                Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0]))) 
            elif smooth_z:
                Lop = pylops.Laplacian(dims=(m_num, nwins[0], nwins[1], nfilt), dirs=(0,2)) 
                Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop]) 
                Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0]))) 
            else:
                Lop = pylops.SecondDerivative(N = (m_num*nwins[0]*nwins[1]*nfilt), dims=(m_num, nwins[0], nwins[1], nfilt), dir=0) 
                Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop]) 
                Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0])))

        
        elif not smooth_mults:  # Allow filters of the same patch different

            if smooth_x and smooth_z:
                Lop = pylops.Laplacian(dims=(m_num, nwins[0], nwins[1], nfilt), dirs=(1,2), weights=(1,1), sampling=(1,1)) # Laplacian operator
                Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop]) # Stack the convolutional operators and the laplacian operator vertically
                Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0]))) # Pad data with 0

            elif smooth_x:
                Lop = pylops.SecondDerivative(N = (m_num*nwins[0]*nwins[1]*nfilt), dims=(m_num, nwins[0], nwins[1], nfilt), dir=1) 
                Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop]) 
                Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0]))) 
            elif smooth_z:
                Lop = pylops.SecondDerivative(N = (m_num*nwins[0]*nwins[1]*nfilt), dims=(m_num, nwins[0], nwins[1], nfilt), dir=2) 
                Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop]) 
                Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0]))) 


        # solve for the filter
        if solver == 'lsqr':
            filt_est_NC = lsqr(Mult_Diag_Stack, Data_Stack_NC, x0=ncp.asarray(solver_dict['x0']), niter=solver_dict['niter'],
                            damp=solver_dict['damp'])[0]
            
        elif solver == 'ADMM':
            filt_est_NC = ADMM(Mult_Diag_Stack, Data_Stack_NC, rho=solver_dict['rho'], nouter=solver_dict['nouter'],
                            ninner=solver_dict['ninner'], eps=solver_dict['eps'])

        multiple_est_NC = (Mult_Diag_Stack * filt_est_NC).T

        # Add filters of non-zero patches to the solution. Clip filter if neccesary
        s = 0 # flag
        for n in non_zero_patches:
            filt_est[n*nfilt*m_num : (n+1)*nfilt*m_num] = filt_est_NC[s*nfilt*m_num : (s+1)*nfilt*m_num]
            multiple_est[n*(nwin[0] * nwin[1]):(n+1)*(nwin[0] * nwin[1])] = multiple_est_NC[s*(nwin[0] * nwin[1]):(s+1)*(nwin[0] * nwin[1])]
            s += 1
            if clipping and max(abs(filt_est[n*nfilt : (n+1)*nfilt])) > 10:
                filt_est[n*nfilt*m_num : (n+1)*nfilt*m_num ] = ncp.ones(nfilt*m_num)

        filt_est = filt_est.reshape((num_patches, m_num, nfilt))

    elif multiples_patched.ndim == 2:

        filt_est = ncp.ones((num_patches * nfilt))

        for n in range(num_patches):

            multiple_patch_n = ncp.reshape(multiples_patched[n], (nwin[0], nwin[1]))  # Reshape flat patches to 2D

            CopStack = []   # Create list to store the convolutional operators and then stack them vertically

            if not (multiple_patch_n==0).all(): 
            # construct the convolutional operator
                for j in range(nwin[0]):
                    C = pylops.utils.signalprocessing.convmtx(multiple_patch_n[j], nfilt)
                    Cop = pylops.basicoperators.MatrixMult(C[nfilt // 2:-(nfilt // 2)])
                    CopStack.append(Cop)

                CopStack = pylops.VStack(CopStack)

                # if no multiples present put filter to 1

                Mult_Diag_Stack.append(CopStack)
                Data_Stack_NC.append(data_patched[n*(nwin[0]*nwin[1]): (n+1)*(nwin[0]*nwin[1])])
                non_zero_patches.append(n)

        Data_Stack_NC = np.array(Data_Stack_NC).ravel()
        
        Mult_Diag_Stack = pylops.BlockDiag(Mult_Diag_Stack)  # construct Diagonal Matrix with the convolutional operators

        if smooth_x and smooth_z:
            Lop = pylops.Laplacian(dims=(nwins[0], nwins[1], nfilt), dirs=(0,1)) # Second derivative operator
            Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop]) # Stack the convolutional operators and the laplacian operator vertically
            Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0]))) # Pad data with 0

        elif smooth_x:
            Lop = pylops.SecondDerivative(N = (nwins[0]*nwins[1]*nfilt), dims=(nwins[0], nwins[1], nfilt), dir=0) # Second derivative operator
            Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop])
            Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0]))) 
        elif smooth_z:
            Lop = pylops.SecondDerivative(N = (nwins[0]*nwins[1]*nfilt), dims=(nwins[0], nwins[1], nfilt), dir=1)
            Mult_Diag_Stack = pylops.VStack([Mult_Diag_Stack, Lop])
            Data_Stack_NC = np.concatenate((Data_Stack_NC, np.zeros(Lop.shape[0]))) 
       
        # solve for the filter  
        if solver=='lsqr':                
            filt_est_NC = lsqr(Mult_Diag_Stack, Data_Stack_NC, x0=ncp.asarray(solver_dict['x0']), niter=solver_dict['niter'], damp=solver_dict['damp'])[0]
        
        elif solver=='ADMM':                   # NC stands for Non-Complete
            filt_est_NC = ADMM(Mult_Diag_Stack, Data_Stack_NC, rho=solver_dict['rho'], nouter=solver_dict['nouter'], ninner=solver_dict['ninner'], eps=solver_dict['eps'])

        multiple_est_NC = (Mult_Diag_Stack * filt_est_NC).T

        # Add filters of non-zero patches to the solution. Clip filters if neccesary
        s = 0 # flag
        for n in non_zero_patches:
            filt_est[n*nfilt : (n+1)*nfilt] = filt_est_NC[s*nfilt : (s+1)*nfilt]
            multiple_est[n*(nwin[0] * nwin[1]):(n+1)*(nwin[0] * nwin[1])] = multiple_est_NC[s*(nwin[0] * nwin[1]):(s+1)*(nwin[0] * nwin[1])]
            s += 1
            if clipping and max(abs(filt_est[n*nfilt : (n+1)*nfilt])) > 10:
                filt_est[n*nfilt : (n+1)*nfilt] = ncp.ones(nfilt)

        filt_est = filt_est.reshape((num_patches, nfilt))

    primary_est = (data_patched - multiple_est.T).T
                
    return primary_est, multiple_est, filt_est

def adasubtraction_joint(data, multiples, nfilt, solver, solver_dict, nwin, cpu_num, clipping=False, smooth_x=False, 
                         smooth_z=False, smooth_mults=False):
    """Applies adaptive subtraction to a shot gather using a joint optimization for all filters of the patches. Smoothing regularization between filters can be added. The input multiples can be three sequential predicted multiples.

            Parameters
            ----------
            data : :obj:`np.ndarray`
                Total data shot gather
            multiples_init : :obj:`np.ndarray`
                Initial predicted multiples. Can be a 3d or 2d np.array. If 3d, the multiples -i and +i 
                predictions are stacked horizontally in the operator.
            nfilt : :obj:`int`
                Size of the filter
            solver :{“lsqr”, “ADMM”}
                Optimizer to find best filter
            solver_dict :obj:`dict`
                Dictionary with solver parameters. See :func:`lsqr` and :func:`ADMM` for more information
            nwin : :obj:`tuple`, optional
                Number of samples of window for patching data. Must be even numbers.
            cpu_num : :obj:`int`
                Total number of cpu cores employed
            clipping : :obj:`boolean`
                Clip filters that exceed 10 to 1
            smooth_x : :obj:`boolean`
                Apply smoothing operator to horizontal filter axis
            smooth_z : :obj:`boolean`
                Apply smoothing operator to vertical filter axis
            smooth_mults : :obj:`boolean`
                Apply smoothing operator to multiples (3) filter axis 

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
            Processes are performed in parallel with different CPU cores.
            """

    ncp = get_array_module(data)

    nwin = nwin  # number of samples of window
    nover = (nwin[0] // 2, nwin[1] // 2)  # number of samples of overlapping part of window

    # reshape input data so it fits the patching
    nr = (nover[0]) * (data.shape[0] // nover[0])
    nt = (nover[1]) * (data.shape[1] // nover[1])
    data = data[:nr, :nt]

    if multiples.ndim == 3:
        multiples = multiples[:, :nr, :nt]
    elif multiples.ndim == 2:
        multiples = multiples[:nr, :nt]

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
        multiples_patched = ncp.array(multiples_patched)
        multiples_patched = ncp.reshape(multiples_patched, (multiples_patched.shape[0], num_patches, nwin[0] * nwin[1]))
    elif multiples.ndim == 2:
        multiples_patched = PatchOpH.H * multiples.ravel()
        multiples_patched = ncp.reshape(multiples_patched, (num_patches, nwin[0] * nwin[1]))

    nproc = cpu_num
    pool = mp.Pool(processes=nproc)
    
    if multiples.ndim == 3:
        out = pool.starmap(adaptive_subtraction, [(data_patched, multiples_patched, nwin, np.array(nwins), solver, nfilt,
                                                  solver_dict, smooth_x, smooth_z, smooth_mults, clipping)])
    elif multiples.ndim == 2:
        out = pool.starmap(adaptive_subtraction, [(data_patched, multiples_patched, nwin, np.array(nwins), solver, nfilt,
                                                  solver_dict, smooth_x, smooth_z, smooth_mults, clipping)])
                                                   
    primary_est = out[0][0]
    primary_est = np.array(primary_est)
    multiple_est = out[0][1] 
    multiple_est = np.array(multiple_est)
    filt_est = out[0][2]
    filt_est = np.array(filt_est)

    # Glue the patches back together. First put the arrays back on the CPU
    multiple_est = cp.asnumpy(multiple_est)
    primary_est = cp.asnumpy(primary_est)
    primary_est = PatchOp * primary_est.ravel()
    multiple_est = PatchOp * multiple_est.ravel()
    primary_est = np.reshape(primary_est, (nr, nt))
    multiple_est = np.reshape(multiple_est, (nr, nt))

    return primary_est, multiple_est, filt_est
