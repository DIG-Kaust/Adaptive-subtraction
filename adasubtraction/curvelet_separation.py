import numpy as np
from curvelops import FDCT2D

def prox_data(x, rho):
    return np.maximum( np.abs( x ) - rho, 0. ) * np.sign(x)       # close-form solution of proximal algorithm when f=|.|_1

def curvelet_separation(Cop, b, b2, eps1, eps2, eta, niter=10):
    """Apply iterative bayesian separation using the curvelet transform, 
    returning corrected multiples and primaries

    Parameters
    ----------
    Cop : :obj:`curvelops.curvelops.FDCT2D`
        Fast Discrete Curvelet Transform Operator
    b : :obj:`np.ndarray`
        Total Data
    b2 : :obj:`np.ndarray`
        Predicted multiples
    eps1 : :obj:`float`
        Primaries sparsity parameter in the curvelet domain
    eps2 : :obj:`float`
        Multiples sparsity parameter in the curvelet domain
    eta : :obj:`float`
        Tradeoff between fitting the total data and fitting the predicted multiples parameter
    niter : :obj:`int`, optional
        Number of iterations

    Returns
    ---------
    s1 : :obj:`np.ndarray`
        Corrected primaries

    Note
    -------
    This algorithm is taken from the work of Saab et al. (2007)

    """
    b1 = b - b2
    # Create positive sparsifying weights
    w1 = np.maximum(eps1 * np.abs(Cop @ b2), 1e-2*np.mean(np.abs(Cop @ b2)))   # avoid too small values for the weights
    w2 = np.maximum(eps2 * np.abs(Cop @ b1), 1e-2*np.mean(np.abs(Cop @ b1)))
    # Initialize coefficients x1 and x2
    x1 = Cop @ b1
    x2 = Cop @ b2
    # Iteratively update solution
    for i in range(niter):
        
        x1_old = x1.copy()

        x1 = prox_data(Cop @ b2 - Cop @ Cop.H @ x2 + Cop @ b1 - Cop @ Cop.H @ x1 + x1, w1/(2*eta))

        x2 = prox_data(Cop @ b2 - Cop @ Cop.H @ x2 + x2 + (eta/(eta + 1))*(Cop @ b1 - Cop @ Cop.H @ x1_old), w2/(2*(1+eta)))

    s1 = np.real(Cop.H @ x1)

    return s1

