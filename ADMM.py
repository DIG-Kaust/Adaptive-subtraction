import numpy as np
from pylops.optimization.sparsity import FISTA
from pylops import LinearOperator
from scipy.sparse.linalg import lobpcg as sp_lobpcg
from scipy.sparse.linalg import eigsh as sp_eigsh


def prox_data(x, rho):
    return np.maximum( np.abs( x ) - rho, 0. ) * np.sign(x)


def ADMM(Op, b, rho, nouter, ninner, eps, x_true=None, decay=None):
    """ADMM for L1-L1

    ADMM algorithm to solve L1-L1 regression problems

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear operator
    data : :obj:`np.ndarray`
        data
    rho : :obj:`float`
        penalty parameter of the augmented Lagrangian term
    nouter : :obj:`int`
        Number of outer iterations
    ninner : :obj:`int`
        Number of inner iterations
    eps : :obj:`float`
        Lagrange multiplier
    x_true : :obj:`np.ndarray`, optional
        True solution (only used to compute the error through iterations)

    Returns
    -------
    x : :obj:`np.ndarray`
        final solution
    cost : :obj:`np.ndarray`, optional
        Error as a function of iterations. Only computed if the true solution x_true is provided. 

    """
    n = Op.shape[0]
    z = np.zeros(n)
    u = np.zeros(n)
    x = np.zeros(Op.shape[1])
    Op1 = LinearOperator(Op.H * Op, explicit=False)
    X = np.random.rand(Op1.shape[0], 1).astype(Op1.dtype)
    maxeig = sp_lobpcg(Op1, X=X, maxiter=10, tol=1e-10)[0][0]
    if decay is None:
        decay = np.ones(nouter)
    # maxeig = sp_eigsh(Op1, k=1, ncv=10, maxiter=10, tol=1e-2)[0]
    alpha = 1.0 / maxeig
    if x_true is not None:
        cost = np.zeros(nouter)
    for iiter in range(nouter):
        x = FISTA(Op, b + z - u, ninner, eps/rho, alpha=alpha, x0=x)[0]
        z = prox_data(Op * x - b + u, 1 / ( rho * decay[iiter] ))
        u = u + Op * x - z - b
        if x_true is not None:
            cost[iiter] = np.linalg.norm( x - x_true.ravel() ) / np.linalg.norm( x_true.ravel() )
    if x_true is None:
        return x
    else:
        return x, cost


def ADMM_curvelet(A, b, rho, outer_its, inner_its, eps):
    m, n = A.shape
    z = np.zeros(m)
    u = np.zeros(m)
    x = np.zeros(n)
    for i in range(outer_its):
        x = FISTA(A, b + z - u, inner_its, eps/rho, alpha=1, x0=x, show=False)[0]
        z = prox_data(A * x + u - b, 1/rho)
        u = u + ( A * x - z - b )
    return x

