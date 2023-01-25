from pylops.basicoperators import Diagonal
from pylops.basicoperators import VStack

def IRLS(Op, data, nouter, ninner, epsR=1e-6, epsI=1, x0=None):
    
    if x0 is None:
        x0 = np.zeros(Op.shape[1])
        
    xnew = x0    
    # Regularization matrix
    bI = Diagonal(np.ones(Op.shape[1])*epsI)
    # Build the stack of model and regularization
    Op = VStack([Op, bI])
    # Data term
    data = np.hstack([data, np.zeros(bI.shape[0])])
    # First estimate of the residual
    r = data - Op * xnew

    for iiter in range(nouter):
        
        xold = xnew
        # Update the weight matrix
        rw = 1. / np.maximum(np.abs(r), epsR)
        R = Diagonal(rw)
        # Update the estimate of the solution
        xnew = RegularizedInversion(Op, None, data, Weight=R,
                                        returninfo=False, 
                                        **dict(iter_lim=ninner, x0=xold))
        # Compute the new residual                          
        r = data - Op * xnew
        
    return xnew