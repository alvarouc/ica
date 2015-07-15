# Import ica function
from ica import ica1
import numpy as np
import matplotlib.pyplot as plt
import time

@profile
def main():
    # Define matrix dimensions
    Nobs = 1000 # Number of observation
    Nvars = 50000 # Number of variables
    Ncomp = 100 # Number of components

    # Simulated true sources
    S_true = np.random.logistic(0,1,(Ncomp,Nvars))
    # Simulated true mixing
    A_true = np.random.normal(0,1,(Nobs,Ncomp))
    # X = AS
    X = np.dot(A_true,S_true)
    # add some noise
    X = X + np.random.normal(0,1,X.shape)
    # apply ICA on X and ask for 2 components

    start = time.time()
    A,S = ica1(X,Ncomp)
    total = start - time.time()
    print('total time: {}'.format(total))
    # compare if our estimates are accurate
    # correlate A with Atrue and take 
    aCorr = np.abs(np.corrcoef(A.T,A_true.T)[:Ncomp,Ncomp:]).max(axis = 0).mean()
    sCorr = np.abs(np.corrcoef(S,S_true)[:Ncomp,Ncomp:]).max(axis = 0).mean()

    print "Accuracy of estimated sources: %.2f"%sCorr
    print "Accuracy of estimated mixing: %.2f"%aCorr

if __name__=="__main__":
    main()