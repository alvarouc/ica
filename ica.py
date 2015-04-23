'''
Independent Component Analysis (ICA):
This script computes ICA using the INFOMAX criteria.
The preprocessing steps include demeaning and whitening.
'''

import numpy as np
from numpy import dot
from numpy.linalg import svd, matrix_rank, pinv
from numpy.random import permutation
# Global constants
EPS = 1e-18
MAX_W = 1e8
ANNEAL = 0.9
MAX_STEP = 500
MIN_LRATE = 1e-6
W_STOP = 1e-6

# Common functions


def pca_whiten(x2d, n_comp, verbose=False):
    """ data Whitening
    *Input
    x2d : 2d data matrix of observations by variables
    n_comp: Number of components to retain
    *Output
    Xwhite : Whitened X
    white : whitening matrix (Xwhite = np.dot(white,X))
    dewhite : dewhitening matrix (X = np.dot(dewhite,Xwhite))
    """
    x_u, x_s, x_v = svd(x2d, full_matrices=False)
    if verbose:
        print "PCA whitening: %.2f%% retained variance" % \
            (100*(sum(x_s[:n_comp])/sum(x_s)))
    x_u, x_v, x_s = (x_u[:, :n_comp], x_v[:n_comp, :], x_s[:n_comp])
    white = dot(np.diag((x_s+EPS)**(-1)), x_u.T)
    dewhite = dot(x_u, np.diag(x_s))
    x_white = x_v
    return (x_white, white, dewhite)


def w_update(unmixer, x_white, bias1, lrate1):
    """ Update rule for infomax
    This function recieves parameters to update W1
    * Input
    W1: unmixing matrix (must be a square matrix)
    Xwhite1: whitened data
    bias1: current estimated bias
    lrate1: current learning rate
    startW1: in case update blows up it will start again from startW1
    * Output
    W1: updated mixing matrix
    bias: updated bias
    lrate1: updated learning rate
    """
    nvox1 = x_white.shape[1]
    ncomp1 = x_white.shape[0]
    block1 = int(np.floor(np.sqrt(nvox1/3)))
    ib1 = np.ones((1, block1))
    permute1 = permutation(nvox1)
    for start in range(0, nvox1, block1):
        if start+block1 < nvox1:
            tt2 = start+block1
        else:
            tt2 = nvox1
            block1 = nvox1 - start

        unmixed = dot(unmixer, x_white[:, permute1[start:tt2]]) + \
            dot(bias1, ib1[:, 0:block1])
        logit = 1/(1 + np.exp(-unmixed))
        unmixer = unmixer + lrate1*dot(block1*np.eye(ncomp1) +
                                       dot(1-2*logit, unmixed.T), unmixer)
        bias1 = (bias1.T + lrate1*(1-2*logit).sum(axis=1)).T
        # Checking if W blows up
        if np.isnan(np.sum(unmixer)) or np.max(np.abs(unmixer)) > MAX_W:
            print "Numeric error! restarting with lower learning rate"
            lrate1 = lrate1 * ANNEAL
            unmixer = np.eye(ncomp1)
            bias1 = np.zeros((ncomp1, 1))
            error = 1

            if lrate1 > 1e-6 and \
               matrix_rank(x_white) < ncomp1:
                print("Data 1 is rank defficient"
                      ". I cannot compute " +
                      str(ncomp1) + " components.")
                return (None, None, None, 1)

            if lrate1 < 1e-6:
                print("Weight matrix may"
                      " not be invertible...")
                return (None, None, None, 1)
            break
        else:
            error = 0

    return(unmixer, bias1, lrate1, error)


# infomax1: single modality infomax
def infomax1(x_white, verbose=False):
    """Computes ICA infomax in whitened data
    Decomposes x_white as x_white=AS
    *Input
    x_white: whitened data (Use PCAwhiten)
    verbose: flag to print optimization updates
    *Output
    A : mixing matrix
    S : source matrix
    W : unmixing matrix
    """
    ncomp = x_white.shape[0]
    # Initialization
    unmixer = np.eye(ncomp)
    old_w = np.eye(ncomp)
    lrate = 0.005/np.log(ncomp)
    bias = np.zeros((ncomp, 1))
    if verbose:
        print "Beginning ICA training..."
    step = 1

    while step < MAX_STEP:

        (unmixer, bias, lrate, error) = w_update(unmixer, x_white, bias, lrate)
        if error == 0:
            wtchange = unmixer - old_w
            old_w = unmixer
            change = np.sum(wtchange**2)
            if step == 1:    # initializing variables
                oldwtchange = wtchange
                oldchange = change
            if step > 2:
                angle_delta = np.arccos(np.sum(oldwtchange*wtchange) /
                                        (np.sqrt(change*oldchange)+EPS))
                angle_delta = angle_delta*180/np.pi
                if angle_delta > 60:
                    lrate = lrate*ANNEAL
                    oldwtchange = wtchange
                    oldchange = change
                if (verbose and step % 10 == 0) or change < W_STOP:
                    print("Step %d: Lrate %.1e,"
                          "Wchange %.1e,"
                          "Angle %.2f" % (step, lrate,
                                          change, angle_delta))
            # Stopping rule
            if step > 2 and change < W_STOP:
                step = MAX_STEP

            step = step + 1
        else:
            step = 1
    # A,S,W
    return (pinv(unmixer), dot(unmixer, x_white), unmixer)

# Single modality ICA


def ica1(x_raw, ncomp, verbose=True):
    '''
    Single modality Independent Component Analysis
    '''
    if verbose:
        print "Whitening data..."
    x_white, _, dewhite = pca_whiten(x_raw, ncomp)
    if verbose:
        print "Done."
    if verbose:
        print "Running INFOMAX-ICA ..."
    mixer, sources, _ = infomax1(x_white, verbose)
    mixer = dot(dewhite, mixer)
    if verbose:
        print "Done."
    return (mixer, sources)
