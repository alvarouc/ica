'''
Independent Component Analysis (ICA):
This script computes ICA using the INFOMAX criteria.
The preprocessing steps include demeaning and whitening.
'''
import numpy as np
from numpy import dot
from numpy.linalg import matrix_rank, inv
from numpy.random import permutation
from scipy.linalg import eigh

# Global constants
EPS = 1e-18
MAX_W = 1e8
ANNEAL = 0.9
MAX_STEP = 500
MIN_LRATE = 1e-6
W_STOP = 1e-6


class ica:

    def __init__(self, n_components=10):
        self.mix = None
        self.sources = None
        self.unmix = None
        self.n_comp = n_components

    def fit(self, x2d):
        self.mix, self.sources, self.unmix = ica1(x2d, self.n_comp)
        return(self.mix, self.sources)


def pca_whiten(x2d, n_comp, verbose=True):
    """ data Whitening
    *Input
    x2d : 2d data matrix of observations by variables
    n_comp: Number of components to retain
    *Output
    Xwhite : Whitened X
    white : whitening matrix (Xwhite = np.dot(white,X))
    dewhite : dewhitening matrix (X = np.dot(dewhite,Xwhite))
    """
    NSUB, NVOX = x2d.shape
    x2d_demean = x2d - x2d.mean(axis=1).reshape((-1, 1))
    cov = dot(x2d_demean, x2d_demean.T) / (x2d.shape[1] - 1)
    w, v = eigh(cov, eigvals=(NSUB-n_comp, NSUB-1))
    D = np.diag(1. / (np.sqrt(w)))
    white = dot(D, v.T)
    D = np.diag(np.sqrt(w))
    dewhite = dot(v, D)
    x_white = dot(white, x2d_demean)
    return (x_white, white, dewhite)


def w_update(weights, x_white, bias1, lrate1):
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
    NVOX = x_white.shape[1]
    NCOMP = x_white.shape[0]
    block1 = int(np.floor(np.sqrt(NVOX / 3)))
    permute1 = permutation(NVOX)
    for start in range(0, NVOX, block1):
        if start + block1 < NVOX:
            tt2 = start + block1
        else:
            tt2 = NVOX
            block1 = NVOX - start

        # unmixed = dot(weights, x_white[:, permute1[start:tt2]]) + \
        #     dot(bias1, ib1[:, 0:block1])
        unmixed = dot(weights, x_white[:, permute1[start:tt2]]) + bias1
        logit = 1 - (2 / (1 + np.exp(-unmixed)))
        weights = weights + lrate1 * dot(block1 * np.eye(NCOMP) +
                                         dot(logit, unmixed.T), weights)
        bias1 = bias1 + lrate1 * logit.sum(axis=1).reshape(bias1.shape)
        # Checking if W blows up
        if (np.isnan(weights)).any() or np.max(np.abs(weights)) > MAX_W:
            print("Numeric error! restarting with lower learning rate")
            lrate1 = lrate1 * ANNEAL
            weights = np.eye(NCOMP)
            bias1 = np.zeros((NCOMP, 1))
            error = 1

            if lrate1 > 1e-6 and \
               matrix_rank(x_white) < NCOMP:
                print("Data 1 is rank defficient"
                      ". I cannot compute " +
                      str(NCOMP) + " components.")
                return (None, None, None, 1)

            if lrate1 < 1e-6:
                print("Weight matrix may"
                      " not be invertible...")
                return (None, None, None, 1)
            break
        else:
            error = 0

    return(weights, bias1, lrate1, error)

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
    NCOMP = x_white.shape[0]
    # Initialization
    weights = np.eye(NCOMP)
    old_weights = np.eye(NCOMP)
    d_weigths = np.zeros(NCOMP)
    old_d_weights = np.zeros(NCOMP)
    lrate = 0.005 / np.log(NCOMP)
    bias = np.zeros((NCOMP, 1))
    change = 1
    angle_delta = 0
    if verbose:
        print("Beginning ICA training...")
    step = 1

    while step < MAX_STEP and change > W_STOP:

        (weights, bias, lrate, error) = w_update(weights, x_white, bias, lrate)

        if error != 0:
            step = 1
            error = 0
            lrate = lrate * ANNEAL
            weights = np.eye(NCOMP)
            old_weights = np.eye(NCOMP)
            d_weigths = np.zeros(NCOMP)
            old_d_weights = np.zeros(NCOMP)
            bias = np.zeros((NCOMP, 1))
        else:
            d_weigths = weights - old_weights
            change = np.linalg.norm(d_weigths.ravel())**2

            if step > 2:
                angle_delta = np.arccos(np.sum(d_weigths * old_d_weights) /
                                        np.sqrt((d_weigths**2).sum()) /
                                        np.sqrt((old_d_weights**2).sum()))
                angle_delta = angle_delta * 180 / np.pi

            old_weights = np.copy(weights)

            if angle_delta > 60:
                lrate = lrate * ANNEAL
                old_d_weights = np.copy(d_weigths)
            elif step == 1:
                old_d_weights = np.copy(d_weigths)

            if (verbose and step % 1 == 0) or change < W_STOP:
                print("Step %d: Lrate %.1e,"
                      "Wchange %.1e,"
                      "Angle %.2f" % (step, lrate,
                                      change, angle_delta))

        step = step + 1

    # A,S,W
    return (inv(weights), dot(weights, x_white), weights)

# Single modality ICA


def ica1(x_raw, ncomp, verbose=False):
    '''
    Single modality Independent Component Analysis
    '''
    if verbose:
        print("Whitening data...")
    x_white, _, dewhite = pca_whiten(x_raw, ncomp)
    if verbose:
        print('x_white shape: %d, %d' % x_white.shape)
        print("Done.")
    if verbose:
        print("Running INFOMAX-ICA ...")
    mixer, sources, unmixer = infomax1(x_white, verbose)
    mixer = dot(dewhite, mixer)

    scale = sources.std(axis=1).reshape((-1, 1))
    sources = sources/scale
    scale = scale.reshape((1, -1))
    mixer = mixer*scale

    if verbose:
        print("Done.")
    return (mixer, sources, unmixer)


def icax(x_raw, ncomp, verbose=True):

    if verbose:
        print("Whitening data...")
    x_white, _, dewhite = pca_whiten(x_raw, ncomp)

    mixer_list = []
    sources_list = []
    for it in range(10):
        if verbose:
            print('Run number %d' % it)
            print("Running INFOMAX-ICA ...")
        mixer, sources, unmix = infomax1(x_white, verbose)
        mixer_list.append(mixer)
        sources_list.append(sources)

    # Reorder all sources to the order of the first
    S1 = sources_list[0]
    for it in range(1, 10):
        S2 = sources_list[it]
        A2 = mixer_list[it]
        cor_m = np.corrcoef(S1, S2)[:ncomp, ncomp:]
        idx = np.argmax(np.abs(cor_m), axis=1)
        S2 = S2[idx, :]
        A2 = A2[:, idx]
        cor_m = np.corrcoef(S1, S2)[:ncomp, ncomp:]
        S2 = S2 * np.sign(np.diag(cor_m)).reshape((ncomp, 1))
        A2 = A2 * np.sign(np.diag(cor_m)).reshape((1, ncomp))
        sources_list[it] = S2
        mixer_list[it] = A2

    # Average sources
    temp_sources = np.zeros(sources.shape)
    temp_mixer = np.zeros(mixer.shape)
    for sources, mixer in zip(sources_list, mixer_list):
        temp_sources = temp_sources + sources
        temp_mixer = temp_mixer + mixer

    temp_sources = temp_sources / 10.0
    temp_mixer = temp_mixer / 10.0

    return (temp_mixer, temp_sources)
