'''
Independent Component Analysis (ICA):
This script computes ICA using the INFOMAX criteria.
The preprocessing steps include demeaning and whitening.
'''
import torch
import torch.linalg as tla
import numpy as np

# Global constants
EPS = 1e-16
MAX_W = 1e8
ANNEAL = 0.9
MAX_STEP = 500
MIN_LRATE = 1e-6
W_STOP = 1e-6


def norm(x):
    """Computes the norm of a vector or the Frobenius norm of a
    matrix_rank

    """
    return torch.norm(x.ravel())


class ica:

    def __init__(self, n_components=10):
        self.n_comp = n_components

    def fit(self, x2d):
        x_white, self.white, self.dewhite\
            = pca_whiten(x2d, self.n_comp)
        self.mix, self.sources, self.unmix\
            = infomax1(x_white, self.n_comp)
        return self.mix, self.sources


def diagsqrts(w):
    """
    Returns direct and inverse square root normalization matrices
    """
    Di = torch.diag(1. / (torch.sqrt(w) + torch.finfo(float).eps))
    D = torch.diag(torch.sqrt(w))
    return D, Di


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
    x2d_demean = x2d - x2d.mean(axis=1).reshape((-1, 1))
    NSUB, NVOX = x2d_demean.shape
    if NSUB > NVOX:
        cov = torch.matmul(x2d_demean.T, x2d_demean) / (NSUB - 1)
        w, v = torch.eigh(cov, eigvals=(NVOX - n_comp, NVOX - 1))
        D, Di = diagsqrts(w)
        u = torch.matmul(torch.matmul(x2d_demean, v), Di)
        x_white = v.T
        white = torch.matmul(Di, u.T)
        dewhite = torch.matmul(u, D)
    else:
        cov = torch.matmul(x2d_demean, x2d_demean.T) / (NVOX - 1)
        w, u = tla.eigh(cov)
        w = w[(NSUB - n_comp):]
        u = u[:, (NSUB - n_comp):]
        D, Di = diagsqrts(w)
        white = torch.matmul(Di, u.T)
        x_white = torch.matmul(white, x2d_demean)
        dewhite = torch.matmul(u, D)
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
    device = weights.device
    NCOMP, NVOX = x_white.shape
    block1 = int(np.floor(np.sqrt(NVOX / 3)))
    permute1 = torch.randperm(NVOX)
    for start in range(0, NVOX, block1):
        if start + block1 < NVOX:
            tt2 = start + block1
        else:
            tt2 = NVOX
            block1 = NVOX - start

        unmixed = torch.matmul(
            weights, x_white[:, permute1[start:tt2]]) + bias1
        logit = 1 - (2 / (1 + torch.exp(-unmixed)))
        weights = weights + lrate1 * torch.matmul(block1 * torch.eye(NCOMP, device=device) +
                                                  torch.matmul(logit, unmixed.T), weights)
        bias1 = bias1 + lrate1 * logit.sum(axis=1).reshape(bias1.shape)
        # Checking if W blows up
        if (torch.isnan(weights)).any() or torch.max(torch.abs(weights)) > MAX_W:
            print("Numeric error! restarting with lower learning rate")
            lrate1 = lrate1 * ANNEAL
            weights = torch.eye(NCOMP, device=device)
            bias1 = torch.zeros((NCOMP, 1), device=device)
            error = 1

            if lrate1 > 1e-6 and \
               tla.matrix_rank(x_white) < NCOMP:
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

    return (weights, bias1, lrate1, error)

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
    device = x_white.device
    NCOMP = x_white.shape[0]
    # Initialization
    weights = torch.eye(NCOMP, device=device)
    old_weights = torch.eye(NCOMP, device=device)
    d_weigths = torch.zeros(NCOMP, device=device)
    old_d_weights = torch.zeros(NCOMP, device=device)
    lrate = 0.005 / np.log(NCOMP)
    bias = torch.zeros((NCOMP, 1), device=device)
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
            weights = torch.eye(NCOMP, device=device)
            old_weights = torch.eye(NCOMP, device=device)
            d_weigths = torch.zeros(NCOMP, device=device)
            old_d_weights = torch.zeros(NCOMP, device=device)
            bias = torch.zeros((NCOMP, 1), device=device)
        else:
            d_weigths = weights - old_weights
            change = norm(d_weigths)**2

            if step > 2:
                angle_delta = torch.arccos(
                    torch.sum(d_weigths * old_d_weights) /
                    (norm(d_weigths) * norm(old_d_weights) + 1e-8)
                ) * 180 / np.pi

            old_weights = weights.clone()

            if angle_delta > 60:
                lrate = lrate * ANNEAL
                old_d_weights = d_weigths.clone()
            elif step == 1:
                old_d_weights = d_weigths.clone()

            if verbose and change < W_STOP:
                print("Step %d: Lrate %.1e,"
                      "Wchange %.1e,"
                      "Angle %.2f" % (step, lrate,
                                      change, angle_delta))

        step = step + 1

    # A,S,W
    return (tla.inv(weights), torch.matmul(weights, x_white), weights)

# Single modality ICA


def ica1(x_raw, ncomp, verbose=False):
    '''
    Single modality Independent Component Analysis
    '''
    device = x_raw.device
    if verbose:
        print("Whitening data...")
    x_white, _, dewhite = pca_whiten(x_raw, ncomp)
    if verbose:
        print('x_white shape: %d, %d' % x_white.shape)
        print("Done.")
    if verbose:
        print("Running INFOMAX-ICA ...")
    mixer, sources, unmixer = infomax1(x_white, verbose)
    mixer = torch.matmul(dewhite, mixer)

    scale = sources.std(axis=1).reshape((-1, 1))
    sources = sources / scale
    scale = scale.reshape((1, -1))
    mixer = mixer * scale

    if verbose:
        print("Done.")
    return (mixer, sources, unmixer)


def icax(x_raw, ncomp, verbose=True):
    device = x_raw.device
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
        cor_m = torch.corrcoef(S1, S2)[:ncomp, ncomp:]
        idx = torch.argmax(np.abs(cor_m), axis=1)
        S2 = S2[idx, :]
        A2 = A2[:, idx]
        cor_m = torch.corrcoef(S1, S2)[:ncomp, ncomp:]
        S2 = S2 * torch.sign(torch.diag(cor_m)).reshape((ncomp, 1))
        A2 = A2 * torch.sign(torch.diag(cor_m)).reshape((1, ncomp))
        sources_list[it] = S2
        mixer_list[it] = A2

    # Average sources
    temp_sources = torch.zeros(sources.shape, device=device)
    temp_mixer = torch.zeros(mixer.shape, device=device)
    for sources, mixer in zip(sources_list, mixer_list):
        temp_sources = temp_sources + sources
        temp_mixer = temp_mixer + mixer

    temp_sources = temp_sources / 10.0
    temp_mixer = temp_mixer / 10.0

    return (temp_mixer, temp_sources)
