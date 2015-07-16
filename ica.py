'''
Independent Component Analysis (ICA):
This script computes ICA using the INFOMAX criteria.
The preprocessing steps include demeaning and whitening.
'''
import numpy as np
from numpy import dot
from numpy.linalg import svd, matrix_rank, pinv, inv
from numpy.random import permutation
from scipy.linalg import eigh

# Theano Imports
import theano.tensor as T
import theano
from theano import shared

# Global constants
EPS = 1e-18
MAX_W = 1e8
ANNEAL = 0.9
MAX_STEP = 500
MIN_LRATE = 1e-6
W_STOP = 1e-6

class ica:

    def __init__(self, n_comp=10):

        # Theano initialization 
        self.T_weights = shared(np.eye(n_comp, dtype=np.float32))
        #T.addbroadcast(self.T_weights, 0)
        self.T_bias = shared(np.ones((n_comp,1), dtype=np.float32))
        
        T_p_x_white = T.fmatrix()
        T_lrate = T.fscalar()
        T_block = T.fscalar()
        T_unmixed = T.dot(self.T_weights,T_p_x_white) + T.addbroadcast(self.T_bias,1)
        T_logit = 1 - 2 / (1 + T.exp(-T_unmixed))

        T_out =  self.T_weights +  T_lrate * T.dot(T_block * T.identity_like(self.T_weights) + T.dot(T_logit, T.transpose(T_unmixed)), self.T_weights)
        T_bias_out = self.T_bias + T_lrate * T.reshape(T_logit.sum(axis=1), (-1,1))
        T_max_w = T.max(self.T_weights)
        T_isnan = T.any(T.isnan(self.T_weights))

        self.w_up_fun = theano.function([T_p_x_white, T_lrate, T_block],
                                        [T_max_w, T_isnan],
                                        updates=[(self.T_weights, T_out),
                                                 (self.T_bias, T_bias_out)],
                                        allow_input_downcast=True)

        #T_out = T.dot(T_weights,T.transpose(T_weights))/T_lrate
        #self.cov_fun = theano.function([T_weights, T_lrate], T_out, allow_input_downcast=True)

        
        self.loading = None
        self.sources = None
        self.weights = None
        self.n_comp = n_comp

    def fit(self, x2d):
        self.loading, self.sources, self.weights = ica1(x2d, self.n_comp, self.T_weights, self.T_bias, self.w_up_fun)
        return(self.loading, self.sources)


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
    x2d_demean = x2d - x2d.mean(axis=1).reshape((-1,1))
    cov = dot(x2d_demean, x2d_demean.T) / ( x2d.shape[1] -1 )
    #cov = cov_fun(x2d_demean, NVOX-1)
    w, v = eigh(cov,eigvals=(NSUB-n_comp,NSUB-1))
    D = np.diag(1./(np.sqrt(w ) ))
    white = dot(D,v.T)
    D = np.diag( np.sqrt(w))
    dewhite = dot(v,D)
    x_white = dot(white,x2d_demean)
    return (x_white, white, dewhite)

@profile
def w_update(x_white, lrate1, T_weights, T_bias, w_up_fun):
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
    error = 0
    NVOX = x_white.shape[1]
    NCOMP = x_white.shape[0]
    block1 = int(np.floor(np.sqrt(NVOX / 3)))
    permute1 = permutation(NVOX)
    p_x_white = x_white[:, permute1].astype(np.float32)

    for start in range(0, NVOX, block1):
        if start + block1 < NVOX:
            tt2 = start + block1
        else:
            tt2 = NVOX
            block1 = NVOX - start

        max_w, isnan = w_up_fun(p_x_white[:,start:tt2], lrate1, block1)

        # Checking if W blows up
        if isnan or max_w > MAX_W:
            print "Numeric error! restarting with lower learning rate"
            lrate1 = lrate1 * ANNEAL
            T_weights.set_value(np.eye(NCOMP, dtype=np.float32))
            T_bias.set_value( np.zeros((NCOMP, 1), dtype=np.float32))
            error = 1

            if lrate1 > 1e-6 and \
               matrix_rank(x_white) < NCOMP:
                print("Data 1 is rank defficient"
                      ". I cannot compute " +
                      str(NCOMP) + " components.")
                return (0, 1)

            if lrate1 < 1e-6:
                print("Weight matrix may"
                      " not be invertible...")
                return (0, 1)
        

    return(lrate1, error)


# infomax1: single modality infomax
def infomax1(x_white,  T_weights, T_bias, w_up_fun, verbose=False):
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
    T_weights.set_value( np.eye(NCOMP, dtype=np.float32))
    old_weights = np.eye(NCOMP)
    d_weigths = np.zeros(NCOMP)
    old_d_weights = np.zeros(NCOMP)
    lrate = 0.005 / np.log(NCOMP)
    T_bias.set_value(np.zeros((NCOMP, 1), dtype=np.float32))
    change = 1
    angle_delta = 0
    if verbose:
        print "Beginning ICA training..."
    step = 1

    while step < MAX_STEP and change > W_STOP:

        (lrate, error) = w_update( x_white, lrate, T_weights, T_bias, w_up_fun)

        if error != 0:
            step = 1
            error = 0
            lrate = lrate * ANNEAL
            T_weights.set_value( np.eye(NCOMP, dtype=np.float32))
            # weights = np.eye(NCOMP)
            old_weights = np.eye(NCOMP)
            d_weigths = np.zeros(NCOMP)
            old_d_weights = np.zeros(NCOMP)
            T_bias.set_value(np.zeros((NCOMP, 1), dtype=np.float32))
        else:
            d_weigths = T_weights.get_value() - old_weights
            change = np.linalg.norm(d_weigths, 'fro')**2

            if step > 2:
                angle_delta = np.arccos(np.sum(d_weigths * old_d_weights) /
                                        (np.linalg.norm(d_weigths, 'fro')) /
                                        (np.linalg.norm(old_d_weights, 'fro')))
                angle_delta = angle_delta * 180 / np.pi

            old_weights = np.copy(T_weights.get_value())

            if angle_delta > 60:
                lrate = lrate * ANNEAL
                old_d_weights = np.copy(d_weigths)
            elif step == 1:
                old_d_weights = np.copy(d_weigths)

            if (verbose and step % 10 == 0) or change < W_STOP:
                print("Step %d: Lrate %.1e,"
                      "Wchange %.1e,"
                      "Angle %.2f" % (step, lrate,
                                      change, angle_delta))

        step = step + 1

    # A,S,W
    return (inv(T_weights.get_value()), dot(T_weights.get_value(), x_white), T_weights.get_value())

# Single modality ICA


def ica1(x_raw, ncomp, T_weights, T_bias, w_up_fun, verbose=True):
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
    mixer, sources, weights = infomax1(x_white,  T_weights, T_bias, w_up_fun, verbose)
    mixer = dot(dewhite, mixer)
    if verbose:
        print "Done."
    return (mixer, sources, weights)


def icax(x_raw, ncomp, verbose=True):

    if verbose:
        print "Whitening data..."
    x_white, _, dewhite = pca_whiten(x_raw, ncomp)

    mixer_list = []
    sources_list = []
    for it in range(10):
        if verbose:
            print 'Run number %d' % it
            print "Running INFOMAX-ICA ..."
        mixer, sources, _ = infomax1(x_white, verbose)
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
