# Import ica function
from ica import ica1
import numpy as np
import matplotlib.pyplot as plt

# Define matrix dimensions
Nobs = 100 # Number of observations
Nvars = 10000 # Number of variables
Ncomp = 2 # Number of components

# Simulated true sources
S_true = np.random.logistic(0,1,(Ncomp,Nvars))
# Simulated true mixing
A_true = np.random.normal(0,1,(Nobs,Ncomp))
# X = AS
X = np.dot(A_true,S_true)
# add some noise
X = X + np.random.normal(0,5,X.shape)
# apply ICA on X and ask for 2 components
A,S = ica1(X,2)
# compare if our estimates are accurate
# correlate A with Atrue and take 
aCorr = np.abs(np.corrcoef(A.T,A_true.T)[:Ncomp,Ncomp:]).max(axis = 0).mean()
sCorr = np.abs(np.corrcoef(S,S_true)[:Ncomp,Ncomp:]).max(axis = 0).mean()

print "Accuracy of estimated sources: %.2f"%sCorr
print "Accuracy of estimated mixing: %.2f"%aCorr
