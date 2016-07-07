# Independent Component Analysis

Python version of INFOMAX Independent component
analysis. Ported from
http://sccn.ucsd.edu/~scott/ica.html

## Installation
    pip install ica

## How to use
    from ica import ica1
    A,S,W = ica1(X, n_components)

Where, ica1 is the infomax ICA in function format. The input `X` is a numpy array and `n_components` is the number of components to estimate

See use example in <a href="ICA-DEMO.ipynb">here</a>

Minimum Requirements

- Numpy >= 1.9.2 www.numpy.org
- Scipy >= 0.15.1 www.scipy.org

Prefered 

- Theano >= 0.7
- matplotlib >= 1.4.3
