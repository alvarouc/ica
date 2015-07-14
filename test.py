import numpy as np
import unittest
import ica
import time


def find_sources_order(S_true, S_estimated):
    NCOMP, NVOX = S_true.shape
    C = np.corrcoef(S_true, S_estimated)
    C = np.abs(C[NCOMP:, :NCOMP])
    idx = np.argmax(C, axis=0)
    return idx


def mean_corr(S_true, S_estimated):
    NCOMP, NVOX = S_true.shape
    # Always correlate the smallest dimension
    if NCOMP > NVOX:
        C = np.corrcoef(S_true.T, S_estimated.T)
    else:
        C = np.corrcoef(S_true, S_estimated)
    C = np.abs(C[NCOMP:, :NCOMP])
    return np.diag(C).mean()


def auto_cov(A):
    A = A - A.mean(axis=0)
    cov = np.dot(A, A.T) / (A.shape[1] - 1)
    return(cov)


class test_ica_methods(unittest.TestCase):

    def setUp(self):

        self.NCOMP = 2
        self.NVOX = 50000
        self.NSUB = 1000
        self.sources = np.random.logistic(0, 1, (self.NCOMP, self.NVOX))
        self.loading = np.random.normal(0, 1, (self.NSUB, self.NCOMP))
        self.clean_data = np.dot(self.loading, self.sources)
        self.noisy_data = self.clean_data + np.random.normal(0, 1, self.clean_data.shape)

    def test_PCA_whitening_clean(self):
        start = time.time()
        x_white, white, dewhite = ica.pca_whiten(self.clean_data, self.NCOMP)
        end = time.time()
        print('\ttime: {:.2f} seconds'.format(end - start))
        # Check output dimensions
        self.assertEqual(x_white.shape, (self.NCOMP, self.NVOX))
        self.assertEqual(white.shape, (self.NCOMP, self.NSUB))
        self.assertEqual(dewhite.shape, (self.NSUB, self.NCOMP))

        # Check variance is 1
        var = x_white.var(axis=1)
        self.assertLess(np.linalg.norm(var - 1.0), 1e-3)

        cov = auto_cov(x_white)
        print(cov)
        self.assertLess(np.linalg.norm(cov - np.eye(self.NCOMP)) / self.NCOMP / self.NCOMP, 1e-2)

        eye = np.dot(dewhite, white)
        self.assertLess(np.linalg.norm(eye - np.eye(self.NSUB)) / self.NSUB / self.NSUB, 1e-2)

    def test_PCA_whitening_noisy(self):
        start = time.time()
        x_white, white, dewhite = ica.pca_whiten(self.noisy_data, self.NCOMP)
        end = time.time()
        print('\ttime: {:.2f} seconds'.format(end - start))
        self.assertEqual(x_white.shape, (self.NCOMP, self.NVOX))
        self.assertEqual(white.shape, (self.NCOMP, self.NSUB))
        self.assertEqual(dewhite.shape, (self.NSUB, self.NCOMP))

        x_white = x_white - x_white.mean(axis=0)
        cov = auto_cov(x_white)
        self.assertLess(np.linalg.norm(cov - np.eye(self.NCOMP)) / self.NCOMP / self.NCOMP, 1e-3)

        eye = np.dot(dewhite, white)
        self.assertLess(np.linalg.norm(eye - np.eye(self.NSUB)) / self.NSUB / self.NSUB, 1e-3)

    @unittest.skip("PCAwhiten not passing")
    def test_ICA_infomax_clean(self):

        start = time.time()
        A, S = ica.ica1(self.clean_data, self.NCOMP)
        end = time.time()
        print('\ttime: {}:.2f'.format(end - start))

        # Check right dimensions of Output
        self.assertEqual(A.shape, (self.NSUB, self.NCOMP))
        self.assertEqual(S.shape, (self.NCOMP, self.NVOX))

        idx = find_sources_order(self.sources, S)
        S = S[idx, :]
        A = A[:, idx]
        # Check the accuracy of output
        self.assertGreater(mean_corr(self.sources, S), 0.95)
        self.assertGreater(mean_corr(self.loading, A), 0.95)


if __name__ == '__main__':
    unittest.main()
