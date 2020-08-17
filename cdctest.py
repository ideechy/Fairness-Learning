import numpy as np
import utils
from kde import KernelDensityEstimation
from cdcstats import CDCStats


class CDCTest(object):
    def __init__(self, x, y, z, num_bootstrap=99, kernel_type='gauss', bandwidth=None,
                 index=1, seed=1, numba=True):
        if kernel_type in {'gauss', 'rectangle'}:
            self.kernel_type = kernel_type
        else:
            self.kernel_type = 'gauss'
        kde_z = KernelDensityEstimation(utils.as_matrix(z), bandwidth)
        self.kernel = kde_z.compute_kernel_density_estimate()
        self.dist_x = utils.compute_distance_matrix(utils.as_matrix(x), index)
        self.dist_y = utils.compute_distance_matrix(utils.as_matrix(y), index)
        assert self.dist_x.shape == self.dist_y.shape == self.kernel.shape
        self.stats = CDCStats(self.dist_x, self.dist_y, self.kernel, numba=numba)
        self.cdcov_stats = 0.
        self.B = num_bootstrap
        self.permuted_cdcov_stats = np.zeros(self.B)
        self.seed = seed
        self.p_value = 0.

    def cdcov(self):
        self.cdcov_stats = self.stats.get_stats(stats_type='cov')
        return self.cdcov_stats, self.stats.cdcov

    def cdcorr(self):
        cdcorr_stats = self.stats.get_stats(stats_type='corr')
        return cdcorr_stats, self.stats.cdcorr

    def conduct_conditional_independence_test(self):
        self.cdcov_stats = self.stats.get_stats()

        index = utils.generate_random_sample_index(self.B, self.kernel, self.seed)

        for i in range(self.B):
            bootstrap_dist_x = utils.rearrange_matrix(self.dist_x, index[i])
            self.stats.set_dist_x(bootstrap_dist_x)
            self.permuted_cdcov_stats[i] = self.stats.get_stats()

        self.p_value = (1 + np.sum(self.permuted_cdcov_stats >= self.cdcov_stats)) / (1 + self.B)
        self.stats.set_dist_x(self.dist_x)
