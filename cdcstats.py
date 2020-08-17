import numpy as np
from utils import weight_distance_anova, compute_condition_distance_covariance_numba


class CDCStats(object):
    def __init__(self, dist_x, dist_y, kernel, stats_type='cov', numba=True):
        self.dist_x = dist_x
        self.dist_y = dist_y
        self.kernel = kernel
        self.type = stats_type
        self.n = self.dist_x.shape[0]
        self.cdcov_stats = 0.
        self.cdcov = np.zeros(self.n)
        self.compute_cov = False
        self.cdcorr_stats = 0.
        self.cdcorr = np.zeros(self.n)
        self.compute_corr = False
        self.numba = numba

    def get_stats(self, stats_type=None):
        if stats_type is None:
            stats_type = self.type
        if stats_type == 'corr':
            return self.get_condition_distance_correlation()
        elif stats_type == 'cov':
            return self.get_condition_distance_covariance()
        else:
            raise('Stats type {:s} not recognized.'.format(type))

    def set_dist_x(self, dist_x):
        self.dist_x = dist_x
        self.compute_cov = False
        self.compute_corr = False

    def set_kernel(self, kernel):
        self.kernel = kernel
        self.compute_cov = False
        self.compute_corr = False

    def compute_condition_distance_covariance(self):
        k_margin = self.kernel.sum(axis=1)
        k = self.kernel/k_margin.reshape(-1, 1)
        for i in range(self.n):
            anova_x = weight_distance_anova(self.dist_x, k[i])
            anova_y = weight_distance_anova(self.dist_y, k[i])
            self.cdcov[i] = np.dot(anova_x * anova_y, k[i]).dot(k[i])
        self.cdcov_stats = 12 * self.cdcov.dot((k_margin / self.n) ** 4) / self.n
        self.compute_cov = True

    def get_condition_distance_covariance(self):
        if not self.compute_cov:
            if self.numba:
                self.cdcov, self.cdcov_stats = \
                    compute_condition_distance_covariance_numba(self.dist_x, self.dist_y, self.kernel)
            else:
                self.compute_condition_distance_covariance()
        return self.cdcov_stats

    def compute_condition_distance_correlation(self):
        k_margin = self.kernel.sum(axis=1)
        k = self.kernel / k_margin.reshape(-1, 1)
        for i in range(self.n):
            anova_x = weight_distance_anova(self.dist_x, k[i])
            anova_y = weight_distance_anova(self.dist_y, k[i])
            if not self.compute_cov:
                self.cdcov[i] = np.dot(anova_x * anova_y, k[i]).dot(k[i])
            cdcov_xx = np.dot(anova_x * anova_x, k[i]).dot(k[i])
            cdcov_yy = np.dot(anova_y * anova_y, k[i]).dot(k[i])
            p = cdcov_xx * cdcov_yy
            if p > 0:
                self.cdcorr[i] = self.cdcov[i]/np.sqrt(p)
        if not self.compute_cov:
            self.cdcov_stats = 12 * self.cdcov.dot((k_margin / self.n) ** 4) / self.n
            self.compute_cov = True
        self.cdcorr_stats = self.cdcorr.mean()
        self.compute_corr = True

    def get_condition_distance_correlation(self):
        if not self.compute_corr:
            self.compute_condition_distance_correlation()
        return self.cdcorr_stats
