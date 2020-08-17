import numpy as np
from scipy.spatial import distance
from utils import select_bandwidth_nrd0


class KernelDensityEstimation(object):
    def __init__(self, z, bandwidth=None, kernel_type='gauss'):
        self.z = z
        self.n = self.z.shape[0]
        self.d = self.z.shape[1]
        self.bandwidth = self.select_bandwidth() if bandwidth is None else np.asarray(bandwidth).squeeze()
        assert self.bandwidth.ndim <= 2
        if self.bandwidth.ndim == 0:
            self.bandwidth_value = self.bandwidth.item()
        elif self.bandwidth.ndim == 2:
            self.bandwidth_matrix = self.bandwidth
        else:
            if self.bandwidth.shape[0] == self.d:
                self.bandwidth_vector = self.bandwidth
        self.type = kernel_type

    def select_bandwidth(self):
        assert self.n >= 2
        if self.d == 1:
            width = select_bandwidth_nrd0(self.z)
        else:
            width = list(map(select_bandwidth_nrd0, self.z.transpose()))
            assert all([w > 0 for w in width])
        return np.asarray(width)

    def compute_kernel_density_estimate(self):
        if self.type == 'gauss':
            if 'bandwidth_value' in dir(self):
                return self.compute_gaussian_kernel_estimate_value()
            elif 'bandwidth_vector' in dir(self):
                return self.compute_gaussian_kernel_estimate_vector()
            else:
                return self.compute_gaussian_kernel_estimate_matrix()
        elif self.type == 'rectangle':
            return self.compute_rectangle_kernel_estimate()

    def compute_gaussian_kernel_estimate_vector(self):
        det = self.bandwidth_vector.prod()
        density = 1 / (pow(2 * np.pi, self.d / 2) * det)
        weight = 1/self.bandwidth_vector ** 2
        kde = np.ones((self.n, self.n)) * density
        for i in range(self.n):
            for j in range(i):
                weight_dist = distance.euclidean(self.z[i], self.z[j], weight) ** 2
                kde[i, j] *= np.exp(-0.5 * weight_dist)
                kde[j, i] = kde[i, j]
        return kde

    def compute_gaussian_kernel_estimate_value(self):
        det = self.bandwidth_value ** self.d
        density = 1 / (pow(2 * np.pi, self.d / 2) * np.sqrt(det))
        kde = np.ones((self.n, self.n)) * density

        for i in range(self.n):
            for j in range(i):
                weight_dist = distance.euclidean(self.z[i], self.z[j]) ** 2 / self.bandwidth_value
                kde[i, j] *= np.exp(-0.5 * weight_dist)
                kde[j, i] = kde[i, j]
        return kde

    def compute_gaussian_kernel_estimate_matrix(self):
        print(self.bandwidth)
        print('Matrix bandwidth not implemented.')
        pass

    def compute_rectangle_kernel_estimate(self):
        pass
