import numpy as np
import numba as nb


def as_matrix(x):
    x = np.asarray(x).squeeze()
    return x.reshape((x.shape + (1,) * 2)[:2])


def euclidean_distance(x, y):
    return np.sum(np.abs(y - x) ** 2, axis=-1) ** (1./2)


def compute_distance_matrix(x, index, threshold=1000000):
    m, k = x.shape

    if m * m * k <= threshold:
        d = euclidean_distance(x[:, np.newaxis, :], x[np.newaxis, :, :])
    else:
        d = np.empty((m, m), dtype=float)
        for i in range(m):
            d[i, :] = euclidean_distance(x[i], x)

    if index != 1:
        d = d ** index
    return d


def weight_distance_anova(d, w):
    m = d.dot(w)
    s = m.dot(w)
    return d - m - m.reshape(-1, 1) + s


@nb.njit
def compute_condition_distance_covariance_numba(x, y, kernel):
    k_margin = kernel.sum(axis=1)
    k = kernel / k_margin.reshape(-1, 1)
    n = k.shape[0]
    cdcov = np.zeros(n)
    for i in range(n):
        m_x = x.dot(k[i])
        s_x = m_x.dot(k[i])
        m_y = y.dot(k[i])
        s_y = m_y.dot(k[i])
        anova_x = x - m_x - m_x.reshape(-1, 1) + s_x
        anova_y = y - m_y - m_y.reshape(-1, 1) + s_y
        cdcov[i] = np.dot(anova_x * anova_y, k[i]).dot(k[i])
    cdcov_stats = 12 * cdcov.dot((k_margin / n) ** 4) / n
    return cdcov, cdcov_stats


def generate_random_sample_index(b, p, seed=1):
    np.random.seed(seed)
    n = p.shape[0]
    index = np.zeros((b, n), dtype=int)
    for j in range(n):
        pvals = p[j]/np.sum(p[j])
        index[:, j] = np.random.multinomial(1, pvals=pvals, size=b).argmax(axis=1)
    return index


def rearrange_matrix(x, index):
    x = x[index]
    x = x.transpose()[index]
    return x.transpose()


def select_bandwidth_nrd0(x):
    n = x.shape[0]
    hi = x.std()
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    # if sys.version_info >= (3, 8):
    #     if not (lo := min(hi, iqr / 1.34)):
    #         (lo := hi) or (lo := abs(self.z_dist[0, 0])) or (lo := 1)
    # else:
    lo = min(hi, iqr / 1.34)
    if not lo:
        if hi:
            lo = hi
        elif x[0, 0]:
            lo = abs(x[0, 0])
        else:
            lo = 1
    return 0.9 * lo * n ** (-0.2)


def select_bandwidth_nrd(x):
    n = x.shape[0]
    hi = x.std()
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    lo = min(hi, iqr / 1.34)
    return 1.06 * lo * n ** (-0.2)
