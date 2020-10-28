import numpy as np
import scipy
from scipy.special import expit
import scipy.integrate as integrate
from scipy.stats import norm


def dat_gen_admission(n, intcp, beta_a, beta_s, lmbd):
    """Generating admission data.

    The sensitive attribute (S) is sex, the only non-sensitive attribute (A)
    is test score, thus the dimension of a is 1. Test score (A) depends on sex
    (S) if lmbd is not zero, and the decision (Y) depends on sex (S) if beta_s
    is not zero.

    Args:
        n: sample size.
        lmbd: historical disadvantage, male will have higher test score (A) on
            average if lmbd is positive.
        intcp: intercept in the logistic model for Y.
        beta_a: coefficient of A in the logistic model for Y.
        beta_s: selection bias, male will have a higher probability of
            admission if beta_s is positive.

    Returns:
        A tuple of three numpy.ndarrays (s, a, y).

        s: sensitive attributes, numpy.ndarray of size (n, 1).
        a: non-sensitive attributes, numpy.ndarray of size (n, 1).
        y: decisions, numpy.ndarray of size (n, 1).
    """
    # 1: advantageous race, 0: disadvantageous race
    s = np.random.binomial(1, p=0.5, size=n)
    eps = np.random.rand(n)
    a = np.clip(lmbd * s + eps, 0, 1)
    y = np.random.binomial(1, p=expit(intcp + beta_a * a + beta_s * s))
    return s.reshape(n, 1), a.reshape(n, 1), y.reshape(n, 1)


def dat_gen_admission_unfairness(intcp, beta_a, beta_s, lmbd):
    """Calculate the unfairness metric of admission data generator.

    """
    return integrate.quad(lambda x:
                          (expit(intcp + beta_a * np.clip(lmbd + x, 0, 1) + beta_s) -
                           expit(intcp + beta_a * np.clip(x, 0, 1))) * norm.pdf(x),
                          -np.inf, np.inf)[0]


def dat_gen_loan_univariate(n, intcp, beta_a, beta_s, lmbd_a, sigma_a=None,
                            adv_race_proportion=0.7):
    """Generating loan data.

    The sensitive attribute (S) is race, the non-sensitive attribute is
    log personal income (A).

    A depends on race (S) if lmbd_i and/or sigma_i are not zeros, and the
    decision (Y) depends on race (S) if beta_s is not zero.

    Args:
        n: sample size.
        intcp: intercept in the logistic model for Y.
        beta_a: coefficient of A in the logistic model for Y.
        beta_s: selection biases as an array of numbers, advantageous group
            (white) will have a higher probability of getting a loan if its
            corresponding element in beta_s (beta_s[0]) is larger than the
            elements of the disadvantageous groups (say, beta_s[1] of the
            black group).
        lmbd_a (float): advantageous group will have higher mean log income if
            positive.
        sigma_a (float): positive number, the log income of advantageous group
            will have larger standard deviation if greater than 1. If None,
            sigma_a is chosen such that the mean income would be the same
            across race groups.
        adv_race_proportion (float): proportion of the advantageous race group.

    Returns:
        A tuple of three numpy.ndarrays (s, a, y).

        s: sensitive attributes, numpy.ndarray of size (n, 1).
        a: non-sensitive attributes, numpy.ndarray of size (n, 1).
        y: decisions, numpy.ndarray of size (n, 1).
    """
    # 1: advantageous race, 0: disadvantageous race
    s = np.random.binomial(1, p=adv_race_proportion, size=n)
    u = np.random.randn(n)
    if sigma_a is None:
        # this sigma makes the means equal
        # sigma_a = np.sqrt(max(1 - 50 * lmbd_a, 0))
        sigma_a = 1
    i = 4 + lmbd_a * s + 0.2 * sigma_a ** s * u
    a = np.exp(i) / 100
    p = expit(intcp + beta_a * a + beta_s * s)
    y = np.random.binomial(1, p=p)
    return s.reshape(n, 1), a.reshape(n, 1), y.reshape(n, 1)
    

def dat_gen_reward_univariate_linear(s, a, y, intcp, beta_a, beta_s):
    n = len(y)
    p = expit(intcp + beta_a * a + beta_s * s)
    r_star = (np.random.binomial(1, p=p) - 0.5) * 2
    r = r_star * y
    return r_star.reshape(n, 1), r.reshape(n, 1)


def dat_gen_reward_univariate_interactive(s, a, y, intcp, beta_a, beta_s, beta_as):
    n = len(y)
    p = expit(intcp + beta_a * a + beta_s * s + beta_as * a * s)
    r_star = (np.random.binomial(1, p=p) - 0.5) * 2
    r = r_star * y
    return r_star.reshape(n, 1), r.reshape(n, 1)


def dat_gen_reward_univariate_quadratic(s, a, y, intcp, beta_a, beta_s, beta_as, beta_q):
    n = len(y)
    p = expit(intcp + beta_q * (beta_a * a + beta_s * s + beta_as * a * s) ** 2)
    r_star = (np.random.binomial(1, p=p) - 0.5) * 2
    r = r_star * y
    return r_star.reshape(n, 1), r.reshape(n, 1)


def dat_gen_reward_univariate_ncdf(s, a, y):
    n = len(y)
    p = norm.cdf(a) * pow(0.75, s)
    r_star = (np.random.binomial(1, p=p) - 0.5) * 2
    r = r_star * y
    return r_star.reshape(n, 1), r.reshape(n, 1)



def dat_gen_loan_univariate_unfairness(intcp, beta_a, beta_s, lmbd_a, sigma_a=1):
    """Calculate the unfairness metric of loan data generator.

    """
    return integrate.quad(lambda x:
                          (expit(intcp + beta_a * np.exp(4 + lmbd_a + 0.2 * sigma_a * x) / 100 + beta_s) -
                           expit(intcp + beta_a * np.exp(4 + 0.2 * x) / 100)) * norm.pdf(x),
                          -np.inf, np.inf)[0]


def dat_gen_loan_multivariate_wrapper(n, intcp, beta_e, beta_i, beta_s1=0,
                                      beta_s2=0, lmbd_e1=0, lmbd_e2=0,
                                      lmbd_i1=0, lmdb_i2=0, lmbd_e0=1.07,
                                      lmbd_i0=0.58, race_proportion=None):
    beta_s = [intcp, intcp + beta_s1, intcp + beta_s2]
    race_mean_edu = np.array([lmbd_e0, lmbd_e0 + lmbd_e1, lmbd_e0 + lmbd_e2])
    race_med_income = np.array([lmbd_i0, lmbd_i0 + lmbd_i1, lmbd_i0 + lmdb_i2])
    return dat_gen_loan_multivariate(n, beta_e, beta_i, beta_s, race_mean_edu,
                                     race_med_income, race_proportion)


def dat_gen_loan_multivariate_wrapper_unfairness(intcp, beta_e, beta_i, beta_s1=0,
                                                 beta_s2=0, lmbd_e1=0, lmbd_e2=0,
                                                 lmbd_i1=0, lmdb_i2=0, lmbd_e0=1.07,
                                                 lmbd_i0=0.58):
    p = np.zeros(3)
    beta_s = [intcp, intcp + beta_s1, intcp + beta_s2]
    race_mean_edu = np.array([lmbd_e0, lmbd_e0 + lmbd_e1, lmbd_e0 + lmbd_e2])
    e_sd = lmbd_e0 / 2.5
    race_med_income = np.array([lmbd_i0, lmbd_i0 + lmbd_i1, lmbd_i0 + lmdb_i2])
    for s in range(3):
        e_mean = race_mean_edu[s]
        i_med = race_med_income[s]
        p[s] = integrate.dblquad(lambda y, x: expit(beta_e * np.clip(e_mean + e_sd * x, 0, None) +
                                                    beta_i * i_med * np.exp(e_sd * x + 0.1 * y) +
                                                    beta_s[s]) * norm.pdf(x) * norm.pdf(y),
                                 -np.inf, np.inf, -np.inf, np.inf)[0]
    return np.max(p) - np.min(p)


def dat_gen_loan_multivariate(n, beta_e, beta_i, beta_s, race_mean_edu=None,
                              race_med_income=None, race_proportion=None):
    """Generating loan data.

    The sensitive attribute (S) is race, the non-sensitive attributes are
    education years (E) and personal income (I), thus the dimension of a is 2.

    E and I depend on race (S) if lmbd_e and lmbd_i are not zeros, and the
    decision (Y) depends on sex (S) if beta_s is not zero.

    Args:
        n: sample size.
        beta_e: coefficient of E in the logistic model for Y.
        beta_i: coefficient of I in the logistic model for Y.
        beta_s: selection biases as an array of numbers, advantage group
            (white) will have a higher probability of getting a loan if its
            corresponding element in beta_s (beta_s[0]) is larger than the
            elements of the disadvantage groups (say, beta_s[1] of the black
            group).
        race_proportion: proportion of white, black, asian people as an array
            of float numbers.
        race_mean_edu: mean education time (10 year) by race.
        race_med_income: median annual income (100,000 dollars) by race.

    Returns:
        A tuple of three numpy.ndarrays (s, a, y).

        s: sensitive attributes, numpy.ndarray of size (n, 1).
        a: non-sensitive attributes (E, I), numpy.ndarray of size (n, 2).
        y: decisions, numpy.ndarray of size (n, 1).
    """
    if race_proportion is None:
        race_proportion = np.array([0.76, 0.16, 0.08])
    else:
        race_proportion = np.array(race_proportion)
    if race_mean_edu is None:
        race_mean_edu = np.array([1.07, 0.99, 1.26])
    else:
        race_mean_edu = np.array(race_mean_edu)
    if race_med_income is None:
        race_med_income = np.array([0.58, 0.40, 0.81])
    else:
        race_med_income = np.array(race_med_income)
    assert race_proportion.ndim == 1 and \
        race_proportion.shape == race_mean_edu.shape == race_med_income.shape

    beta_s = np.asarray(beta_s)
    assert beta_s.ndim == 1 and beta_s.shape == race_proportion.shape

    # onehot encoded s
    s_onehot = np.random.multinomial(1, pvals=race_proportion, size=n)
    # s as labels
    s = np.argmax(s_onehot, axis=1)
    # education years e = e_mean + e_eps
    e_mean = race_mean_edu.take(s)
    e_sd = race_mean_edu[0] / 2.5
    e_eps = np.random.normal(0, e_sd, size=n)
    e = np.clip(e_mean + e_eps, 0, None)
    # income log(i) = log(i_med) + e_eps + i_eps
    i_med = race_med_income.take(s)
    i_mu = np.log(i_med) + e_eps
    i = np.random.lognormal(i_mu, 0.1)
    # conditional expectation of y
    p = expit(beta_e * e + beta_i * i + beta_s.take(s))
    y = np.random.binomial(1, p=p)
    # non-sensitive attributes
    a = np.column_stack((e, i))
    return s.reshape(n, 1), a, y.reshape(n, 1)

def dat_gen_reward_multivariate_linear(s, a, y, beta_e, beta_i, beta_s):
    n = len(y)
    beta_s = np.asarray(beta_s)
    assert beta_s.ndim == 1 and beta_s.shape[0] == len(np.unique(s))
    e, i = a[:, 0].reshape(n, 1), a[:, 1].reshape(n, 1)
    # reward
    p = expit(beta_e * e + beta_i * i + beta_s.take(s))
    assert p.size == n
    r_star = (np.random.binomial(1, p=p) - 0.5) * 2
    r = r_star * y
    assert r.size == n
    return r_star.reshape(n, 1), r.reshape(n, 1)

def dat_gen_reward_multivariate_quadratic(s, a, y, beta_e, beta_e2, beta_i, beta_i2, beta_ei, beta_s):
    n = len(y)
    beta_s = np.asarray(beta_s)
    assert beta_s.ndim == 1 and beta_s.shape[0] == len(np.unique(s))
    e, i = a[:, 0].reshape(n, 1), a[:, 1].reshape(n, 1)
    # reward
    p = expit(beta_e * e + beta_e2 * e**2 + beta_ei * e * i + beta_i * i \
        + beta_i2 * i**2 + beta_s.take(s))
    assert p.size == n
    r_star = (np.random.binomial(1, p=p) - 0.5) * 2
    r = r_star * y
    assert r.size == n
    return r_star.reshape(n, 1), r.reshape(n, 1)
