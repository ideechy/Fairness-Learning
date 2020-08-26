import numpy as np
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from cdctest import CDCTest


class FairData(object):
    def __init__(self, s_train, a_train, y_train, preprocess_method='o', mode='predict'):
        """Initialization of data.

        Args:
            s_train (numpy.ndarray): categorical sensitive training attributes,
                must have shape (n, 1).
            a_train (numpy.ndarray): non-sensitive training attributes, must
                have shape (n, d).
            y_train (numpy.ndarray): binary decisions, must have shape (n, 1).
            preprocess_method (str): 'o' for orthogonalization, 'm' for
                marginal distribution mapping.

        """
        # check dimensions
        assert s_train.ndim == a_train.ndim == y_train.ndim == 2
        self.n, self.d = a_train.shape
        assert s_train.shape[0] == y_train.shape[0] == self.n
        assert s_train.shape[1] == y_train.shape[1] == 1
        # categories of sensitive attributes
        self.c = np.unique(s_train).size
        # one-hot encoding of sensitive attributes, shape=(n, c)
        self.s_encoder = OneHotEncoder(categories='auto')
        self.s_encoder.fit(s_train)
        self.s_train = self.s_encoder.transform(s_train).toarray()
        # non-sensitive attributes
        self.a_train = a_train
        # training target
        self.y_train = y_train

        self.s_prop, self.a_sort, self.a_cmean, self.a_ecdf = {}, {}, {}, {}
        self.a_mean = np.zeros(self.d)
        for i, c in enumerate(self.s_encoder.categories_[0]):
            # empirical distribution of s as a dict {int : float}
            self.s_prop[c] = np.mean(self.s_train[:, i])
            # sorted a (each attribute respectively) across s as a dict {int : (n_s, d)}
            self.a_sort[c] = np.sort(a_train[s_train.squeeze() == c], axis=0)
            # conditional mean of a across s as a dict {int : (d, )}
            self.a_cmean[c] = np.mean(self.a_sort[c], axis=0)
            # empirical CDF of a across s (as a dict {int : [function(float) -> float]})
            self.a_ecdf[c] = [ECDF(a_j) for a_j in self.a_sort[c].transpose()]
            # sample mean of a (d, )
            self.a_mean += self.a_cmean[c] * self.s_prop[c]

        # preprocess data
        self.preprocess_method = preprocess_method
        self.a_prime = self.process(s_train, a_train)

        if mode == 'predict':
            # training features with intercept term, shape=(n, d+c)
            dat_train = np.column_stack((self.s_train, a_train))
            # machine learning model of y
            self.ml = sm.Logit(y_train, dat_train).fit(disp=False)
            # fairness-through-unawareness model of y
            self.ftu = sm.Logit(y_train, sm.add_constant(a_train)).fit(disp=False)
            dat_prime = np.column_stack((self.s_train, self.a_prime))
            # machine learning model of y with processed a
            self.mlp = sm.Logit(y_train, dat_prime).fit(disp=False)
            # fairness-through-unawareness model of y with processed a
            self.ftup = sm.Logit(y_train, sm.add_constant(self.a_prime)).fit(disp=False)

    def cit(self, type=None, **kwargs):
        if type is None:
            if self.n < 100:
                type = 'cdc'
            else:
                type = 'parametric'
        if type == 'cdc':
            return self.cit_cdc(**kwargs)
        elif type == 'parametric':
            return self.cit_parametric(**kwargs)
        else:
            raise ValueError('Conditional Independent Test type {:s} not implemented'.format(type))

    def cit_cdc(self, b=99, numba=True):
        test = CDCTest(self.s_train[:, 1:], self.y_train, self.a_prime, num_bootstrap=b, numba=numba)
        test.conduct_conditional_independence_test()
        return test.p_value

    def cit_parametric(self, summary=False):
        try:
            parametric_model = self.mlp
        except AttributeError:
            dat_prime = np.column_stack((self.s_train, self.a_prime))
            parametric_model = sm.Logit(self.y_train, dat_prime).fit(disp=False)
        A = np.zeros((self.c - 1, self.c + self.d))
        for i in range(self.c - 1):
            A[i, i+1] = 1
        test = parametric_model.f_test(A)
        if summary: print(test)
        return test.pvalue.item()

    def assert_(self, a, s=None, s_is_onehot=True):
        """Assert inputs are of the right shape.

        If the inputs are acceptable, a will be transformed into np.ndarray of
        shape (*, d) and s will be transformed into one-hot np.ndarray of
        shape (*, c).
        """
        a = np.array(a, ndmin=2)
        assert a.ndim == 2 and a.shape[1] == self.d
        if s is None:
            return a
        else:
            s = np.array(s, ndmin=2)
            assert s.ndim == 2 and s.shape[0] == a.shape[0]
            assert s.shape[1] == 1 or s.shape[1] == self.c
            if s_is_onehot and s.shape[1] == 1:
                # transform s if it's not one-hot encoded
                s = self.s_encoder.transform(s).toarray()
            elif not s_is_onehot and s.shape[1] == self.c:
                s = self.s_encoder.inverse_transform(s)
            return a, s

    def process(self, s, a, method=None):
        """Wrapper for preprocessing data.

        Args:
            s (numpy.ndarray): categorical sensitive training attributes of
                shape (*, 1) or one-hot encoded attributes of shape (*, c).
            a (numpy.ndarray): non-sensitive training attributes, shape (*, d).
            method (str): 'o' for orthogonalization, 'm' for marginal
                distribution mapping.

        Returns:
            A numpy.ndarray of processed non-sensitive training attributes
            with shape (*, d).

        """
        a, s = self.assert_(a, s, s_is_onehot=False)
        if not method:
            method = self.preprocess_method
        if method == 'o':
            return self.process_orthogonal(s, a)
        elif method == 'm':
            return self.process_margin(s, a)
        else:
            raise ValueError('Preprocess method {:s} not implemented'.format(method))

    def process_orthogonal(self, s, a):
        """Preprocess data using orthogonalization.

        Args:
            s (numpy.ndarray): shape (*, 1).
            a (numpy.ndarray): shape (*, d).

        Returns:
            A numpy.ndarray of shape (*, d).

        """
        # conditional mean of a given s, shape=(*, d)
        a_s = np.array([self.a_cmean[s_i[0]] for s_i in s])
        a_prime = a - a_s + self.a_mean
        return a_prime

    def process_margin(self, s, a, s_prime=None):
        """Preprocess data using marginal distribution mapping.

        Args:
            s (numpy.ndarray): shape (*, 1).
            a (numpy.ndarray): shape (*, d).
            s_prime (int): If not None, a(s_prime)|s, a is returned; otherwise,
                the average of a(s_prime)|s, a over s_prime is returned.

        Returns:
            A numpy.ndarray of shape (*, d).

        """
        assert s_prime is None or isinstance(s_prime, int)
        a_prime = np.zeros_like(a)
        # number of samples for each s
        n_s = [self.a_sort[c].shape[0] for c in range(self.c)]
        for i in range(a.shape[0]):
            for j, ecdf in enumerate(self.a_ecdf[s[i, 0]]):
                p = ecdf(a[i, j]) * (1 - 1e-10)
                if s_prime is not None:
                    a_prime[i, j] = self.a_sort[s_prime][int(n_s[s_prime] * p), j]
                else:
                    for _s, prob_s in self.s_prop.items():
                        a_prime[i, j] += self.a_sort[_s][int(n_s[_s] * p), j] * prob_s
        return a_prime

    def f_ml(self, s_new, a_new):
        """Machine learning prediction.

        Args:
            s_new (numpy.ndarray): categorical sensitive training attributes
                of shape (*, 1) or one-hot encoded attributes of shape (*, c).
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        a_new, s_new = self.assert_(a_new, s_new)
        f = self.ml.predict(np.column_stack((s_new, a_new)))
        return f.squeeze()

    def f_eo(self, a_new):
        """Equal opportunity prediction (Wang et al., 2019)

        Args:
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        a_new = self.assert_(a_new)
        r = a_new.shape[0]
        f = np.zeros(r)
        for s, prob_s in self.s_prop.items():
            f += self.f_ml(np.broadcast_to(s, (r, 1)), a_new) * prob_s
        return f.squeeze()

    def f_aa(self, s_new, a_new):
        """Affirmative action prediction (Wang et al., 2019)

        Args:
            s_new (numpy.ndarray): categorical sensitive training attributes
                of shape (*, 1) or one-hot encoded attributes of shape (*, c).
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        a_new, s_new = self.assert_(a_new, s_new, s_is_onehot=False)
        f = np.zeros(a_new.shape[0])
        # conditional mean of a given s, shape=(*, d)
        a_s = np.array([self.a_cmean[s_i[0]] for s_i in s_new])
        # shape=(*, d)
        tmp = a_new - a_s
        for s, prob_s in self.s_prop.items():
            # shape=(*, d)
            a_new_prime = tmp + self.a_cmean[s]
            f += self.f_eo(a_new_prime) * prob_s
        return f.squeeze()

    def f_ftu(self, a_new):
        """Fairness-through-unawareness prediction

        Args:
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        a_new = self.assert_(a_new)
        f = self.ftu.predict(sm.add_constant(a_new, has_constant='add'))
        return f.squeeze()

    def f_mlp(self, s_new, a_new):
        """Machine learning prediction with preprocessed input

        Args:
            s_new (numpy.ndarray): categorical sensitive training attributes
                of shape (*, 1) or one-hot encoded attributes of shape (*, c).
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        a_new, s_new = self.assert_(a_new, s_new)
        f = self.mlp.predict(np.column_stack((s_new, a_new)))
        return f.squeeze()

    def f_ftup(self, a_new):
        """Fairness-through-unawareness prediction with preprocessed input.

        Args:
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        a_new = self.assert_(a_new)
        f = self.ftup.predict(sm.add_constant(a_new, has_constant='add'))
        return f.squeeze()

    def f_1(self, s_new, a_new, preprocess_method=None):
        """

        Args:
            s_new (numpy.ndarray): categorical sensitive training attributes
                of shape (*, 1) or one-hot encoded attributes of shape (*, c).
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).
            preprocess_method (str): 'o' for orthogonalization, 'm' for
                marginal distribution mapping.

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        a_new, s_new = self.assert_(a_new, s_new, s_is_onehot=False)
        if not preprocess_method:
            preprocess_method = self.preprocess_method
        r = a_new.shape[0]
        f = np.zeros(r)
        a_new_prime = self.process(s_new, a_new, method=preprocess_method)
        for s, prob_s in self.s_prop.items():
            f += self.f_mlp(np.broadcast_to(s, (r, 1)), a_new_prime) * prob_s
        return f.squeeze()

    def f_2(self, s_new, a_new, preprocess_method=None):
        """

        Args:
            s_new (numpy.ndarray): categorical sensitive training attributes
                of shape (*, 1) or one-hot encoded attributes of shape (*, c).
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).
            preprocess_method (str): 'o' for orthogonalization, 'm' for
                marginal distribution mapping.

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        a_new, s_new = self.assert_(a_new, s_new, s_is_onehot=False)
        if not preprocess_method:
            preprocess_method = self.preprocess_method
        a_new_prime = self.process(s_new, a_new, method=preprocess_method)
        f = self.f_ftup(a_new_prime)
        return f.squeeze()

    def eo_metric(self, a):
        """Evaluation with test data.

        Args:
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).

        """
        n = a.shape[0]
        y_ml, y_aa, y_1, y_2 = \
            np.zeros(self.c), np.zeros(self.c), np.zeros(self.c), np.zeros(self.c)
        for g in range(self.c):
            y_ml[g] = np.mean(self.f_ml(np.broadcast_to(g, (n, 1)), a))
            y_aa[g] = np.mean(self.f_aa(np.broadcast_to(g, (n, 1)), a))
            y_1[g] = np.mean(self.f_1(np.broadcast_to(g, (n, 1)), a))
            y_2[g] = np.mean(self.f_2(np.broadcast_to(g, (n, 1)), a))
        eo_ml = np.max(np.abs(y_ml.reshape(-1, 1) - y_ml))
        eo_aa = np.max(np.abs(y_aa.reshape(-1, 1) - y_aa))
        eo_1 = np.max(np.abs(y_1.reshape(-1, 1) - y_1))
        eo_2 = np.max(np.abs(y_2.reshape(-1, 1) - y_2))
        return np.asarray((eo_ml, 0., 0., eo_aa, eo_1, eo_2))

    def aa_metric(self, s, a):
        """Affirmative Action metric.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).

        """
        y_ml, y_ftu, y_eo, y_aa, y_1, y_2 = \
            np.zeros(self.c), np.zeros(self.c), np.zeros(self.c), np.zeros(self.c), np.zeros(self.c), np.zeros(self.c)
        for g in range(self.c):
            a_prime = self.process_margin(s, a, g)
            y_ml[g] = np.mean(self.f_ml(np.broadcast_to(g, s.shape), a_prime))
            y_ftu[g] = np.mean(self.f_ftu(a_prime))
            y_eo[g] = np.mean(self.f_eo(a_prime))
            y_aa[g] = np.mean(self.f_aa(np.broadcast_to(g, s.shape), a_prime))
            y_1[g] = np.mean(self.f_1(np.broadcast_to(g, s.shape), a_prime))
            y_2[g] = np.mean(self.f_2(np.broadcast_to(g, s.shape), a_prime))
        aa_ml = np.max(np.abs(y_ml.reshape(-1, 1) - y_ml))
        aa_ftu = np.max(np.abs(y_ftu.reshape(-1, 1) - y_ftu))
        aa_eo = np.max(np.abs(y_eo.reshape(-1, 1) - y_eo))
        aa_aa = np.max(np.abs(y_aa.reshape(-1, 1) - y_aa))
        aa_1 = np.max(np.abs(y_1.reshape(-1, 1) - y_1))
        aa_2 = np.max(np.abs(y_2.reshape(-1, 1) - y_2))
        return np.asarray((aa_ml, aa_ftu, aa_eo, aa_aa, aa_1, aa_2))

    def accuracy(self, s, a, y):
        """Accuracy in test data.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y (numpy.ndarray): binary decisions with size *.

        """
        y = np.array(y).squeeze()
        acc_ml = np.mean(((self.f_ml(s, a) > 0.5) == y).astype(np.int))
        acc_ftu = np.mean(((self.f_ftu(a) > 0.5) == y).astype(np.int))
        acc_eo = np.mean(((self.f_eo(a) > 0.5) == y).astype(np.int))
        acc_aa = np.mean(((self.f_aa(s, a) > 0.5) == y).astype(np.int))
        acc_1 = np.mean(((self.f_1(s, a) > 0.5) == y).astype(np.int))
        acc_2 = np.mean(((self.f_2(s, a) > 0.5) == y).astype(np.int))
        return np.hstack((acc_ml, acc_ftu, acc_eo, acc_aa, acc_1, acc_2))

    def evaluate(self, s_test=None, a_test=None, y_test=None, metrics=None):
        """Evaluation with test data.

        Args:
            s_test (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a_test (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y_test (numpy.ndarray): binary decisions, must have shape (*, 1).
            metrics (list): list of strings

        Returns:
            rtn (tuple): metric values in the order of metrics.

        """
        if metrics is None:
            metrics = ['eo', 'aa', 'acc']
        rtn = ()
        for metric in metrics:
            if metric == 'eo':
                rtn += (self.eo_metric(a_test),)
            elif metric == 'aa':
                rtn += (self.aa_metric(s_test, a_test),)
            elif metric == 'acc':
                rtn += (self.accuracy(s_test, a_test, y_test),)
            else:
                raise ValueError('Metric {:s} not implemented'.format(metric))
        return rtn
