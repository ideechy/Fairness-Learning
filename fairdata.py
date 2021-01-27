import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import entropy
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from cdctest import CDCTest


class FairData(object):
    def __init__(self, s_train, a_train, y_train, preprocess_method='o', mode='predict', a_iscategory=None):
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
        s_train = np.asarray(s_train)
        a_train = np.asarray(a_train)
        y_train = np.asarray(y_train)
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
        self.a_iscategory = self.infer_atype() if a_iscategory is None else a_iscategory
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
            # machine learning model of y, i.e. the propensity model pi(s, a; gamma)
            self.ml = sm.Logit(y_train, dat_train).fit(disp=False)
            # fairness-through-unawareness model of y
            self.ftu = sm.Logit(y_train, sm.add_constant(a_train)).fit(disp=False)
            dat_prime = np.column_stack((self.s_train, self.a_prime))
            # machine learning model of y with processed a
            self.mlp = sm.Logit(y_train, dat_prime).fit(disp=False)
            # fairness-through-unawareness model of y with processed a
            self.ftup = sm.Logit(y_train, sm.add_constant(self.a_prime)).fit(disp=False)
            # extract residuals of a regressing on s as features
            a_s = np.array([self.a_cmean[s_i] for s_i in s_train.squeeze()])
            a_res = a_train - a_s
            # machine learning model of y with residuals
            self.mlr = sm.Logit(y_train, sm.add_constant(a_res)).fit(disp=False)

        # runtime variables
        self.bound = dict()

    def infer_atype(self, c=1, m=10):
        """Infer if attributes are categorical.

        A non-sensitive attribute is considered as a categorical attribute if
            #distinct_value < max( sqrt( #sample ) * c, m )

        Args:
            c (float)
            m (float)

        Returns:
            A list of boolean

        """
        return [len(set(self.a_train[:, j])) < max(np.sqrt(self.n) * c, m) for j in range(self.d)]

    def cit(self, type=None, **kwargs):
        if type is None:
            type = 'cdc' if self.n < 100 else 'parametric'
        if type == 'cdc':
            return self.cit_cdc(**kwargs)
        if type == 'parametric':
            return self.cit_parametric(**kwargs)
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
        shape (*, c) if `s_is_onehot=True`, or categorical np.ndarray of shape
        (*, 1) if `s_is_onehot=False`.
        
        """
        a = np.array(a, ndmin=2)
        assert a.ndim == 2 and a.shape[1] == self.d
        if s is None:
            return a
        s = np.array(s, ndmin=2)
        assert s.ndim == 2 and s.shape[0] == a.shape[0]
        assert s.shape[1] == 1 or s.shape[1] == self.c
        if s_is_onehot and s.shape[1] == 1:
            # transform s if it's not one-hot encoded
            s = self.s_encoder.transform(s).toarray()
        elif not s_is_onehot and s.shape[1] == self.c:
            s = self.s_encoder.inverse_transform(s)
        return a, s

    def process(self, s, a, s_prime=None, method=None):
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
            return self.process_orthogonal(s, a, s_prime)
        if method == 'm':
            return self.process_margin(s, a, s_prime)
        if method == 'mr' or method == 'r':
            return self.process_margin_random(s, a, s_prime)
        raise ValueError('Preprocess method {:s} not implemented'.format(method))

    def process_orthogonal(self, s, a, s_prime=None):
        """Preprocess data using orthogonalization.

        Args:
            s (numpy.ndarray): shape (*, 1).
            a (numpy.ndarray): shape (*, d).
            s_prime (int): If not None, a(s_prime)|s, a is returned; otherwise,
                the average of a(s_prime)|s, a over s_prime is returned.

        Returns:
            A numpy.ndarray of shape (*, d).

        """
        assert s_prime is None or isinstance(s_prime, int)
        # conditional mean of a given s, shape=(*, d)
        a_s = np.array([self.a_cmean[s_i[0]] for s_i in s])
        if s_prime is not None:
            return a - a_s + self.a_cmean[s_prime]
        return a - a_s + self.a_mean

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
        a_prime = np.zeros_like(a, dtype='float')
        # number of samples for each s
        n_s = [self.a_sort[c].shape[0] for c in range(self.c)]
        for i in range(a.shape[0]):
            for j, ecdf in enumerate(self.a_ecdf[s[i, 0]]):
                p = ecdf(a[i, j]) 
                if s_prime is not None:
                    a_prime[i, j] = self.a_sort[s_prime][int((n_s[s_prime] - 1) * p), j]
                else:
                    for _s, prob_s in self.s_prop.items():
                        a_prime[i, j] += self.a_sort[_s][int((n_s[_s] - 1) * p), j] * prob_s
        return a_prime

    def sample_margin(self, s, a, s_prime, p_range=0.05, b=50):
        n, d = a.shape
        a_prime = np.zeros((n, b, d), dtype='float')
        # number of samples for each s
        n_s = [self.a_sort[c].shape[0] for c in range(self.c)]
        for i in range(n):
            for j, ecdf in enumerate(self.a_ecdf[s[i, 0]]):
                p = ecdf(a[i, j])
                idx_lo = int((n_s[s_prime]-1) * max(0, p-p_range))
                idx_hi = int((n_s[s_prime]-1) * min(1, p+p_range)) + 1
                idx = np.random.choice(np.arange(idx_lo, idx_hi), b)
                a_prime[i, :, j] = self.a_sort[s_prime][idx, j]
        return a_prime

    def process_margin_random(self, s, a, s_prime=None):
        """Preprocess data using marginal distribution mapping.

        When a contains categorical attributes, the processed value of the 
        categorical attribute is the counterfactual value of it had the unit 
        been in a randomly selected sensitive group.

        Args:
            s (numpy.ndarray): shape (*, 1).
            a (numpy.ndarray): shape (*, d).
            s_prime (int): If not None, a(s_prime)|s, a is returned; otherwise,
                the average of a(s_prime)|s, a over s_prime is returned.

        Returns:
            A numpy.ndarray of shape (*, d).

        """
        assert s_prime is None or isinstance(s_prime, int)
        a_prime = np.zeros_like(a, dtype='float')
        # number of samples for each s
        n_s = [self.a_sort[c].shape[0] for c in range(self.c)]
        for i in range(a.shape[0]):
            for j, ecdf in enumerate(self.a_ecdf[s[i, 0]]):
                p = ecdf(a[i, j]) 
                if s_prime is not None:
                    a_prime[i, j] = self.a_sort[s_prime][int((n_s[s_prime] - 1) * p), j]
                elif self.a_iscategory[j]:
                    _s = np.random.choice(list(self.s_prop.keys()), p=list(self.s_prop.values()))
                    a_prime[i, j] = self.a_sort[_s][int((n_s[_s] - 1) * p), j]
                else:
                    for _s, prob_s in self.s_prop.items():
                        a_prime[i, j] += self.a_sort[_s][int((n_s[_s] - 1) * p), j] * prob_s
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

    def f_fl(self, s_new, a_new):
        """FairLearning prediction (Kusner et al., 2017)
        Implementation taken from (Wang et al., 2019)

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
        res = a_new - a_s
        f = self.mlr.predict(sm.add_constant(res, has_constant='add'))
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

    def f_wrapper(self, method, a_new, s_new=None, **kwargs):
        method = method.upper()
        if method == 'ML':
            assert s_new is not None
            return self.f_ml(s_new, a_new)
        if method == 'FTU':
            return self.f_ftu(a_new)
        if method == 'FL':
            return self.f_fl(s_new, a_new)
        if method == 'EO':
            return self.f_eo(a_new)
        if method == 'AA':
            assert s_new is not None
            return self.f_aa(s_new, a_new)
        if method == 'FLAP-1':
            assert s_new is not None
            preprocess_method = kwargs.get('preprocess_method', None)
            return self.f_1(s_new, a_new, preprocess_method=preprocess_method)
        if method == 'FLAP-2':
            assert s_new is not None
            preprocess_method = kwargs.get('preprocess_method', None)
            return self.f_2(s_new, a_new, preprocess_method=preprocess_method)
        raise ValueError('Method {:s} not implemented'.format(method))

    def eo_metric(self, a, methods, **kwargs):
        """Equal Oppurtunity metric.

        Args:
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            methods: names of decision making methods to evaluate.

        """
        metrics = np.zeros(len(methods))
        for i, method in enumerate(methods):
            if method in ['FTU', 'EO']:
                continue
            y = np.zeros(self.c)
            for g in range(self.c):
                y[g] = np.mean(self.f_wrapper(
                    method, a, np.broadcast_to(g, (a.shape[0], 1)), **kwargs
                ))
            metrics[i] = np.max(np.abs(y.reshape(-1, 1) - y))
        return metrics

    def kl_metric(self, a, methods, **kwargs):
        """KL divergence metric.

        Args:
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            methods: names of decision making methods to evaluate.

        """
        metrics = np.zeros(len(methods))
        for i, method in enumerate(methods):
            y = np.zeros((self.c, a.shape[0]))
            for g in range(self.c):
                y[g] = self.f_wrapper(
                    method, a, np.broadcast_to(g, (a.shape[0], 1)), **kwargs
                )
            for g1 in range(self.c - 1):
                for g2 in range(g1 + 1, self.c):
                    kl = np.mean(entropy([y[g1], 1-y[g1]], [y[g2], 1-y[g2]]))
                    metrics[i] = max(metrics[i], kl)
        return metrics

    def aa_metric(self, s, a, methods, **kwargs):
        """Affirmative Action metric.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            methods: names of decision making methods to evaluate.

        """
        metrics = np.empty(len(methods))
        a_prime = dict()
        for g in range(self.c):
            a_prime[g] = self.process(s, a, g, method='o')
        for i, method in enumerate(methods):
            y = np.zeros(self.c)
            for g in range(self.c):
                y[g] = np.mean(self.f_wrapper(
                    method, a_prime[g], np.broadcast_to(g, s.shape), **kwargs
                ))
            metrics[i] = np.max(np.abs(y.reshape(-1, 1) - y))
        return metrics

    def cf_metric(self, s, a, methods, **kwargs):
        """Counterfactual Fairness metric.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            methods: names of decision making methods to evaluate.

        """
        metrics = np.empty(len(methods))
        a_prime = dict()
        for g in range(self.c):
            a_prime[g] = self.process(s, a, g, method='m')
        for i, method in enumerate(methods):
            y = np.zeros(self.c)
            for g in range(self.c):
                y[g] = np.mean(self.f_wrapper(
                    method, a_prime[g], np.broadcast_to(g, s.shape), **kwargs
                ))
            metrics[i] = np.max(np.abs(y.reshape(-1, 1) - y))
        return metrics

    def cf_bound(self, s, a, methods, p_range=0.05, b=50, **kwargs):
        """Absolute bound for counterfactual fairness
        
        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            methods: names of decision making methods to evaluate.

        """
        metrics = np.empty(len(methods))
        n, d = a.shape
        a_prime = dict()
        for g in range(self.c):
            a_prime[g] = self.sample_margin(s, a, g, p_range, b).reshape(-1, d)
        for i, method in enumerate(methods):
            pred = self.f_wrapper(method, a, s, **kwargs).repeat(b)
            y = np.zeros(self.c)
            for g in range(self.c):
                pred_cf = self.f_wrapper(
                    method, a_prime[g], np.broadcast_to(g, (n*b, 1)), **kwargs)
                y[g] = np.max(np.abs(pred_cf - pred))
            metrics[i] = y.max()
        return metrics

    def cf_bound_mean(self, s, a, methods, p_range=0.05, b=50, **kwargs):
        """Absolute bound for counterfactual fairness
        
        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            methods: names of decision making methods to evaluate.

        """
        metrics = np.empty(len(methods))
        n, d = a.shape
        a_prime = dict()
        for g in range(self.c):
            a_prime[g] = self.sample_margin(s, a, g, p_range, b).reshape(-1, d)
        for i, method in enumerate(methods):
            pred = self.f_wrapper(method, a, s, **kwargs)
            y = np.zeros(self.c)
            for g in range(self.c):
                pred_cf = self.f_wrapper(
                    method, a_prime[g], np.broadcast_to(g, (n*b, 1)), **kwargs)
                y[g] = np.max(np.abs(pred_cf.reshape(n, b).mean(axis=1) - pred))
            metrics[i] = y.max()
        return metrics

    def cf_true(self, a, methods, **kwargs):
        """True counterfactual fairness difference for simulated data
        
        Args:
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (c, *, d).
            methods: names of decision making methods to evaluate.

        """
        n = a.shape[1]
        metrics = np.zeros(len(methods))
        for i, method in enumerate(methods):
            y = np.zeros((self.c, n))
            for g in range(self.c):
                y[g] = self.f_wrapper(
                    method, a[g], np.broadcast_to(g, (n, 1)), **kwargs
                )
            for g1 in range(self.c):
                for g2 in range(g1):
                    metrics[i] = max(metrics[i], np.max(np.abs(y[g1] - y[g2])))
        return metrics

    def accuracy(self, s, a, y, methods, **kwargs):
        """Accuracy in test data (deprecated due to randomness).

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y (numpy.ndarray): binary decisions with size *.
            methods: names of decision making methods to evaluate.

        """
        y = np.array(y).squeeze()
        metrics = np.empty(len(methods))
        for i, method in enumerate(methods):
            p = self.f_wrapper(method, a, s, **kwargs)
            metrics[i] = accuracy_score(y, p)
        return metrics

    def mae(self, s, a, y, methods, **kwargs):
        """Mean Absolute Error in test data.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y (numpy.ndarray): binary decisions with size *.
            methods: names of decision making methods to evaluate.

        """
        y = np.array(y).squeeze()
        metrics = np.empty(len(methods))
        for i, method in enumerate(methods):
            p = self.f_wrapper(method, a, s, **kwargs)
            metrics[i] = np.mean(np.abs(p - y))
        return metrics

    def roc_auc(self, s, a, y, methods, **kwargs):
        """Area under the ROC curve in test data.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y (numpy.ndarray): binary decisions with size *.
            methods: names of decision making methods to evaluate.

        """
        y = np.array(y).squeeze()
        metrics = np.empty(len(methods))
        for i, method in enumerate(methods):
            p = self.f_wrapper(method, a, s, **kwargs)
            metrics[i] = roc_auc_score(y, p)
        return metrics

    def average_precision(self, s, a, y, methods, **kwargs):
        """Area under the PR curve in test data.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y (numpy.ndarray): binary decisions with size *.
            methods: names of decision making methods to evaluate.

        """
        y = np.array(y).squeeze()
        metrics = np.empty(len(methods))
        for i, method in enumerate(methods):
            p = self.f_wrapper(method, a, s, **kwargs)
            metrics[i] = average_precision_score(y, p)
        return metrics

    def evaluate(
        self, a_test, s_test=None, y_test=None, 
        metrics=None, methods=None, **kwargs):
        """Evaluation with test data.

        Args:
            a_test (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            s_test (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            y_test (numpy.ndarray): binary decisions, must have shape (*, 1).
            metrics: list of evaluation metrics.
            methods: list of decision making methods to evaluate.

        Returns:
            rtn (tuple): metric values in the order of metrics.

        """
        a_test = np.asarray(a_test)
        if s_test is not None: 
            s_test = np.asarray(s_test)
        if y_test is not None: 
            y_test = np.asarray(y_test)
        if metrics is None:
            metrics = ['eo', 'cf', 'mae']
        if methods is None:
            methods = ['ML', 'FTU', 'FL', 'AA', 'FLAP-1', 'FLAP-2']
        rtn = ()
        func_dict = {
            'eo': 'eo_metric',
            'aa': 'aa_metric',
            'cf': 'cf_metric',
            'cfb': 'cf_bound',
            'cfbm': 'cf_bound_mean',
            'kl': 'kl_metric',
            'acc': 'accuracy', 
            'mae': 'mae', 
            'roc': 'roc_auc', 
            'ap': 'average_precision',
        }
        for metric in metrics:
            func = getattr(self, func_dict[metric])
            if metric == 'eo' or metric == 'kl':
                rtn += (func(a=a_test, methods=methods, **kwargs),)
            elif metric in ['cf', 'aa', 'cfb', 'cfbm']:
                assert s_test is not None
                rtn += (func(s=s_test, a=a_test, methods=methods, **kwargs),)
            elif metric in ['acc', 'mae', 'roc', 'ap']:
                assert s_test is not None and y_test is not None
                rtn += (func(s=s_test, a=a_test, y=y_test, methods=methods, **kwargs),)
            else:
                raise ValueError('Metric {:s} not implemented'.format(metric))
        return rtn
