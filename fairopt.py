import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from scipy import optimize

from fairdata import FairData

class FairOptimization(FairData):
    def __init__(self, s_train, a_train, y_train, r_train, preprocess_method='m', a_iscategory=None):
        """Initialization of data.

        Args:
            s_train (numpy.ndarray): categorical sensitive training attributes,
                must have shape (n, 1).
            a_train (numpy.ndarray): non-sensitive training attributes, must
                have shape (n, d).
            y_train (numpy.ndarray): binary decisions, must have shape (n, 1).
            r_train (numpy.ndarray): observed rewards, must have shape (n, 1).
            preprocess_method (str): 'o' for orthogonalization, 'm' for
                marginal distribution mapping.
        """
        super().__init__(
            s_train, a_train, y_train, preprocess_method, 
            mode='predict', a_iscategory=a_iscategory)
        assert r_train.ndim == 2 and r_train.shape == (self.n, 1)
        self.r_train = r_train
        # training features with intercept term, shape=(n, d+c)
        dat_train = np.column_stack((self.s_train, self.a_train))
        # machine learning model of r
        self.rml = sm.Logit(
            (r_train[y_train.squeeze() == 1] + 1) / 2, 
            dat_train[y_train.squeeze() == 1]
        ).fit(disp=False)

    def f_deterministic(self, eta, s_new=None, a_new=None):
        """Prediction with preprocessed input and deterministic function 
        indexed by eta

        Args:
            eta (numpy.ndarray): parameters of shape (d + 1, ).
            s_new (numpy.ndarray): categorical sensitive training attributes
                of shape (*, 1) or one-hot encoded attributes of shape (*, c).
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).

        Returns:
            A numpy.ndarray of predicted decisions (0 or 1) with shape (*, ).

        """
        if a_new is None:
            a_prime = self.a_prime
        else:
            assert s_new is not None
            a_prime = self.process(s_new, a_new)
        eta = np.asarray(eta).reshape(-1, )
        assert eta.shape[0] == self.d + 1
        return (np.dot(sm.add_constant(a_prime), eta) > 0).astype(np.int)


    def f_expit(self, eta, s_new=None, a_new=None):
        """Prediction with preprocessed input and expit function indexed by eta

        Args:
            eta (numpy.ndarray): parameters of shape (d + 1, ).
            s_new (numpy.ndarray): categorical sensitive training attributes
                of shape (*, 1) or one-hot encoded attributes of shape (*, c).
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        if a_new is None:
            a_prime = self.a_prime
        else:
            assert s_new is not None
            a_prime = self.process(s_new, a_new)
        eta = np.asarray(eta).reshape(-1, )
        assert eta.shape[0] == self.d + 1
        return expit(np.dot(sm.add_constant(a_prime), eta))

    def f_rml(self, s_new, a_new):
        """Mahcine learning prediction to optimize return.

        Args:
            s_new (numpy.ndarray): categorical sensitive training attributes
                of shape (*, 1) or one-hot encoded attributes of shape (*, c).
            a_new (numpy.ndarray): non-sensitive training attributes, shape
                (*, d).

        Returns:
            A numpy.ndarray of predicted decisions with shape (*, ).

        """
        a_new, s_new = self.assert_(a_new, s_new)
        f = self.rml.predict(np.column_stack((s_new, a_new)))
        return f.squeeze()

    def f_wrapper(self, method, a_new, s_new=None, **kwargs):
        method = method.upper()
        if method == 'FLAP-ETA' or method[:10] == 'FLAP-ETA-D':
            assert s_new is not None and 'eta' in kwargs
            return self.f_deterministic(kwargs['eta'], s_new, a_new)
        elif method[:10] == 'FLAP-ETA-S':
            assert s_new is not None and 'eta' in kwargs
            return self.f_expit(kwargs['eta'], s_new, a_new)
        elif method == 'RML':
            assert s_new is not None
            return self.f_rml(s_new, a_new)
        else:
            return super().f_wrapper(method, a_new, s_new, **kwargs)

    def ipwe(self, eta, deterministic=True):
        """Inverse probability weighted estimation of the expected reward

        Args:
            eta (numpy.ndarray): parameters of shape (d + 1, ).

        Returns:
            A float number for the estimated expected reward.

        """
        if deterministic:
            y_hat = self.f_deterministic(eta)
            p = y_hat
        else:
            p = self.f_expit(eta)
            y_hat = np.random.binomial(1, p)
        c = (y_hat == self.y_train).astype(np.int)
        pi = self.ml.predict().squeeze()
        pi_c = pi * p + (1 - pi) * (1 - p)
        return np.mean(c * self.r_train / pi_c)

    def aipwe(self, eta, deterministic=True):
        """Augmented ipwe of the expected reward

        Args:
            eta (numpy.ndarray): parameters of shape (d + 1, ).

        Returns:
            A float number for the estimated expected reward.

        """
        if deterministic:
            y_hat = self.f_deterministic(eta)
            p = y_hat
        else:
            p = self.f_expit(eta)
            y_hat = np.random.binomial(1, p)
        c = (y_hat == self.y_train).astype(np.int)
        pi = self.ml.predict().squeeze()
        pi_c = pi * p + (1 - pi) * (1 - p)
        dat_train = np.column_stack((self.s_train, self.a_train))
        r_hat = (self.rml.predict(dat_train).squeeze() - 0.5) * 2
        return np.mean((c * self.r_train + (c - pi_c) * y_hat * r_hat) / pi_c)

    def optimize(
        self, estimation_fun, estimation_args=None, method=None, **kwargs):
        """Find the optimal parameter which maximizes the estimated expect reward

        Args:
            estimation_fun (callable): estimation function of the expected 
                reward which takes the function parameter eta as the only input.
            estimation_args (dict): keyword arguments passed to the estimation
                function.
            method (str): type of solver passed to `scipy.optimize.minimize`.

        Returns:
            The optimization result as `OptimizeResult` object. Important 
            attributes are: `x` the solution array, `success` a Boolean flag 
            indicating if the optimizer exited successfully and `message` which 
            describes the cause of the termination. 

        """
        eta0 = self.ftup.params
        bounds = [(-1, 1)] * (self.d + 1)
        if estimation_args is None:
            estimation_args = dict()
        fun = lambda x: -estimation_fun(x, **estimation_args)
        if method == 'shgo':
            eta_opt = optimize.shgo(fun, bounds, **kwargs)
        elif method == 'dual_annealing':
            eta_opt = optimize.dual_annealing(fun, bounds, **kwargs)
        elif method == 'differential_evolution':
            eta_opt = optimize.differential_evolution(fun, bounds, **kwargs)
        elif method == 'basinhopping':
            eta_opt = optimize.basinhopping(fun, bounds, **kwargs)
        else:
            eta_opt = optimize.minimize(fun, eta0, method=method, **kwargs)
        return eta_opt

    def reward_simulation(self, s, a, r_star, methods, **kwargs):
        """Average reward in the simulated test data.

        The potential reward should be fully observed in the simulated data.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            r_star (numpy.ndarray): potential rewards with size *.
            methods: names of decision making methods to evaluate.
        
        """
        r_star = np.array(r_star).squeeze()
        metrics = np.empty(len(methods))
        for i, method in enumerate(methods):
            p = self.f_wrapper(method, a, s, **kwargs)
            metrics[i] = np.mean(p * r_star)
        return metrics

    def reward_estimate(self, s, a, y, p, r, repeat=1):
        """AIPWE of reward on test data.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y (numpy.ndarray): observed binary decisions with size *.
            p (numpy.ndarray): probabilities of choosing Y hat = 1, size *.
            r (numpy.ndarray): observed rewards with size *.
            repeat (int): number of replications to simulate Y hat.
        
        """
        a, s = self.assert_(a, s)
        dat_test = np.column_stack((s, a))
        pi = self.ml.predict(dat_test).squeeze()
        pi_c = pi * p + (1 - pi) * (1 - p)
        r_hat = (self.rml.predict(dat_test).squeeze() - 0.5) * 2
        aipwe = 0
        for _ in range(repeat):
            y_hat = np.random.binomial(1, p)
            c = (y_hat == y).astype(np.int)
            aipwe += np.mean((c * r + (c - pi_c) * y_hat * r_hat) / pi_c)
        return aipwe / repeat

    def reward(self, s, a, y, r, methods, repeat=50, **kwargs):
        """Estimated average reward without the knowledge of R*.

        The potential reward is partially observed as R = R^*Y. If the chosen 
        decision Y hat = 1 while Y = 0, the reward is not observed. The average 
        reward under the eta decision rule is estimated using the AIPWE.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y (numpy.ndarray): observed binary decisions with size *.
            r (numpy.ndarray): observed rewards with size *.
            methods: names of decision making methods to evaluate.
            repeat (int): number of replications to calculate the AIPWE.
        
        """
        y = np.array(y).squeeze()
        r = np.array(r).squeeze()
        metrics = np.empty(len(methods))
        for i, method in enumerate(methods):
            p = self.f_wrapper(method, a, s, **kwargs)
            metrics[i] = self.reward_estimate(s, a, y, p, r, repeat)
        return metrics

    def evaluate(
        self, a_test, s_test=None, y_test=None, 
        metrics=None, methods=None, **kwargs):
        if metrics is None:
            metrics = ['cf', 'mae', 'er']
        if methods is None:
            methods = ['ML', 'RML', 'FTU', 'AA', 'FLAP-1', 'FLAP-2', 'FLAP-ETA']
        rtn = ()
        for metric in metrics:
            if metric == 'er':
                assert s_test is not None and y_test is not None 
                assert 'r_test' in kwargs
                r_test = kwargs.pop('r_test')
                rtn += (self.reward(
                    s_test, a_test, y_test, r_test, methods, **kwargs
                ),)
            elif metric[:3] == 'ers':
                assert s_test is not None
                assert 'r_star_test' in kwargs
                r_star_test = kwargs.pop('r_star_test')
                rtn += (self.reward_simulation(
                    s_test, a_test, r_star_test, methods, **kwargs
                ),)
            else:
                rtn += super().evaluate(
                    a_test, s_test, y_test, [metric], methods, **kwargs
                )
        return rtn
