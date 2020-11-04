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

    def f_expit(self, eta, s_new=None, a_new=None):
        """Prediction with preprocessed input and expit function indexed by eta

        Args:
            eta (numpy.ndarray): parameters of shape (d, ).
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

    def ipwe(self, eta):
        """Inverse probability weighted estimation of the expected reward

        Args:
            eta (numpy.ndarray): parameters of shape (d, ).

        Returns:
            A float number for the estimated expected reward.

        """
        p = self.f_expit(eta)
        y_hat = np.random.binomial(1, p)
        c = (y_hat == self.y_train).astype(np.int)
        pi = self.ml.predict().squeeze()
        pi_c = pi * p + (1 - pi) * (1 - p)
        return np.mean(c * self.r_train / pi_c)

    def aipwe(self, eta):
        """Augmented ipwe of the expected reward

        Args:
            eta (numpy.ndarray): parameters of shape (d, ).

        Returns:
            A float number for the estimated expected reward.

        """
        p = self.f_expit(eta)
        y_hat = np.random.binomial(1, p)
        # y_hat = (p > 0.5).astype(np.int)
        c = (y_hat == self.y_train).astype(np.int)
        pi = self.ml.predict().squeeze()
        pi_c = pi * p + (1 - pi) * (1 - p)
        # pi_c = pi * y_hat + (1 - pi) * (1 - y_hat)
        dat_train = np.column_stack((self.s_train, self.a_train))
        r_hat = (self.rml.predict(dat_train).squeeze() - 0.5) * 2
        return np.mean((c * self.r_train + (c - pi_c) * y_hat * r_hat) / pi_c)



    def optimize(self, estimation_fun, method=None, eta0=None, bounds=None, **kwargs):
        """Find the optimal parameter which maximizes the estimated expect reward

        Args:
            estimation_fun (callable): estimation function of the expected 
                reward which takes the function parameter eta as the only input.
            method (str): type of solver passed to `scipy.optimize.minimize`.
            eta0 (numpy.ndarray): initial guess of the parameter with shape 
                (d, ). If not given, set to be the coefficients of the logistic
                regression of the training decisions on the preprocessed data.
            bounds (sequence): search range of the parameters. `(min, max)` 
                pairs for each element in eta, defining the lower and upper 
                bounds for the optimizing argument of `estimation_fun`.

        Returns:
            The optimization result as `OptimizeResult` object. Important 
            attributes are: `x` the solution array, `success` a Boolean flag 
            indicating if the optimizer exited successfully and `message` which 
            describes the cause of the termination. 

        """
        if eta0 is None:
            eta0 = self.ftup.params
        else:
            eta0 = np.asarray(eta0).reshape(-1, )
            assert eta0.shape[0] == self.d + 1
        if bounds is None:
            bounds = [(e / 10, e * 10) if e > 0 else (e * 10, e / 10) for e in eta0]
        else:
            assert len(bounds) == self.d + 1
        fun = lambda x: -estimation_fun(x)
        if method == 'shgo':
            eta_opt = optimize.shgo(fun, bounds)
        elif method == 'dual_annealing':
            eta_opt = optimize.dual_annealing(fun, bounds)
        elif method == 'differential_evolution':
            eta_opt = optimize.differential_evolution(fun, bounds)
        elif method == 'basinhopping':
            eta_opt = optimize.basinhopping(fun, bounds)
        else:
            eta_opt = optimize.minimize(fun, eta0, method=method, **kwargs)
        return eta_opt

    def cf_metric(self, s, a, eta):
        """Counterfactual Fairness metric.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            eta (numpy.ndarray): index parameter of the decision class.

        """
        y_ml, y_rml, y_ftu, y_aa, y_1, y_2, y_eta = \
            (np.zeros(self.c) for _ in range(7))
        for g in range(self.c):
            a_prime = self.process_margin(s, a, g)
            y_ml[g] = np.mean(self.f_ml(np.broadcast_to(g, s.shape), a_prime))
            y_rml[g] = np.mean(self.f_rml(np.broadcast_to(g, s.shape), a_prime))
            y_ftu[g] = np.mean(self.f_ftu(a_prime))
            y_aa[g] = np.mean(self.f_aa(np.broadcast_to(g, s.shape), a_prime))
            y_1[g] = np.mean(self.f_1(np.broadcast_to(g, s.shape), a_prime))
            y_2[g] = np.mean(self.f_2(np.broadcast_to(g, s.shape), a_prime))
            y_eta[g] = np.mean(self.f_expit(eta, np.broadcast_to(g, s.shape), a_prime))
        cf_ml = np.max(np.abs(y_ml.reshape(-1, 1) - y_ml))
        cf_rml = np.max(np.abs(y_rml.reshape(-1, 1) - y_rml))
        cf_ftu = np.max(np.abs(y_ftu.reshape(-1, 1) - y_ftu))
        cf_aa = np.max(np.abs(y_aa.reshape(-1, 1) - y_aa))
        cf_1 = np.max(np.abs(y_1.reshape(-1, 1) - y_1))
        cf_2 = np.max(np.abs(y_2.reshape(-1, 1) - y_2))
        cf_eta = np.max(np.abs(y_eta.reshape(-1, 1) - y_eta))
        return np.asarray((cf_ml, cf_rml, cf_ftu, cf_aa, cf_1, cf_2, cf_eta))

    def mae(self, s, a, y, eta):
        """Mean Absolute Error in test data.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y (numpy.ndarray): observed binary decisions with size *.
            eta (numpy.ndarray): index parameter of the decision class.

        """
        y = np.array(y).squeeze()
        mae_ml = np.mean(np.abs(self.f_ml(s, a) - y))
        mae_rml = np.mean(np.abs(self.f_rml(s, a) - y))
        mae_ftu = np.mean(np.abs(self.f_ftu(a) - y))
        mae_aa = np.mean(np.abs(self.f_aa(s, a) - y))
        mae_1 = np.mean(np.abs(self.f_1(s, a) - y))
        mae_2 = np.mean(np.abs(self.f_2(s, a) - y))
        mae_eta = np.mean(np.abs(self.f_expit(eta, s, a) - y))
        return np.hstack((mae_ml, mae_rml, mae_ftu, mae_aa, mae_1, mae_2, mae_eta))

    def reward_simulation(self, s, a, r_star, eta):
        """Average reward in the simulated test data.

        The potential reward should be fully observed in the simulated data.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            r_star (numpy.ndarray): potential rewards with size *.
            eta (numpy.ndarray): index parameter of the decision class.
        
        """
        r_star = np.array(r_star).squeeze()
        r_ml = np.mean(self.f_ml(s, a) * r_star) 
        r_rml = np.mean(self.f_rml(s, a) * r_star) 
        r_ftu = np.mean(self.f_ftu(a) * r_star) 
        r_aa = np.mean(self.f_aa(s, a) * r_star) 
        r_1 = np.mean(self.f_1(s, a) * r_star) 
        r_2 = np.mean(self.f_2(s, a) * r_star) 
        r_eta = np.mean(self.f_expit(eta, s, a) * r_star) 
        return np.hstack((r_ml, r_rml, r_ftu, r_aa, r_1, r_2, r_eta))

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

    def reward(self, s, a, y, r, eta, repeat=50):
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
            eta (numpy.ndarray): index parameter of the decision class.
            repeat (int): number of replications to calculate the AIPWE.
        
        """
        y = np.array(y).squeeze()
        r = np.array(r).squeeze()
        r_ml = self.reward_estimate(s, a, y, self.f_ml(s, a), r, repeat)
        r_rml = self.reward_estimate(s, a, y, self.f_rml(s, a), r, repeat)
        r_ftu = self.reward_estimate(s, a, y, self.f_ftu(a), r, repeat)
        r_aa = self.reward_estimate(s, a, y, self.f_aa(s, a), r, repeat)
        r_1 = self.reward_estimate(s, a, y, self.f_1(s, a), r, repeat)
        r_2 = self.reward_estimate(s, a, y, self.f_2(s, a), r, repeat)
        r_eta = self.reward_estimate(s, a, y, self.f_expit(eta, s, a), r, repeat)
        return np.hstack((r_ml, r_rml, r_ftu, r_aa, r_1, r_2, r_eta))

    def evaluate(self, eta, s_test, a_test, y_test=None, r_test=None, r_star_test=None, metrics=None):
        if metrics is None:
            metrics = ['cf', 'mae', 'er']
        rtn = ()
        for metric in metrics:
            if metric == 'cf':
                rtn += (self.cf_metric(s_test, a_test, eta),)
            elif metric == 'mae':
                assert y_test is not None
                rtn += (self.mae(s_test, a_test, y_test, eta),)
            elif metric == 'er':
                assert y_test is not None and r_test is not None
                rtn += (self.reward(s_test, a_test, y_test, r_test, eta),)
            elif metric[:3] == 'ers':
                assert r_star_test is not None
                rtn += (self.reward_simulation(s_test, a_test, r_star_test, eta),)
            else:
                raise ValueError('Metric {:s} not implemented'.format(metric))
        return rtn
