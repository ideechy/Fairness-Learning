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
        super().__init__(s_train, a_train, y_train, preprocess_method, mode='predict', a_iscategory=a_iscategory)
        assert r_train.ndim == 2 and r_train.shape == (self.n, 1)
        self.r_train = r_train
        # training features with intercept term, shape=(n, d+c)
        dat_train = np.column_stack((self.s_train, self.a_train))
        # machine learning model of r
        self.rml = sm.Logit(
            (r_train[y_train.squeeze() == 1] + 1) / 2, 
            dat_train[y_train.squeeze() == 1]
        ).fit(disp=False)

    def f_expit(self, eta, s=None, a=None):
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
        if a is None:
            a_prime = self.a_prime
        else:
            assert s is not None
            a_prime = self.process(s, a)
        eta = np.asarray(eta).reshape(-1, )
        assert eta.shape[0] == self.d + 1
        return expit(np.dot(sm.add_constant(a_prime), eta))

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
        """Augmented inverse probability weighted estimation of the expected reward

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
        y_ml, y_ftu, y_aa, y_1, y_2, y_eta = \
            np.zeros(self.c), np.zeros(self.c), np.zeros(self.c), np.zeros(self.c), np.zeros(self.c), np.zeros(self.c)
        for g in range(self.c):
            a_prime = self.process_margin(s, a, g)
            y_ml[g] = np.mean(self.f_ml(np.broadcast_to(g, s.shape), a_prime))
            y_ftu[g] = np.mean(self.f_ftu(a_prime))
            y_aa[g] = np.mean(self.f_aa(np.broadcast_to(g, s.shape), a_prime))
            y_1[g] = np.mean(self.f_1(np.broadcast_to(g, s.shape), a_prime))
            y_2[g] = np.mean(self.f_2(np.broadcast_to(g, s.shape), a_prime))
            y_eta[g] = np.mean(self.f_expit(eta, np.broadcast_to(g, s.shape), a_prime))
        cf_ml = np.max(np.abs(y_ml.reshape(-1, 1) - y_ml))
        cf_ftu = np.max(np.abs(y_ftu.reshape(-1, 1) - y_ftu))
        cf_aa = np.max(np.abs(y_aa.reshape(-1, 1) - y_aa))
        cf_1 = np.max(np.abs(y_1.reshape(-1, 1) - y_1))
        cf_2 = np.max(np.abs(y_2.reshape(-1, 1) - y_2))
        cf_eta = np.max(np.abs(y_eta.reshape(-1, 1) - y_eta))
        return np.asarray((cf_ml, cf_ftu, cf_aa, cf_1, cf_2, cf_eta))

    def mae(self, s, a, y, eta):
        """Mean Absolute Error in test data.

        Args:
            s (numpy.ndarray): categorical sensitive test attributes,
                must have shape (*, 1).
            a (numpy.ndarray): non-sensitive test attributes, must
                have shape (*, d).
            y (numpy.ndarray): binary decisions with size *.
            eta (numpy.ndarray): index parameter of the decision class.

        """
        y = np.array(y).squeeze()
        mae_ml = np.mean(np.abs(self.f_ml(s, a) - y))
        mae_ftu = np.mean(np.abs(self.f_ftu(a) - y))
        mae_aa = np.mean(np.abs(self.f_aa(s, a) - y))
        mae_1 = np.mean(np.abs(self.f_1(s, a) - y))
        mae_2 = np.mean(np.abs(self.f_2(s, a) - y))
        mae_eta = np.mean(np.abs(self.f_expit(eta, s, a) - y))
        return np.hstack((mae_ml, mae_ftu, mae_aa, mae_1, mae_2, mae_eta))

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
        r_ftu = np.mean(self.f_ftu(a) * r_star) 
        r_aa = np.mean(self.f_aa(s, a) * r_star) 
        r_1 = np.mean(self.f_1(s, a) * r_star) 
        r_2 = np.mean(self.f_2(s, a) * r_star) 
        r_eta = np.mean(self.f_expit(eta, s, a) * r_star) 
        return np.hstack((r_ml, r_ftu, r_aa, r_1, r_2, r_eta))