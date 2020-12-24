
# Import the feature mapping
from phi import Phi

import numpy as np
import cvxpy as cvx
import scipy.special as scp

import time

from sklearn.base import BaseEstimator, ClassifierMixin


class MRClog(BaseEstimator, ClassifierMixin):
    '''
    Minimax risk classifier with using univariate threshold-based feature mappings
    Submitted to ICML 2020
    '''

    def __init__(self, r, equality=False, s=0.25, deterministic=True, seed=0, solver='SCS', phi='gaussian', k=400, gamma='avg_ann_50'):
        '''

        :param r: the number of values of class variable
        :param phi: Features of the LPC
        :param equality: the type of Learning. If true the LPC is asymptotically calibrated, if false the LPC is
        approximately calibrated.
        :param deterministic: if deterministic is false the LPC decision function is arg_c rand p(c|x) and if it is true
        the decision function is arg_c max p(c|x)
        :param seed: random seed
        '''
        self.r= r
        self.solver = solver
        self.equality = equality
        self.s = s
        self.deterministic = deterministic
        self.seed= seed

        if self.r> 4:
            self.linConstr= False
        else:
            self.linConstr= True

        # Define the feature mapping
        self.phi = Phi(r=self.r, _type=phi, k=k, gamma=gamma)

        # solver list available in cvxpy
        solvers = ['SCS', 'ECOS']

    def fit(self, X, Y):
        """
        Fit learning using....

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        y : array-like, shape (n_samples)

        Returns
        -------
        self : returns an instance of self.

        """
        self.phi.linConstr= self.linConstr
        self.phi.fit(X, Y)
        self._minimaxRisk(X,Y)


    def _minimaxRisk(self, X, Y):
        '''
        Solves the minimax risk problem
        :param X: unlabeled instances, numpy.array(float) with shape (numInstances,numDims)
        :param Y: the labels, numpy.array(int) with shape (numInstances,)
        :return: the upper bound of the MRC given directly by the solution to the minimax risk problem
        '''

        # Constants
        n= X.shape[0]
        m= self.phi.len

        self.tau= self.phi.estExp(X,Y)
        self.sigma = self.phi.estStd(X,Y)

        self.a= self.tau- (self.s * self.sigma)/np.sqrt(n)
        self.b= self.tau+ (self.s * self.sigma)/np.sqrt(n)

        # Variables
        mu = cvx.Variable(m)
        zhi = cvx.Variable(m)
        nu = cvx.Variable()

        # Cost function
        cost = (1/2)*(self.b - self.a).T@zhi - (1/2)*(self.b + self.a).T@mu - nu

        # Objective function
        objective = cvx.Minimize(cost)

        # Constraints
        constraints= [zhi + mu >= 0, zhi - mu >= 0]

        F = self.phi.getLowerConstr()
        numConstr = F.shape[0]
        constraints.extend([cvx.log_sum_exp(F[i, :, :]@mu + np.ones(self.r) * nu) <= 0 for i in range(numConstr)])

        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        _ = prob.solve(solver=self.solver, verbose=False)

        # Optimal values
        self.mu= mu.value
        self.zhi= zhi.value
        self.nu= nu.value

        # if the solver could not find values of mu for the given solver
        if self.mu is None or self.zhi is None or self.nu is None:

            # try with a different solver for solution
            for s in self.solvers:
                if s != self.solver:

                    # Solve the problem
                    _ = prob.solve(solver=s, verbose=False)

                    # Check the values
                    self.mu= mu.value
                    self.zhi= zhi.value
                    self.nu= nu.value

                    # Break the loop once the solution is obtained
                    if self.mu is not None and self.zhi is not None and self.nu is not None:
                        break

        # Upper bound
        self.upper= (1/2)*(self.b - self.a).T@zhi.value - (1/2)*(self.b + self.a).T@mu.value - nu.value

    def getLowerBound(self):
        '''
        Obtains the lower bound of the fitted classifier: unbounded...

        :param X: unlabeled instances, numpy.array(float) with shape (numInstances,numDims)
        :param Y: the labels, numpy.array(int) with shape (numInstances,)
        :return: the lower bound of the MRC by solving an additional optimization
        '''

        # Variables
        m= self.phi.len
        low_mu = cvx.Variable(m)
        low_zhi = cvx.Variable(m)
        low_nu = cvx.Variable()

        # Cost function
        cost = (1/2)*(self.b + self.a).T@low_mu - (1/2)*(self.b - self.a).T@low_zhi + low_nu

        # Objective function
        objective = cvx.Maximize(cost)

        # Constraints
        constraints= [low_zhi + low_mu >= 0, low_zhi - low_mu >= 0]

        F = self.phi.getLowerConstr()
        numConstr= F.shape[0]

        # epsilon
        eps = F@self.mu
        eps = np.tile(scp.logsumexp(eps, axis=1), (self.r, 1)).transpose() - eps

        constraints.extend(
            [F[i, :, :]@low_mu + low_nu <= eps[i, :] \
                    for i in range(numConstr)])

        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        _ = prob.solve(solver=self.solver, verbose=False)

        # Lower bound
        self.mu_l= low_mu.value
        self.zhi_l= low_zhi.value
        self.nu_l= low_nu.value

        # if the solver could not find values of mu for the given solver
        if self.mu_l is None or self.zhi_l is None or self.nu_l is None:

            # try with a different solver for solution
            for s in self.solvers:
                if s != self.solver:

                    # Solve the problem
                    _ = prob.solve(solver=s, verbose=False)

                    # Check the values
                    self.mu_l= low_mu.value
                    self.zhi_l= low_zhi.value
                    self.nu_l= low_nu.value

                    # Break the loop once the solution is obtained
                    if self.mu_l is not None and self.zhi_l is not None and self.nu_l is not None:
                        break

        self.mu_l[np.isclose(self.mu_l,0)]= 0
        self.zhi_l[np.isclose(self.zhi_l,0)]= 0
        self.nu_l[np.isclose(self.nu_l,0)]= 0

        # Get the lower bound
        self.lower= (1/2)*(self.b + self.a).T@self.mu_l - (1/2)*(self.b - self.a).T@self.zhi_l + self.nu_l

        return self.lower

    def predict_proba(self, X):
        '''
        Return the class conditional probabilities for each unlabeled instance along 
        with vector v used for the prediction
        :param X: the unlabeled instances, np.array(double) n_instances X dimensions
        :return: p(Y|X), np.array(float) n_instances X n_classes
        '''

        # n_instances X n_classes X phi.len
        Phi = self.phi.eval(X)

        v = np.dot(Phi, self.mu)

        # Unnormalized conditional probabilityes
        hy_x = np.vstack(np.sum(np.exp(v - np.tile(v[:,i], (self.r, 1)).transpose()), axis=1) \
                        for i in range(self.r)).transpose()

        hy_x = np.reciprocal(hy_x)                

        return hy_x

    def predict(self, X):
        '''Returns the predicted classes for X samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        returns
        -------
        y_pred : array-like, shape (n_samples, )
            y_pred is of the same type as self.classes_.

        '''

        # n_instances X n_classes X phi.len
        Phi = self.phi.eval(X)

        # Deterministic classification
        v = np.dot(Phi, self.mu)
        ind = np.argmax(v, axis=1)

        # if not self.deterministic:
        #     np.random.seed(self.seed)

        # proba = self.predict_proba(X)

        # if self.deterministic:
        #     ind = np.argmax(proba, axis=1)
        # else:
        #     ind = [np.random.choice(self.r, size= 1, p=pc)[0] for pc in proba]

        return ind
