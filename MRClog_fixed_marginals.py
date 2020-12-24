
# Import the feature mapping
from phi import Phi

import numpy as np
import cvxpy as cvx
import itertools as it
import scipy.special as scs

import time

from sklearn.base import BaseEstimator, ClassifierMixin


class MRClog_fixed_marginals(BaseEstimator, ClassifierMixin):
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
        :param solver: determines the solver to use for optimization
        :param deterministic: if deterministic is false the LPC decision function is arg_c rand p(c|x) and if it is true
        the decision function is arg_c max p(c|x)
        :param seed: random seed
        '''
        self.r= r
        self.equality = equality
        self.s = s
        self.solver = solver
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
        self.phi.fit(X, Y, learn_config=False)
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

        Phi= self.phi.eval(X)

        # Objective function
        objective = cvx.Minimize((1/2)*(self.b - self.a).T@zhi - (1/2)*(self.b + self.a).T@mu)

        for i in range(n):
            objective = objective + cvx.Minimize((1/n)*cvx.log_sum_exp(Phi[i,:,:]@mu))

        # Constraints
        constraints= [zhi + mu >= 0, zhi - mu >= 0]

        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        _ = prob.solve(solver=self.solver, verbose=False)

        # Optimal values
        self.mu= mu.value
        self.zhi= zhi.value

        # if the solver could not find values of mu for the given solver
        if self.mu is None or self.zhi is None:

            # try with a different solver for solution
            for s in self.solvers:
                if s != self.solver:

                    # Solve the problem
                    _ = prob.solve(solver=s, verbose=False)

                    # Check the values
                    self.mu= mu.value
                    self.zhi= zhi.value

                    # Break the loop once the solution is obtained
                    if self.mu is not None and self.zhi is not None:
                        break


    def predict_proba(self, X):
        '''
        Return the class conditional probabilities for each unlabeled instance
        :param X: the unlabeled instances, np.array(double) n_instances X dimensions
        :return: p(Y|X), np.array(float) n_instances X n_classes
        '''

        # n_instances X n_classes X phi.len
        Phi = self.phi.eval(X)

        v = np.dot(Phi, self.mu)

        # Unnormalized conditional probabilities
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

        if not self.deterministic:
            np.random.seed(self.seed)

        proba = self.predict_proba(X)

        if self.deterministic:
            ind = np.argmax(proba, axis=1)
        else:
            ind = [np.random.choice(self.r, size= 1, p=pc)[0] for pc in proba]

        return ind
