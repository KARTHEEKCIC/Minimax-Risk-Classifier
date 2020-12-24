
# Import the feature mapping
from phi import Phi

import numpy as np
import cvxpy as cvx
import itertools as it
import scipy.special as scs

import time

from sklearn.base import BaseEstimator, ClassifierMixin


class MRC0_1_fixed_marginals(BaseEstimator, ClassifierMixin):
    '''
    Minimax risk classifier with using univariate threshold-based feature mappings
    Submitted to ICML 2020
    '''

    def __init__(self, r, equality=False, s=0.25, deterministic=False, seed=0, solver='SCS', phi='gaussian', k=400, gamma='avg_ann_50'):
        '''

        @param r: the number of values of class variable
        @param phi: Features of the LPC
        @param equality: the type of Learning. If true the LPC is asymptotically calibrated, if false the LPC is
                         approximately calibrated.
        @param deterministic: if deterministic is false the LPC decision function is arg_c rand p(c|x) and if it is true
                                the decision function is arg_c max p(c|x)
        @param seed: random seed
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
        self.solvers = ['SCS', 'ECOS']

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

        @param X: unlabeled instances, numpy.array(float) with shape (numInstances,numDims)
        @param Y: the labels, numpy.array(int) with shape (numInstances,)
        @return: the upper bound of the MRC given directly by the solution to the minimax risk problem
        '''

        # Constants
        n= X.shape[0]
        m= self.phi.len

        self.tau= self.phi.estExp(X,Y)
        self.sigma= self.phi.estStd(X,Y)

        self.a= self.tau- (self.s * self.sigma)/np.sqrt(n)
        self.b= self.tau+ (self.s * self.sigma)/np.sqrt(n)

        # Variables
        mu = cvx.Variable(m)
        zhi = cvx.Variable(m)

        # We compute the F here again(apart from phi) as we want to use the duplicate data points
        # The F function computed by phi avoids duplicate values

        # F is the sum of phi for different subset of Y for each data point 
        Phi= self.phi.eval(X)
        F= np.vstack((np.sum(Phi[:, S, ], axis=1)
                            for numVals in range(1, self.r+1)
                            for S in it.combinations(np.arange(self.r), numVals)))
        # Number of classes in each set
        cardS= np.arange(1, self.r+1).repeat([n*scs.comb(self.r, numVals)
                                            for numVals in np.arange(1, self.r+1)])        

        # Objective function
        objective = cvx.Minimize((1/2)*(self.b - self.a).T@zhi - (1/2)*(self.b + self.a).T@mu)

        # Calculate the psi function and add it to the objective function
        # First we calculate the all possible values of psi for all the points
        psi_arr = (np.ones(cardS.shape[0])-(F@mu + cardS))/cardS
        for i in range(n):
            # Get psi for each data point and add the min value to objective
            psi_arr_xi = psi_arr[np.arange(i, psi_arr.shape[0], n)]
            objective = objective + cvx.Minimize((-1/n)*cvx.min((psi_arr_xi)))

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
        Conditional probabilities corresponding to each class for each unlabeled instance

        @param X: the unlabeled instances, np.array(double) n_instances X dimensions
        @return: p(Y|X), np.array(float) n_instances X n_classes
        '''

        n = X.shape[0]
        # n_instances X n_classes X phi.len
        Phi= self.phi.eval(X)

        # Sum of phi for different subset of Y for each data point
        F= np.vstack((np.sum(Phi[:, S, ], axis=1)
                            for numVals in range(1, self.r+1)
                            for S in it.combinations(np.arange(self.r), numVals)))
        # Number of classes in each set
        cardS= np.arange(1, self.r+1).repeat([n*scs.comb(self.r, numVals)
                                            for numVals in np.arange(1, self.r+1)])

        # Compute psi
        psi = np.zeros(n)

        # First we calculate the all possible values of psi for all the points
        psi_arr = (np.ones(cardS.shape[0])-(F@self.mu + cardS))/cardS

        for i in range(n):
            # Get psi values for each data point and find the min value
            psi_arr_xi = psi_arr[np.arange(i, psi_arr.shape[0], n)]
            psi[i] = np.min(psi_arr_xi)

        # Conditional probabilities
        hy_x = np.clip(np.ones((n,self.r)) + np.dot(Phi, self.mu) + \
                np.tile(psi, (self.r,1)).transpose(), 0., None)


        # normalization constraint
        c = np.sum(hy_x, axis=1)
        # check when the sum is zero
        zeros = np.isclose(c, 0)
        c[zeros] = 1
        hy_x[zeros, :] = 1 / self.r
        c = np.tile(c, (self.r, 1)).transpose()

        return hy_x / c

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
