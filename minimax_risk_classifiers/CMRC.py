# Import the MRC super class
from _MRC_ import _MRC_

import numpy as np
import cvxpy as cvx
import itertools as it
import scipy.special as scs
from sklearn.base import BaseEstimator, ClassifierMixin


class CMRC(BaseEstimator, ClassifierMixin, _MRC_):
    """
    Minimax risk classifier using the additional marginals constraints on the instances.
    It also implements two kinds of loss functions, namely 0-1 and log loss.
    This is a subclass of the super class _MRC_.
    """

    def _minimaxRisk(self, X, Y):
        """
        Solves the marginally constrained minimax risk problem 
        for different types of loss (0-1 and log loss)

        Parameters 
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Training instances.

        Y : array-like of shape (n_samples,)
            Labels corresponding to the instances.

        """

        # Constants
        n= X.shape[0]
        m= self.phi.len

        # Variables
        mu = cvx.Variable(m)
        zhi = cvx.Variable(m)

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # # We compute the learn constraints without omitting the duplicate elements
            M= self.phi.getAllSubsetConfig(self.phi.getLearnConfig())
            # F is the sum of phi for different subset of Y for each data point 
            F = M[:, :m]
            cardS= M[:, -1]    

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

        elif self.loss == 'log':
            # Constraints in case of log loss function

            # Objective function
            objective = cvx.Minimize((1/2)*(self.b - self.a).T@zhi - (1/2)*(self.b + self.a).T@mu)
            for i in range(n):
                objective = objective + cvx.Minimize((1/n)*cvx.log_sum_exp(self.phi.getLearnConfig()[i,:,:]@mu))

        # Constraints
        constraints= [zhi + mu >= 0, zhi - mu >= 0]

        self.mu, self.zhi = self.trySolvers(objective, constraints, mu, zhi)

    def predict_proba(self, X):
         """
        Conditional probabilities corresponding to each class for each unlabeled instance

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Testing instances for which 
            the prediction probabilities are calculated for each class.
        
        Returns
        -------
        hy_x : ndarray of shape (n_samples, n_classes)
            The probabilities (p(y|x)) corresponding to the predictions
            for each class.

        """

        n = X.shape[0]
        # n_instances X n_classes X phi.len
        Phi= self.phi.eval(X)
        m = self.phi.len

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # M= self.phi.getAllSubsetConfig(Phi, True)
            # # F is the sum of phi for different subset of Y for each data point 
            # F = M[:, :m]
            # print('This is : ', F.shape)
            # print('M : ', self.mu)
            # cardS= M[:, -1]

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
            hy_x = hy_x / c

        elif self.loss == 'log':
            # Constraints in case of log loss function

            v = np.dot(Phi, self.mu)

            # Unnormalized conditional probabilities
            hy_x = np.vstack(np.sum(np.exp(v - np.tile(v[:,i], (self.r, 1)).transpose()), axis=1) \
                        for i in range(self.r)).transpose()
            hy_x = np.reciprocal(hy_x)

        return hy_x

    def setLearnConfigType(self):
        """
        Learn the duplicate configuration in F to be used in the objective function 
        in case of this constrained MRC. Duplicate configuration are observed 
        when the dataset contains duplication entries.
        """
        self.learn_duplicates = True
