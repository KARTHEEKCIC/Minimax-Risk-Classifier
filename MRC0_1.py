import numpy as np
import cvxpy as cvx

# Import the feature mapping
from phi import Phi

import time

from sklearn.base import BaseEstimator, ClassifierMixin


class MRC0_1(BaseEstimator, ClassifierMixin):
    '''
    Minimax risk classifier with 0-1 loss using different feature mappings
    '''

    def __init__(self, r, equality=False, s=0.25, deterministic=False, seed=0, solver='SCS', phi='gaussian', k=400, gamma='avg_ann_50'):
        '''

        @param r: the number of values of class variable
        @param solver: the solver to use for optimization
        @param phi: features of the LPC
        @param s: 
        @param equality: the type of Learning. If true the LPC is asymptotically calibrated, if false the LPC is
                         approximately calibrated.
        @param deterministic: if deterministic is false the LPC decision function is arg_c rand p(c|x) and if it is true
                              the decision function is arg_c max p(c|x)
        @param seed: random seed
        @param solvers: list of solvers available in cvxpy (will be depreciated in future)
        '''
        self.r= r
        self.equality = equality
        self.s = s
        self.deterministic = deterministic
        self.seed= seed
        self.solver = solver

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
        self.phi.fit(X, Y)
        self._minimaxRisk(X,Y)


    def _minimaxRisk(self, X, Y):
        '''
        Solves the minimax risk problem

        @param X: unlabeled instances, numpy.array(float) with shape (numInstances,numDims)
        @param Y: the labels, numpy.array(int) with shape (numInstances,)
        @return: the upper bound of the MRC given directly by the solution to the minimax risk problem
        '''

        # Constants
        n, d= X.shape
        m= self.phi.len

        self.tau= self.phi.estExp(X,Y)
        self.sigma= self.phi.estStd(X,Y)

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

        if self.linConstr:
            #Exponential number in num_class of linear constraints
            M = self.phi.getLearnConstr(self.linConstr)
            #F is the sum of phi for different subset of Y for each data point in case of linear constr 
            F = M[:, :m]
            cardS= M[:, -1]
            numConstr= M.shape[0]
            constraints.extend([F[i, :]@mu + cardS[i]*nu + cardS[i]*1 <= 1 for i in range(numConstr)])
        else:
            #Constant number in num_class of non-linear constraints
            F = self.phi.getLearnConstr(self.linConstr)
            numConstr = F.shape[0]
            constraints.extend([cvx.sum(cvx.pos((np.ones(self.r) + F[i, :, :]@mu + np.ones(self.r) * nu))) <= 1 for i in range(numConstr)])

        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        _ = prob.solve(solver=self.solver, verbose=False)

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

        @param X: unlabeled instances, numpy.array(float) with shape (numInstances,numDims)
        @param Y: the labels, numpy.array(int) with shape (numInstances,)
        @return: the lower bound for MRC given by solving an optimization
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
        eps = np.clip(1 + F@self.mu + self.nu, 0, None)
        c= np.sum(eps, axis=1)
        zeros= np.isclose(c, 0)
        c[zeros]= 1
        eps[zeros, :]= 1/self.r
        c= np.tile(c, (self.r, 1)).transpose()
        eps/= c
        eps = 1 - eps

        constraints.extend(
            [F[j, y, :]@low_mu + low_nu <= eps[j, y]
             for j in range(numConstr) for y in range(self.r)])


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
        Conditional probabilities corresponding to each class for each unlabeled instance

        @param X: the unlabeled instances, np.array(double) n_instances X dimensions
        @return: p(Y|X), np.array(float) n_instances X n_classes
        '''

        # n_instances X n_classes X phi.len
        Phi = self.phi.eval(X)

        # Unnormalized conditional probabilityes
        hy_x = np.clip(1 + np.dot(Phi, self.mu) + self.nu, 0., None)


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

        proba = self.predict_proba(X)

        if self.deterministic:
            ind = np.argmax(proba, axis=1)
        else:
            np.random.seed(self.seed)
            ind = [np.random.choice(self.r, size= 1, p=pc)[0] for pc in proba]

        return ind
