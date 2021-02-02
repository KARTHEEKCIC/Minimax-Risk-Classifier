# Import the feature mapping base class
from minimax_risk_classifiers.phi.phi import Phi
from sklearn.utils import check_array

class PhiLinear(Phi):
    """
    Phi (feature function) obtained using the linear kernel i.e.,
    the features are the instances itself with some intercept added.

    Parameters
    ----------
    n_classes : int
        The number of classes in the dataset

    Attributes
    ----------
    is_fitted_ : bool
        True if the feature mappings has learned its hyperparameters (if any)
        and the length of the feature mapping is set.

    """

    def fit(self, X, Y=None):
        """
        Learn the set of Phi features from the dataset by one-hot encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances used to learn the feature configurations

        Y : array-like of shape (n_samples,), default=None
            This argument will never be used in this case. 
            It is present in the signature for consistency 
            in the signature among different feature mappings.

        Returns
        -------
        self : 
            Fitted estimator

        """

        X = check_array(X, accept_sparse=True)

        d= X.shape[1]
        # Defining the length of the phi
        self.m = d+1
        self.len = self.m*self.n_classes
        self.is_fitted_ = True
        
        return self

    def transform(self, X):
        """
        Transform the given instances to the features.
        The features in case of a linear kernel are the instances itself.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances.

        Returns
        -------
        X_feat : array-like of shape (n_samples, n_features)
            Transformed features from the given instances i.e., 
            the instances itself.

        """

        X = check_array(X, accept_sparse=True)
        X_feat = X

        return X_feat

    