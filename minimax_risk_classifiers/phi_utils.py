import numpy as np
from sklearn.tree import DecisionTreeClassifier

def d_tree_split(X, Y, k=None):
    """
    Learn the univariate thresholds 
    by using the split points of decision trees for each dimension of data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_dimensions)
        Unlabeled instances.

    Y : array-like of shape (n_samples,)
        Labels corresponding to the instances.

    k : int, default = None
        Maximum limit on the number of thresholds obtained

    Returns
    -------
    prodThrsDim : array-like of shape (n_thresholds)
        The dimension in which the thresholds are defined.

    prodThrsVal : array-like of shape (n_thresholds)
        The threshold value in the corresponding dimension.

    """

    (n, d) = X.shape

    prodThrsVal = []
    prodThrsDim = []

    # One order thresholds: all the univariate thresholds
    for dim in range(d):
        if k== None:
            dt = DecisionTreeClassifier()
        else:
            dt= DecisionTreeClassifier(max_leaf_nodes=k+1)

        dt.fit(np.reshape(X[:,dim],(n,1)),Y)

        dimThrsVal= np.sort(dt.tree_.threshold[dt.tree_.threshold!= -2])

        for t in dimThrsVal:
            prodThrsVal.append([t])
            prodThrsDim.append([dim])

    return prodThrsDim, prodThrsVal

def thrs_features(X, thrsDim, thrsVal):
    """
    Find the features of the given instances based on the 
    thresholds obtained using the instances

    Parameters
    ----------
    X : array-like of shape (n_samples, n_dimensions)
        Unlabeled instances.

    thrsDim : array-like of shape (n_thresholds)
        The dimension in which the thresholds are defined. 

    thrsVal : array-like of shape (n_thresholds)
        The threshold value in the corresponding dimension.

    Returns
    -------
    X_feat : array-like of shape (n_instances, n_thresholds)
        The 0-1 features developed using the thresholds.

    @param thrsDim: dimension of univariate thresholds given in the form of array of arrays like - [[0], [1], ...]
    @param thrsVal: value of the univariate thresholds given in the form of array of arrays like - [[0.5], [0.7], ...]
    @return: the 0-1 features developed using the thresholds
    """

    n = X.shape[0]

    # Store the features based on the thresholds obtained
    X_feat = np.zeros((n, len(thrsDim)), dtype=int)

    # Calculate the threshold features
    for thrsInd in range(len(thrsDim)):
        X_feat[:, thrsInd] = np.all(X[:, thrsDim[thrsInd]] <= thrsVal[thrsInd],
                                axis=1).astype(np.int)

    return X_feat

