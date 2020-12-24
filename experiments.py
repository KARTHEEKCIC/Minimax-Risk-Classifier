import numpy as np

from MRC0_1 import MRC0_1
from MRClog import MRClog
from MRC0_1_fixed_marginals import MRC0_1_fixed_marginals
from MRClog_fixed_marginals import MRClog_fixed_marginals

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

import warnings

import time

#import the datasets
from datasets import *

#data sets

loaders = [load_mammographic, load_haberman, load_indian_liver, load_diabetes, load_credit]
dataName= ["mammographic", "haberman", "indian_liver", "diabetes", "credit"]

clf_MRC_exp = [MRC0_1, MRClog]
clf_MRC_exp_desc = ["MRC with 0-1 loss", "MRC with log loss"]

is_log = [False, True]

clf_ConstMRC_exp = [MRC0_1_fixed_marginals, MRClog_fixed_marginals]
clf_ConstMRC_exp_desc = ["fixed marginals constrained MRC with 0-1 loss",
                         "fixed marginals constrained MRC with log loss"]

def ConstrainedMRCExp(s=0.3, mumVars= 400, _type='gaussian', _gamma='scale', random_seed= 1, _solver='SCS'):
    '''
    Experimentation: Constrained MRC with different losses and feature mappings
    '''
    res_mean = np.zeros(len(dataName))
    res_std = np.zeros(len(dataName))
    np.random.seed(random_seed)

    print('\n ####### The value of S is - ', s)
    print('\n\n')

    for k, classif in enumerate(clf_ConstMRC_exp):

        if _type == 'gaussian':
            print('\n ####### ' + str(_type) + ' (' + str(_gamma) + ') ' + str(clf_ConstMRC_exp_desc[k]) + '\n')
        else:
            print('\n ####### ' + str(_type) + ' ' + str(clf_ConstMRC_exp_desc[k]) + '\n')

        for j, load in enumerate(loaders):
            X, origY = load(return_X_y=True)
            n, d= X.shape

            #Map the values of Y from 0 to r-1
            domY= np.unique(origY)
            r= len(domY)
            Y= np.zeros(X.shape[0], dtype= np.int)
            for i,y in enumerate(domY):
                Y[origY==y]= i

            print(" ############## \n" + dataName[j] + " n= " + str(n) + " , d= " + str(d) + ", cardY= "+ str(r))

            clf = classif(r=r, s=s, solver=_solver, phi= _type, k=np.max((3,int(mumVars/(r*d)))), gamma='avg_ann_50')

            # Preprocess
            trans = SimpleImputer(strategy='median')
            X = trans.fit_transform(X, Y)

            # Generate the partitions of the stratified cross-validation
            cv = StratifiedKFold(n_splits=10, random_state=random_seed)

            np.random.seed(random_seed)
            cvError= list()
            auxTime= 0

            # Paired and stratified cross-validation
            for train_index, test_index in cv.split(X, Y):

                # Save start time for computing training time
                startTime= time.time()

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                # Normalizing the data
                std_scale = preprocessing.StandardScaler().fit(X_train, y_train)
                X_train = std_scale.transform(X_train)
                X_test = std_scale.transform(X_test)

                clf.fit(X_train, y_train)

                # Calculate the training time
                auxTime+= time.time() - startTime

                y_pred= clf.predict(X_test)

                cvError.append(np.average(y_pred != y_test))

            res_mean[j] = np.average(cvError)
            res_std[j] = np.std(cvError)

            print(" error= " + ":\t" + str(res_mean[j]) + "\t+/-\t" + str(res_std[j]) +
                "\navg_train_time= " + ":\t" + str(auxTime/10) + ' secs' + 
                "\n ############## \n\n\n")

def MRCExp(s=0.3, mumVars= 400, _type='gaussian', _gamma='scale', random_seed= 1, _solver='SCS'):
    '''
    Experimentation: MRC without additional constraints with different losses and feature mappings
    '''
    res_mean = np.zeros(len(dataName))
    res_std = np.zeros(len(dataName))
    np.random.seed(random_seed)

    print('\n ####### The value of S is - ', s)
    print('\n\n')

    # Loop through the variants of MRC using different losses
    for k, classif in enumerate(clf_MRC_exp):

        if _type == 'gaussian':
            print('\n ####### ' + str(_type) + ' (' + str(_gamma) + ') ' + str(clf_MRC_exp_desc[k]) + '\n')
        else:
            print('\n ####### ' + str(_type) + ' ' + str(clf_MRC_exp_desc[k]) + '\n')

        for j, load in enumerate(loaders):
            X, origY = load(return_X_y=True)
            n, d= X.shape

            #Map the values of Y from 0 to r-1
            domY= np.unique(origY)
            r= len(domY)
            Y= np.zeros(X.shape[0], dtype= np.int)
            for i,y in enumerate(domY):
                Y[origY==y]= i

            print(" ############## \n" + dataName[j] + " n= " + str(n) + " , d= " + str(d) + ", cardY= "+ str(r))

            clf = classif(r=r, s=s, solver=_solver, phi= _type, k=np.max((3,int(mumVars/(r*d)))), gamma='avg_ann_50')

            # Preprocess
            trans = SimpleImputer(strategy='median')
            X = trans.fit_transform(X, Y)

            # Generate the partitions of the stratified cross-validation
            cv = StratifiedKFold(n_splits=10, random_state=random_seed)

            np.random.seed(random_seed)
            cvError= list()

            # For the case log MRC, to calculate log error to check for bounds
            if is_log[k] == True:
                cvLogError = list()

            auxTime= 0
            upper= 0
            lower= 0
            # Paired and stratified cross-validation
            for train_index, test_index in cv.split(X, Y):

                # Save start time for computing training time
                startTime= time.time()

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                # Normalizing the data
                std_scale = preprocessing.StandardScaler().fit(X_train, y_train)
                X_train = std_scale.transform(X_train)
                X_test = std_scale.transform(X_test)

                clf.fit(X_train, y_train)
                upper += clf.upper
                lower += clf.getLowerBound()

                # Calculate the training time
                auxTime+= time.time() - startTime

                y_pred= clf.predict(X_test)

                if is_log[k] == True:
                    y_pred_proba = clf.predict_proba(X_test)
                    y_pred_proba = np.asarray([y_pred_proba[i, y_test[i]] for i in range(y_test.shape[0])])
                    cvLogError.append(np.average(-1*np.log(y_pred_proba)))

                cvError.append(np.average(y_pred != y_test))

            res_mean[j] = np.average(cvError)
            res_std[j] = np.std(cvError)

            print(" error= " + ":\t" + str(res_mean[j]) + "\t+/-\t" + str(res_std[j]) + "\n")

            # In case of log print the log error to check for bounds
            if is_log[k] == True:
                print(" log_error= " + ":\t" + str(np.average(cvLogError)) + "\n")

            print(" upper= " + str(upper/10)+"\t lower= " + str(lower/10) + 
                "\navg_train_time= " + ":\t" + str(auxTime/10) + ' secs' + 
                "\n ############## \n")


if __name__ == '__main__':

    # Supress the warnings
    warnings.simplefilter('ignore')

    print('######### EXPERIMENTATION WITH MIXED FEATURE MAPPINGS ######### \n\n')

    MRCExp(_type='gaussian', s=0.3, _solver = 'SCS')
    ConstrainedMRCExp(_type='gaussian', s=0.3, _solver = 'SCS')
