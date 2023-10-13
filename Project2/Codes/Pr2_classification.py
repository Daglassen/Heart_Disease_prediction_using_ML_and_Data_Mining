import pandas as pd
import numpy as np
from toolbox_02450 import mcnemar
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
import scipy.stats as st
from matplotlib.pyplot import figure, plot, subplot, title, hist, xlabel, ylabel, show, legend
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)

#from scipy.linalg import svd
#import matplotlib.pyplot as plt

#### Functions to select hyperparameters in inner cross validation ####

def log_validate(X,y,lambdas,K):
    CV = model_selection.KFold(K, shuffle=True)
    N,M = X.shape
    test_error = np.empty((K,len(lambdas)))
    
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma
        
        for l in range(0,len(lambdas)):
        
            mdl = LogisticRegression(penalty='l2', C=1/lambdas[l] )
            mdl.fit(X_train, y_train)
            y_test_est = mdl.predict(X_test).T
            test_error[k,l] = np.sum(y_test_est != y_test) / len(y_test)
            
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    return opt_lambda


def knn_validate(X,y,Ks,K):
    CV = model_selection.KFold(K, shuffle=True)
    N,M = X.shape
    test_error = np.empty((K,len(Ks)))
    
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X - mu) / sigma
    
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        for l in range(0,len(Ks)):
            
            knclassifier = KNeighborsClassifier(n_neighbors=Ks[l], p=2, 
                                    metric='minkowski',
                                    metric_params={})
            knclassifier.fit(X_train, y_train)
            y_test_est = knclassifier.predict(X_test)
            test_error[k,l] = np.sum(y_test_est != y_test) / len(y_test)
            
    opt_K = Ks[np.argmin(np.mean(test_error,axis=0))]
    return opt_K


#### read data ####

filename = 'SouthAfricanHeartDiseaseFixed.txt'
df = pd.read_csv(filename)
raw_data = df.to_numpy()

# create y and X without rownames and family history
y = np.array(raw_data[:,10],dtype=np.int32)
X = np.delete(raw_data,[0,5,10],axis=1)
X = np.array(X,dtype=np.float64)

# define other relevant variables
N,M = np.shape(X)
attributeNames = list(df.columns)
del attributeNames[10]
del attributeNames[5]
del attributeNames[0]
C = 2
classNames = ['healthy','sick']

# cross-validation parameters
K = 10
internal_cross_validation = 10
CV = model_selection.KFold(K, shuffle=True)

# hyperparameters
lambdas = np.linspace(1,200,10)
Ks = range(1,10)

# Store results for table 1
Error_test_logistic = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
Error_test_knn = np.empty((K,1))
lambda_logistic = np.empty((K,1))
K_knn = np.empty((K,1))

# Store predictions for statistics part
pred_logistic = np.empty((N))
pred_nofeatures = np.empty((N))
pred_knn = np.empty((N))

# Loop over outer folds
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    
    # extract training and test for this outer fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # inner cross validation to get optimal lambda for logistic model
    opt_lambda = log_validate(X_train, y_train, lambdas, internal_cross_validation)
    lambda_logistic[k] = opt_lambda
    
    # standardize train and test
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma   
    
    # fit logistic model on whole outer fold training set    
    mdl = LogisticRegression(penalty='l2', C=1/opt_lambda )
    mdl.fit(X_train, y_train)
    y_test_est = mdl.predict(X_test).T
    pred_logistic[test_index] = y_test_est.astype(int)
    Error_test_logistic[k] = np.sum(y_test_est != y_test) / len(y_test)
    
    # baseline model (most prevalent class)
    y_test_est = (np.repeat(round(y_test.sum()/len(y_test)),len(y_test))).astype(int)
    pred_nofeatures[test_index] = y_test_est.astype(int)
    Error_test_nofeatures[k] = np.sum(y_test_est != y_test) / len(y_test)
    
    # get optimal K nearest neighbors for KNN model
    opt_K = knn_validate(X_train,y_train,Ks,internal_cross_validation)
    K_knn[k] = opt_K
    
    # fit KNN model on whole outer fold training set
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma 

    knclassifier = KNeighborsClassifier(n_neighbors=opt_K, p=2, 
                                    metric='minkowski',
                                    metric_params={})
    knclassifier.fit(X_train, y_train)
    y_test_est = knclassifier.predict(X_test)
    pred_knn[test_index] = y_test_est.astype(int)
    Error_test_knn[k] = np.sum(y_test_est != y_test) / len(y_test)
    
    
# statistically compare models (McNemars test, setup I)
# logistic and baseline as example
[thetahat, CI, p] = mcnemar(y, pred_logistic, pred_nofeatures, alpha=0.05)

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_test_est);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
figure(2);
imshow(cm, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

show()



