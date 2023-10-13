import pandas as pd
import numpy as np
import torch
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
from sklearn import model_selection
from scipy import stats
from matplotlib.pyplot import figure, plot, subplot, title, hist, xlabel, ylabel, show, legend
#from scipy.linalg import svd
#import matplotlib.pyplot as plt

#### This function is used to select best h (it does the inner cross validation) ####

def ann_validate(X,y,hs,K=10,max_iter=2000):
    
    test_error = np.empty((K,len(hs)))
    N,M = X.shape
    n_replicates = 1
    CV = model_selection.KFold(K, shuffle=True)
    loss_fn = torch.nn.MSELoss() 
    
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):  
        
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        y_train = torch.reshape(y_train, (y_train.shape[0], 1))
        y_test = torch.reshape(y_test, (y_test.shape[0], 1))
        
        for l in range(0,len(hs)):
            
            n_hidden_units = hs[l]
            model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        )
        
            print('fitting inner loop',k,'of',K,',','for h=',n_hidden_units)
            net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
            
            y_test_est = net(X_test)
            se = (y_test_est.float()-y_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            test_error[k,l] = mse
       
    opt_h = hs[np.argmin(np.mean(test_error,axis=0))]
    return opt_h

filename = 'SouthAfricanHeartDiseaseFixed.txt'
df = pd.read_csv(filename)
raw_data = df.to_numpy()

#convert string variable to numeric
hist = raw_data[:,5]
raw_data[:,5] = np.unique(hist, return_inverse=True)[1].tolist()


#### Regression Part A - regularized linear regression ####


# create X and y. predict bp ~ (tobbacco, ldl, adiposity)
X = np.delete(raw_data,[0,1],axis=1)
X = np.array(X,dtype=np.float64)
y = np.array(raw_data[:,1], dtype=np.float64)


# add offset attribute to X
Xr = np.concatenate((np.ones((X.shape[0],1)),X),1)

# estimate generalization error for different lambda
K = 5
lambdas = np.linspace(0,200,20)
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(Xr, y, lambdas, K)

# plot lambda vs. estimated generalization error
f = figure()
plot(lambdas, train_err_vs_lambda, '-*')
plot(lambdas, test_err_vs_lambda, '-*')
xlabel('lambda')
ylabel('est. gen. error')
legend(['Error_train','Error_test'])

    
#### Regression Part B - compare rlr, ANN, and baseline using 2-layer cross-validation ####

# Xr, Mr is with the offset attribute
N, M = X.shape
N, Mr = Xr.shape

# Hyperparameters
lambdas = np.linspace(0,200,10)
hs = range(1,5)

# Store results
Error_test_rlr = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
Error_test_ANN = np.empty((K,1))
lambda_rlr = np.empty((K,1))
h_ANN = np.empty((K,1))

# Cross validation settings
K = 5
internal_cross_validation = 5
CV = model_selection.KFold(K, shuffle=True)

# ANN settings
loss_fn = torch.nn.MSELoss()
n_replicates = 1
max_iter = 2000

# Loop over outer folds
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    
    Xr_train = Xr[train_index]
    y_train = y[train_index]
    Xr_test = Xr[test_index]
    y_test = y[test_index]
       
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(Xr_train, y_train, lambdas, internal_cross_validation)
    lambda_rlr[k] = opt_lambda

    # Standardize outer fold based on training set
    mu = np.mean(Xr_train[:, 1:], 0)
    sigma = np.std(Xr_train[:, 1:], 0)
    Xr_train[:, 1:] = (Xr_train[:, 1:] - mu ) / sigma
    Xr_test[:, 1:] = (Xr_test[:, 1:] - mu ) / sigma
    
    Xty = Xr_train.T @ y_train
    XtX = Xr_train.T @ Xr_train
    
    # Baseline test error
    Error_test_nofeatures[k] = np.square(y_test-y_train.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(Mr)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # rlr test error with optimal lambda
    Error_test_rlr[k] = np.square(y_test-Xr_test @ w_rlr).sum(axis=0)/y_test.shape[0]

    # now do ANN
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    y_train = torch.reshape(y_train, (y_train.shape[0], 1))
    y_test = torch.reshape(y_test, (y_test.shape[0], 1))
    
    # get optimal h
    n_hidden_units = ann_validate(X_train, y_train, hs, internal_cross_validation, max_iter)
    h_ANN[k] = n_hidden_units
    
    # fit net to entire training set with optimal h
    
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        )
    
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    y_train = torch.reshape(y_train, (y_train.shape[0], 1))
    
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    # ann test error with optimal h
    y_test_est = net(X_test)
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    Error_test_ANN[k] = mse
    
    print('DONE WITH OUTER LOOP',k+1)
    
    
    