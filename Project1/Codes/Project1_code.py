##################################################################
#The second and the third parts are used for some visualizaions,
#this is why we decided to include them together with the main code
#although, they run seperately
#################################################################

import pandas as pd
import numpy as np
from scipy import stats
from scipy.linalg import svd
import matplotlib.pyplot as plt

filename = 'SouthAfricanHeartDiseaseFixed.txt'
df = pd.read_csv(filename)
raw_data = df.to_numpy()

# create y and X without rownames and family history
y = np.array(raw_data[:,10],dtype=np.int32)
X = np.delete(raw_data,[0,5,10],axis=1)
X = np.array(X,dtype=np.float64)

# other variables
N = np.shape(X)[0]
M = np.shape(X)[1]
attributeNames = list(df.columns)
del attributeNames[10]
del attributeNames[5]
del attributeNames[0]
C = 2
classNames = ['healthy','sick']

# summary statistics
np.mean(X,axis=0)
np.std(X,axis=0)
X.min(axis=0)
np.percentile(X,25,axis=0)
np.percentile(X,50,axis=0)
np.percentile(X,75,axis=0)
X.max(axis=0)

# scatter plots with color-coded classes
plt.figure()
i = 4
j = 5
class_mask0 = (y == 0)
class_mask1 = (y == 1)
plt.plot(X[class_mask0,i], X[class_mask0,j], '.')
plt.plot(X[class_mask1,i], X[class_mask1,j], '.')
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])
plt.legend(classNames)

# correlation matrix
c = np.corrcoef(np.transpose(X))


    ####### PCA ########

# normalize X
Xt = X - np.ones((N,1))*X.mean(0)
Xt = Xt*(1/np.std(Xt,0))

# plot variance explained
U,S,Vh = svd(Xt,full_matrices=False)
V = Vh.T
rho = (S*S) / (S*S).sum() 
threshold = 0.9
plt.figure(figsize=(8,5))
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative'])
plt.grid()
plt.show()

# scatterplot of first 2 components
Z = Xt @ V
i = 0
j = 1
f=plt.figure(figsize=(8,5))
plt.title('PCA')
plt.plot(Z[class_mask0,i], Z[class_mask0,j], 'o', alpha=.5)
plt.plot(Z[class_mask1,i], Z[class_mask1,j], 'o', alpha=.5)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.legend(classNames)

# inspect components directly
temp = np.array([attributeNames,V[:,0].T,V[:,1].T,V[:,2].T])



######################################################################
# everything from this point forward was done by a different person
# so variables are redefined differently
######################################################################

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:01:36 2020

@author: Jens
"""

import toolbox_02450
import numpy as np
import matplotlib.pyplot as plt
import inspect
import xlrd
import pandas as pd
import os
from scipy.io import loadmat
import sklearn
from scipy.linalg import svd
import seaborn as sns

filename = 'C:/Users/lfirl/Documents/DTU/Introduction to Machine Learning/project/SouthAfricanHeartDiseaseFixed.txt'
df = pd.read_csv(filename)

raw_data = df.to_numpy()

cols = [1,2,3,4,6,7,8,9]

attributeNames = np.asarray(df.columns[cols])

X = np.array(raw_data[:, cols], dtype=np.float64)

classLabels = raw_data[:,-1] 

classNames = np.unique(classLabels)

classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N = len(y)
M = len(attributeNames)
C = len(classNames)

ones = np.ones((N,1))
mean_cols = X.mean(axis = 0)
std_cols = np.std(np.array(X, dtype=np.float64),axis = 0)


normalized_attribute_values = (X - ones*mean_cols)/(ones*std_cols)

normalized_attribute_values.shape

U,S,V = svd(Y,full_matrices=False)

rho = (S*S) / (S*S).sum() 

#plt.figure()
#plt.plot(range(1,len(rho)+1),rho,'x-')
#plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
#plt.plot([1,len(rho)],[threshold, threshold],'k--')
#plt.title('Variance explained by principal components');
#plt.xlabel('Principal component');
#plt.ylabel('Variance explained');
#plt.legend(['Individual','Cumulative','Threshold'])
#plt.grid()
#plt.show()

pos = [1,2,3,4,5,6,7,8]


#fig, (x1,x2) = plt.subplots(2,1)
#plt.figure()
#x1.violinplot(normalized_attribute_values[:,0:4], show)
#x1.legend([1,2,3,4], attributeNames[0:4])
#x2.boxplot(normalized_attribute_values[:,0:4], labels = attributeNames[0:4])

#x1.violinplot(normalized_attribute_values[:,0:4], labels = attributeNames[0:4])
#x2.boxplot(normalized_attribute_values[:,4:8], labels = attributeNames[4:8])

# violin plot: 

plt.figure(figsize=(8,8))
plt.violinplot(normalized_attribute_values,positions=(pos), vert = False,  showmedians=(True), showextrema=(True))
plt.yticks(pos,attributeNames)


plt.figure()
plt.plot(V[:,1],V[:,2],'o')

Z = Y @ V.T

#plt.figure()
#plt.plot(Z[:,1],Z[:,2],'o')

# plotting histogram distributions:

row_nr = 0    

#for i in range(8):
 #   plt.figure()
  #  plt.hist(X[:,i][y==0],bins = 15,density = True, alpha = 0.4)
   # plt.hist(X[:,i][y==1], bins = 15,density = True, alpha = 0.4)
    #plt.title("Histogram of: " + attributeNames[i])
    #plt.legend(["No-CHD", "CHD"])
    #plt.show()
    
plt.hist(X[:,i][y==0],bins = 15,density = True, alpha = 0.4)
plt.hist(X[:,i][y==1], bins = 15,density = True, alpha = 0.4)    

for i in range(8):
    plt.figure()
    sns.distplot(X[:,i][y==0],bins =9)
    sns.distplot(X[:,i][y==1],bins =9)
    plt.title("Histogram of: " + attributeNames[i])

#fig1 = plt.figure(figsize=(15,10))
#ax1 = fig1.add_subplot(241)
#sns.distplot(X[:,0][y==0],bins = 9)
#sns.distplot(X[:,0][y==1],bins = 9)
#plt.legend(["No-CHD", "CHD"])
#plt.title("Histogram of: " + attributeNames[0])

fig1 = plt.figure(figsize=(15,10))
ax1 = fig1.add_subplot(241)
plt.hist(X[:,0][y==0],bins = 9,density = True, alpha = 0.4)
plt.hist(X[:,0][y==1], bins = 9,density = True, alpha = 0.4)  
plt.legend(["Healthy", "Sick"])
plt.title("Histogram of: " + attributeNames[0])

ax2 = fig1.add_subplot(242)
plt.hist(X[:,1][y==0],bins = 9,density = True, alpha = 0.4)
plt.hist(X[:,1][y==1], bins = 9,density = True, alpha = 0.4)  
plt.title("Histogram of: " + attributeNames[1])

ax3 = fig1.add_subplot(243)
plt.hist(X[:,2][y==0],bins = 9,density = True, alpha = 0.4)
plt.hist(X[:,2][y==1], bins = 9,density = True, alpha = 0.4)  
plt.title("Histogram of: " + attributeNames[2])

ax4 = fig1.add_subplot(244)
plt.hist(X[:,3][y==0],bins = 9,density = True, alpha = 0.4)
plt.hist(X[:,3][y==1], bins = 9,density = True, alpha = 0.4)  
plt.title("Histogram of: " + attributeNames[3])

ax5 = fig1.add_subplot(245)
plt.hist(X[:,4][y==0],bins = 9,density = True, alpha = 0.4)
plt.hist(X[:,4][y==1], bins = 9,density = True, alpha = 0.4)  
plt.title("Histogram of: " + attributeNames[4])

ax6 = fig1.add_subplot(246)
plt.hist(X[:,5][y==0],bins = 9,density = True, alpha = 0.4)
plt.hist(X[:,5][y==1], bins = 9,density = True, alpha = 0.4)  
plt.title("Histogram of: " + attributeNames[5])

ax7 = fig1.add_subplot(247)
plt.hist(X[:,6][y==0],bins = 9,density = True, alpha = 0.4)
plt.hist(X[:,6][y==1], bins = 9,density = True, alpha = 0.4)  
plt.title("Histogram of: " + attributeNames[6])

ax8 = fig1.add_subplot(248)
plt.hist(X[:,7][y==0],bins = 9,density = True, alpha = 0.4)
plt.hist(X[:,7][y==1], bins = 9,density = True, alpha = 0.4)  
plt.title("Histogram of: " + attributeNames[7])

######################################################################
# from this point forward the code was done by the third person
# variables are redefined differently
######################################################################
'''
@author: Angelos
'''

import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd
import seaborn as sns
doc = xlrd.open_workbook('SouthAfricanHeartDisease.xlsx').sheet_by_index(0)
#the history column has been moved into last position

attributeNames = doc.row_values(rowx=0, start_colx=1, end_colx=11) 
#all the column Names

FamHistory = doc.col_values(10,1,463) 
HistoryType = sorted(set(FamHistory))
HistoryDict = dict(zip(HistoryType,range(len(HistoryType))))
#Transformed Family History into integers

y = np.array([HistoryDict[value] for value in FamHistory])
#the family history column with integers instead of strings

X = np.empty((462,9))
for i in range(0,9):
    X[:,i] = np.array(doc.col_values(i+1,1,463)).T
    #the whole table except the first and last column
    
N = len(y)
M = len(attributeNames)
C = len(HistoryType)
#we will need those

X_r=X.copy()
X_r = np.delete(X_r,[8],axis=1)
attributeNames_r=attributeNames.copy()
del attributeNames_r[9]
del attributeNames_r[8]
df = pd.DataFrame(X_r,columns=[attributeNames_r])
corrMat=df.corr()

X_c = X.copy()
y_c = y.copy()
attributeNames_c = attributeNames.copy()
#copies of our variables

alldata=np.concatenate((X_c, np.expand_dims(y_c,axis=1)), axis=1)
#complete matrix with all the data

alldataframe=pd.DataFrame(alldata)
alldataframe.describe()
print(alldataframe.describe())
#transformed alldata array into pandas dataframe
#to calculate the basic statistics of the attributes



chdlabels=doc.col_values(9,1,463)
chdTypeflo = sorted(set(chdlabels))
chdType=[int(i) for i in chdTypeflo]
chdType[1]='sick'
chdType[0]='healthy'


#renamed all 0s & 1s to healthy and sick 
#(it should be done with a dict though)




i =7 ; j =1 ;
color = ['g','r']
plt.title('Heart disease incidents')
for n in range(len(chdType)):
    idx = y_c == n
    plt.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[n], 
                s=20, alpha=0.5,
                label=chdType[n])
plt.legend(chdType)
m, b = np.polyfit(X_c[idx, i], X_c[idx, j], 1)  #for linear regression
plt.plot(X_c[idx, i], m*X_c[idx, i] + b)

plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.show()
#scatter plot with coloured groups for sick and healthy
#we only change i and j (numbers 0-7)


Y = X - np.ones((N,1))*X.mean(axis=0)
Y = Y*(1/np.std(Y,0))
U,S,V = svd(Y,full_matrices=False)
rho = (S*S) / (S*S).sum() 
threshold = 0.9
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()




