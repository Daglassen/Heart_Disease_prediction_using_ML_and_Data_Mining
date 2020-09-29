import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd
doc = xlrd.open_workbook('Data\SouthAfricanHeartDiseaseFixed.xlsx').sheet_by_index(0)
#the history column has been moved into last position

attributeNames = doc.row_values(rowx=0, start_colx=1, end_colx=11) 
#all the column Names

FamHistory = doc.col_values(10,1,463) 
HistoryType = sorted(set(FamHistory))
HistoryDict = dict(zip(HistoryType,range(len(HistoryType))))
#Transformed Family History into integers

y = np.array([HistoryDict[value] for value in FamHistory])
#the array with integers instead of strings

X = np.empty((462,9))
for i in range(0,9):
    X[:,i] = np.array(doc.col_values(i+1,1,463)).T
    #the whole table except the first and last column
    
N = len(y)
M = len(attributeNames)
C = len(HistoryType)
#we will need those

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

y_r=alldata[:, 2]
X_r=alldata[:,[0,1,3,4,5,6,7,8,9]]
#created a matrix (y_r) with 1 attribute (chd) 
#and a matrix (X_r) with the rest

i = 3; j = 5;
color = ['g','r']
plt.title('Heart disease insidents')
for c in range(len(chdType)):
    idx = y_c == c
    plt.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[c], 
                s=20, alpha=0.5,
                label=chdType[c])
plt.legend(chdType)
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.show()
#scatter plot with coloured groups for sick and healthy
#we only change i and j (numbers 0-8)


Y = X - np.ones((N,1))*X.mean(axis=0)
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
#PCA Maybe????



