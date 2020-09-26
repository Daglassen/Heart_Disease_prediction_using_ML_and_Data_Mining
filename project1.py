
import numpy as np
import xlrd
doc = xlrd.open_workbook('Data\SouthAfricanHeartDiseaseFixed.xlsx').sheet_by_index(0)
#the sheet doesn't contain the first column and 
#the history has been moved into last position
attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=10) 
#all the column Names
FamHistory = doc.col_values(9,1,463) 
HistoryType = sorted(set(FamHistory))
HistoryDict = dict(zip(HistoryType,range(len(HistoryType))))
#Transformed Family History into integers
y = np.array([HistoryDict[value] for value in FamHistory])
#the array with integers instead of strings
X = np.empty((462,9))
for i in range(9):
    X[:,i] = np.array(doc.col_values(i,1,463)).T
    #the whole table exept the first last column
N = len(y)
M = len(attributeNames)
C = len(HistoryType)
#we will need those