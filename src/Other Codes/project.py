import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

train_data=pd.read_csv('./all/train_data.csv',sep=',')
train_label=pd.read_csv('./all/train_label.csv',sep=',')
test_data=pd.read_csv('./all/test_data.csv',sep=',')

train_X11=train_data[train_data.columns[3]]
train_X12=train_data[train_data.columns[4]]
train_X13=pd.concat([train_X11, train_X12], axis=1)
#print(train_X13.isnull())
train_X14=train_X13.fillna(axis=1,method='ffill')
train_X1=train_X14[train_X14.columns[1]]

#train_X1=train_data[train_data.columns[4]]


train_X22=train_data[train_data.columns[6]]
train_X3=train_data[train_data.columns[7]]
train_y=train_label[train_label.columns[1:]]

#train_X1=train_X1.dropna()




train_X1=pd.get_dummies(train_X1)
train_X2=(train_X22 - train_X22.min()) / (train_X22.max() - train_X22.min())
train_X3=pd.get_dummies(train_X3)
train_X = pd.concat([train_X1, train_X2,train_X3], axis=1)
train_y=np.ravel(train_y)
#print(train_y)

test_X11=test_data[test_data.columns[3]]
test_X12=test_data[test_data.columns[4]]
test_X13=pd.concat([test_X11, test_X12], axis=1)
#print(train_X13.isnull())
test_X14=test_X13.fillna(axis=1,method='ffill')
test_X1=test_X14[test_X14.columns[1]]



#test_X1=test_data[test_data.columns[4]]
test_X22=test_data[test_data.columns[6]]
test_X3=test_data[test_data.columns[7]]
test_X1=pd.get_dummies(test_X1)
test_X2=(test_X22 - test_X22.min()) / (test_X22.max() - test_X22.min())
test_X3=pd.get_dummies(test_X3)
test_X4=test_data[test_data.columns[0]]
test_X = pd.concat([test_X1, test_X2,test_X3], axis=1)

rf = RandomForestRegressor(n_estimators = 100, oob_score =True, n_jobs = -1,random_state =10,
                                max_features = "auto", min_samples_leaf = 50)
rf.fit(train_X,train_y)
result=rf.predict(test_X)
result1=(result - result .min()) / (result .max() - result .min())
column_result = pd.Series(result1, name='score')

csv=pd.concat([test_X4,column_result],axis=1)

csv.to_csv("submission.csv",sep=',',index=0,float_format='%.2f')
