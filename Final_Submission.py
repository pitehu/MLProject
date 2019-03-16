# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:56:07 2018

@author: superhhu
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

#from sklearn.preprocessing import SimpleImputer
none_missing=pd.read_csv(r'C:\sTUDY\CS4375\Project\train.nmv.txt',header=None,delimiter=r"\s+")
train_nm=none_missing
#test_nm=none_missing[8000:]
train_x_nm=train_nm.drop(columns=[205])
train_y_nm=train_nm[205]
train_missing=pd.read_csv(r'C:\sTUDY\CS4375\Project\train.mv.txt',header=None,delimiter=r"\s+")
train_x_m=train_missing.drop(columns=[205])
train_y_m=train_missing[205]

prelim=pd.read_csv(r'C:\sTUDY\CS4375\Project\prelim-nmv-noclass.txt',header=None,delimiter=r"\s+")
prelim=prelim.drop(columns=[205])
prelim_y_m=pd.read_csv(r'C:\sTUDY\CS4375\Project\prelim-gold.txt',header=None,delimiter=r"\s+")
#test_x_nm=test_nm.drop(columns=[205])
#test_y_nm=test_nm[205]
all_train=pd.concat([train_x_nm, prelim])
all_test=pd.Series(pd.concat([train_y_nm,prelim_y_m])[0])

test_x=pd.read_csv(r'C:\sTUDY\CS4375\Project\final-nmv-noclass.txt',header=None,delimiter=r"\s+")
test_x=test_x.drop(columns=[205])


estimator=lgb.LGBMClassifier(num_leaves=86)
estimator.fit(all_train,all_test)
y_predict=estimator.predict(test_x)
np.savetxt(r'C:\sTUDY\CS4375\Project\final_predict.txt',y_predict,'%d')


