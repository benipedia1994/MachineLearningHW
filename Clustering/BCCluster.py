from numpy import mean

import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.cluster import KMeans



def Main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    df = pd.read_csv("./Data/BreastCancer/data.csv")
    
    df['diagnosis']=pd.Categorical(df['diagnosis'])
    df['diagnosis'] = df['diagnosis'].cat.codes
    
    
    
    normallized = df.loc[:,'diagnosis':]/df.loc[:,'diagnosis':].max()
    
    kmeans = KMeans(n_clusters=2,random_state=0).fit(normallized)
    
    