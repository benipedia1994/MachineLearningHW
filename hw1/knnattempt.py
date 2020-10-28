import pandas as pd
from numpy import mean
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import chisquare
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_acc_auc_kfold(X,Y,model,k=3):
    accuracy = []
    auc = []
    #model = tree.DecisionTreeClassifier(max_depth=5)
    #model = RandomForestClassifier(max_depth=5)
    #model = LogisticRegression()
    #model = KNeighborsClassifier(n_neighbors=3)
    kf = KFold(n_splits=k)
    Yarray = np.ravel(Y)
    
    for train_index, test_index in kf.split(X):
        
        model.fit(X.iloc[train_index],Yarray[train_index])
        predict = model.predict(X.iloc[test_index])
        accuracy.append(accuracy_score(Yarray[test_index],predict))
        auc.append(roc_auc_score(Yarray[test_index],predict))
        
    acc_avg=mean(accuracy)
    auc_avg=mean(auc)
    
    return acc_avg, auc_avg

def get_acc_auc_randomisedCV(X,Y,model,iterNo=3,test_percent=0.35):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    
    accuracy = []
    auc = []
    
    Yarray = np.ravel(Y)
    #model = KNeighborsClassifier(n_neighbors=3)
    rs = ShuffleSplit(n_splits=iterNo, test_size=test_percent)
    for train_index, test_index in rs.split(X):
        model.fit(X.iloc[train_index],Yarray[train_index])
        predict = model.predict(X.iloc[test_index])
        accuracy.append(accuracy_score(Yarray[test_index],predict))
        auc.append(roc_auc_score(Yarray[test_index],predict))
        #result = chisquare(predict,Yarray[test_index])
        
    acc_avg=mean(accuracy)
    auc_avg=mean(auc)
    
    
    return acc_avg, auc_avg


def get_acc_auc_randomised_large(X,Y,model,iterNo=8,test_percent=0.80):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    
    accuracy = []
    auc = []
    
    Yarray = np.ravel(Y)
    #model = KNeighborsClassifier(n_neighbors=3)
    rs = ShuffleSplit(n_splits=iterNo, test_size=test_percent)
    for train_index, test_index in rs.split(X):
        model.fit(X.iloc[train_index],Yarray[train_index])
        predict = model.predict(X.iloc[test_index])
        accuracy.append(accuracy_score(Yarray[test_index],predict))
        auc.append(roc_auc_score(Yarray[test_index],predict))
        #result = chisquare(predict,Yarray[test_index])
        
    acc_avg=mean(accuracy)
    auc_avg=mean(auc)
    
    
    return acc_avg, auc_avg



def seperate_variables(df):
    X = df.loc[:,'Age':'Thallium']

    Y = df[['Heart Disease']]
    
    return X,Y

def test_model(X,Y,model):
    acc_k, auc_k= get_acc_auc_kfold(X,Y,model)
    print(("Average Accuracy in KFold CV: "+str(acc_k)))
    print(("Average AUC in KFold CV: "+str(auc_k)))
    
    acc_r,auc_r = get_acc_auc_randomisedCV(X,Y,model)
    print(("Average Accuracy in Randomised CV: "+str(acc_r)))
    print(("Average AUC in Randomised CV: "+str(auc_r)))
    
    acc_l, auc_l = get_acc_auc_randomised_large(X, Y, model)
    print(("Average Accuracy in Large Randomised CV: "+str(acc_l)))
    print(("Average AUC in Large Randomised CV: "+str(auc_l)))
    



def main():
    df = pd.read_csv("./data/heart-disease/Heart_Disease_Prediction.csv")
    
    df['Heart Disease']=pd.Categorical(df['Heart Disease'])
    df['Heart Disease'] = df['Heart Disease'].cat.codes
    normallized = df/df.max()
    
    
    X, Y = seperate_variables(normallized)
    #model = KNeighborsClassifier(n_neighbors=3)
    model = SVC(C = .5)
    #model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=6))
    #model = RandomForestClassifier(max_depth=5)
    #model = MLPClassifier()
    #model = XGBClassifier()
    test_model(X,Y,model)
    
    
    
    
    
if __name__ == "__main__":
	main()
    
    