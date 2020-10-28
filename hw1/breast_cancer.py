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
from sklearn.model_selection import train_test_split
import time


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


def get_acc_auc_test_train(X,Y,model,iterNo=8,test_percent=0.80):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    
    Yarray = np.ravel(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Yarray, test_size=0.20, random_state = 0)
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)

    acc= accuracy_score(Y_test,prediction)
    auc= roc_auc_score(Y_test,prediction)
    
    
    acc= accuracy_score(Y_test,prediction)
    auc= roc_auc_score(Y_test,prediction)
    precision = precision_score(Y_test, prediction)
    recall = recall_score(Y_test,prediction)
    f1 = f1_score(Y_test,prediction)
    
    
    return acc, auc, precision, recall, f1



def seperate_variables(df):
    X = df.loc[:,'radius_mean':'fractal_dimension_worst']

    Y = df[['diagnosis']]
    
    return X,Y

def test_model(X,Y,model):
    start = time.time()
    acc_k, auc_k= get_acc_auc_kfold(X,Y,model)
    print(("Average Accuracy in KFold CV: "+str(acc_k)))
    print(("Average AUC in KFold CV: "+str(auc_k)))
    
    acc_r,auc_r = get_acc_auc_randomisedCV(X,Y,model)
    print(("Average Accuracy in Randomised CV: "+str(acc_r)))
    print(("Average AUC in Randomised CV: "+str(auc_r)))
    print('\n')
    
    acc_l, auc_l, precision, recall, f1 = get_acc_auc_test_train(X, Y, model)
    print(("Accuracy in Test-Train: "+str(acc_l)))
    print(("AUC in Test-Train"+str(auc_l)))
    print(("Precision in Test-Train") +str(precision))
    print(("Recall in Test-Train") +str(recall))
    print(("f1 in Test-Train")+str(f1))
    end = time.time()
    print("Time to run tests: " + str(end - start))
    
    return acc_k,auc_k, acc_r,auc_r, acc_l,auc_l
    



def main():
    df = pd.read_csv("./data/breast-cancer/data.csv")
    
    df['diagnosis']=pd.Categorical(df['diagnosis'])
    df['diagnosis'] = df['diagnosis'].cat.codes
    
    
    
    normallized = df.loc[:,'diagnosis':]/df.loc[:,'diagnosis':].max()
    testResults = pd.DataFrame(columns = ['acc_kfolds','auc_kfolds','acc_rand','auc_rand','acc_test','auc_test'])
    
    X, Y = seperate_variables(normallized)
    print("percentage of positive diagnosis:" + str(((Y.loc[Y['diagnosis']==1]).shape[0])/Y.shape[0]))
    
    #model = KNeighborsClassifier(n_neighbors=3)
    model = SVC(C = 8)
    print("Testing SVM rbf")
    result = test_model(X,Y,model)
    testResults.loc['SVMrbf']=result
    print("\n")
    
    model = SVC(C = 15, kernel = 'linear')
    print("Testing SVM linear")
    result = test_model(X,Y,model)
    testResults.loc['SVMlin']=result
    print("\n")
    
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5))
    print("AdaBoost Tree")
    result = test_model(X,Y,model)
    testResults.loc['AdaBoost']=result
    print("\n")
    
    model = RandomForestClassifier(max_depth=5)
    print("Testing Random Forest Tree")
    result = test_model(X,Y,model)
    testResults.loc['RandomForest']=result
    print("\n")
    
    model = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=10,gamma=0.51)
    print("Testing XGB Tree")
    result = test_model(X,Y,model)
    testResults.loc['XGBTree']=result
    print("\n")
    
    
    #model = MLPClassifier(solver = 'lbfgs',learning_rate='adaptive',alpha = .005)
    model = MLPClassifier(solver = 'lbfgs', learning_rate='adaptive')
    print("Testing Neural Network")
    result = test_model(X,Y,model)
    testResults.loc['NeuralNW']=result
    print("\n")
    
    
    model = KNeighborsClassifier(n_neighbors=5)
    print("Testing K nearest Neighbors")
    result = test_model(X,Y,model)
    testResults.loc['KNN']=result
    print("\n")
    
    model = LogisticRegression(C=50)
    print("Testing Logistic Regression")
    result = test_model(X,Y,model)
    testResults.loc['LogReg']=result
    print("\n")
    
    testResults.plot.bar(ylim=(.80,1),figsize=(24,24),fontsize=26)
    
    
    
    
    
    
    
    
if __name__ == "__main__":
	main()
    