import six
import sys
sys.modules['sklearn.externals.six']=six
import numpy as np
import mlrose
import math
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_csv("./data/heart-disease/Heart_Disease_Prediction.csv")
df['Heart Disease']=pd.Categorical(df['Heart Disease'])
df['Heart Disease'] = df['Heart Disease'].cat.codes
normallized = df/df.max()


datamat = normallized.to_numpy()
#‘identity’, ‘relu’, ‘sigmoid’ or ‘tanh’.


avg_train = []
avg_test = []
avg_loss = []
for i in range(0,5):
    #random_hill_climb
    X_train, X_test, y_train, y_test = train_test_split(datamat[:,0:13], datamat[:,13], \
                                                        test_size = 0.2, random_state = i)
        
    hc_model = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                     algorithm = 'random_hill_climb', max_iters = 3000, \
                                     bias = True, is_classifier = True, learning_rate = 0.1, \
                                     early_stopping = True, clip_max = 100,restarts = 20, max_attempts = 100, \
                                     random_state = i)
    
    hc_model.fit(X_train, y_train)
    
    
    #print("it done")
    # Predict labels for train set and assess accuracy
    y_train_pred = hc_model.predict(X_train)
    
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    avg_train.append(y_train_accuracy)
    
    #print("hill climb")
    #print(y_train_accuracy)
    
    
    # Predict labels for test set and assess accuracy
    y_test_pred = hc_model.predict(X_test)
    
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    avg_test.append(y_test_accuracy)
    avg_loss.append(hc_model.loss)
    
    #print(y_test_accuracy)

print("hill climbing")
print("Train" + str(np.mean(avg_train)))
print("Test" + str(np.mean(avg_test)))
print("Loss:" + str(np.mean(avg_loss)))


#simulated annealing
avg_train = []
avg_test = []
avg_loss = []
for i in range(0,5):
    X_train, X_test, y_train, y_test = train_test_split(datamat[:,0:13], datamat[:,13], \
                                                        test_size = 0.2, random_state = i)
    
    #schedule = mlrose.ExpDecay()
    schedule = mlrose.GeomDecay()
    #schedule = mlrose.ArithDecay()    
    hc_model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                     algorithm = 'simulated_annealing', max_iters = 5000, \
                                     bias = True, is_classifier = True, learning_rate = 0.1, \
                                     early_stopping = False, clip_max = 5, max_attempts = 100, \
                                     random_state = i*5,schedule = schedule)
    
    hc_model.fit(X_train, y_train)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = hc_model.predict(X_train)
    
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    
    #print("simulated annealing")
    #print(y_train_accuracy)
    avg_train.append(y_train_accuracy)
    
    # Predict labels for test set and assess accuracy
    y_test_pred = hc_model.predict(X_test)
    
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    avg_test.append(y_test_accuracy)
    avg_loss.append(hc_model.loss)
    #print(y_test_accuracy)

print("Simulated Annealing")
print("Train" + str(np.mean(avg_train)))
print("Test" + str(np.mean(avg_test)))
print("Loss:" + str(np.mean(avg_loss)))

#genetic algorithm
avg_train = []
avg_test = []
avg_loss = []
for i in range(0,5):
    X_train, X_test, y_train, y_test = train_test_split(datamat[:,0:13], datamat[:,13], \
                                                        test_size = 0.2, random_state = i)
    
    hc_model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                     algorithm = 'genetic_alg', max_iters = 10000, \
                                     bias = True, is_classifier = True, learning_rate = 0.0001, \
                                     early_stopping = True, clip_max = 5, max_attempts = 200, \
                                     random_state =i*10,pop_size = 200,mutation_prob = .4)
    
    hc_model.fit(X_train, y_train)
    
    # Predict labels for train set and assess accuracy
    y_train_pred = hc_model.predict(X_train)
    
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    avg_train.append(y_train_accuracy)
    
    
    #print("genetic")
    #print(y_train_accuracy)
    
    # Predict labels for test set and assess accuracy
    y_test_pred = hc_model.predict(X_test)
    
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    avg_test.append(y_test_accuracy)
    avg_loss.append(hc_model.loss)
    #print(y_test_accuracy)
print("Genetic Algorithm")
print("Train" + str(np.mean(avg_train)))
print("Test" + str(np.mean(avg_test)))
print("Loss:" + str(np.mean(avg_loss)))

avg_train = []
avg_test = []
avg_loss = []
for i in range(0,5):
#gradient descent
    X_train, X_test, y_train, y_test = train_test_split(datamat[:,0:13], datamat[:,13], \
                                                        test_size = 0.2, random_state = i)
        
    hc_model = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                     algorithm = 'gradient_descent', max_iters = 1000, \
                                     bias = True, is_classifier = True, learning_rate = 0.001, \
                                     early_stopping = True, clip_max = 5, max_attempts = 100, \
                                     random_state = i)
    
    hc_model.fit(X_train, y_train)
    
    
    
    # Predict labels for train set and assess accuracy
    y_train_pred = hc_model.predict(X_train)
    
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    avg_train.append(y_train_accuracy)
    #print("gradient descent")
    #print(y_train_accuracy)
    
    
    # Predict labels for test set and assess accuracy
    y_test_pred = hc_model.predict(X_test)
    
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    avg_test.append(y_test_accuracy)
    avg_loss.append(hc_model.loss)
    #print(y_test_accuracy)
print("gradient descent")
print("Train" + str(np.mean(avg_train)))
print("Test" + str(np.mean(avg_test)))
print("Loss:" + str(np.mean(avg_loss)))