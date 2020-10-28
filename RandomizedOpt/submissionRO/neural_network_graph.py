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
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv("./data/heart-disease/Heart_Disease_Prediction.csv")
df['Heart Disease']=pd.Categorical(df['Heart Disease'])
df['Heart Disease'] = df['Heart Disease'].cat.codes
normallized = df/df.max()

datamat = normallized.to_numpy()

#random_hill_climb
X_train, X_test, y_train, y_test = train_test_split(datamat[:,0:13], datamat[:,13], \
                                                    test_size = 0.2, random_state = 3)
    
hc_model = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 3000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = True, clip_max = 100,restarts = 20, max_attempts = 100, \
                                 random_state = 3, curve = True)

hc_model.fit(X_train, y_train)

plt.plot(hc_model.fitness_curve)
plt.xlabel('iterations')
plt.ylabel('RHC fitness')
plt.show()



# Predict labels for train set and assess accuracy
y_train_pred = hc_model.predict(X_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print("hill climb")
print(y_train_accuracy)


# Predict labels for test set and assess accuracy
y_test_pred = hc_model.predict(X_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print(y_test_accuracy)
print("Loss:" + str(hc_model.loss))

#simulated annealing
X_train, X_test, y_train, y_test = train_test_split(datamat[:,0:13], datamat[:,13], \
                                                        test_size = 0.2, random_state = 3)
    
#schedule = mlrose.ExpDecay()
schedule = mlrose.GeomDecay()
#schedule = mlrose.ArithDecay()    
hc_model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 5000, \
                                 bias = True, is_classifier = True, learning_rate = 0.1, \
                                 early_stopping = False, clip_max = 5, max_attempts = 100, \
                                 random_state = 3,schedule = schedule,curve = True)

hc_model.fit(X_train, y_train)
plt.plot(hc_model.fitness_curve)
plt.xlabel('iterations')
plt.ylabel('SA fitness')
plt.show()

# Predict labels for train set and assess accuracy
y_train_pred = hc_model.predict(X_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print("simulated annealing")
print(y_train_accuracy)
#avg_train.append(y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = hc_model.predict(X_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)
#avg_test.append(y_test_accuracy)
print(y_test_accuracy)
print("Loss:" + str(hc_model.loss))


#genetic
X_train, X_test, y_train, y_test = train_test_split(datamat[:,0:13], datamat[:,13], \
                                                    test_size = 0.2, random_state = 3)

hc_model = mlrose.NeuralNetwork(hidden_nodes = [6], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 10000, \
                                 bias = True, is_classifier = True, learning_rate = 0.01, \
                                 early_stopping = True, clip_max = 5, max_attempts = 200, \
                                 random_state =10,pop_size = 100,mutation_prob = .5,curve = True)

hc_model.fit(X_train, y_train)
plt.plot(hc_model.fitness_curve)
plt.ylabel('GA fitness')
plt.xlabel('iterations')
plt.show()

# Predict labels for train set and assess accuracy
y_train_pred = hc_model.predict(X_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)



print("genetic")
print(y_train_accuracy)


# Predict labels for test set and assess accuracy
y_test_pred = hc_model.predict(X_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)


print(y_test_accuracy)
print("Loss:" + str(hc_model.loss))


#gradient descent
X_train, X_test, y_train, y_test = train_test_split(datamat[:,0:13], datamat[:,13], \
                                                    test_size = 0.2, random_state = 3)
    
hc_model = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.001, \
                                 early_stopping = True, clip_max = 5, max_attempts = 100, \
                                 random_state = 10, curve = True)

hc_model.fit(X_train, y_train)

plt.plot(hc_model.fitness_curve)
plt.xlabel('iterations')
plt.ylabel('GD fitness')
plt.show()



# Predict labels for train set and assess accuracy
y_train_pred = hc_model.predict(X_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print("gradient descent")
print(y_train_accuracy)


# Predict labels for test set and assess accuracy
y_test_pred = hc_model.predict(X_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print(y_test_accuracy)
print("Loss:" + str(hc_model.loss))
