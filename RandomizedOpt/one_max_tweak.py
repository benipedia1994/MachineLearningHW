import six
import sys
sys.modules['sklearn.externals.six']=six
import numpy as np
import mlrose
import math
import matplotlib.pyplot as plt
import time

import pandas as pd





arrlength = 100
fitness = mlrose.OneMax()

random_seed = 50



problem = mlrose.DiscreteOpt(length = arrlength, fitness_fn = fitness, maximize = True)
init_state = np.zeros(arrlength)


#hill climbing
start = time.time()
best_state,hc_fitness = mlrose.random_hill_climb(problem, max_attempts=30, restarts =20,
		max_iters=1000,init_state=init_state,random_state=random_seed)
end = time.time()

print("hill climbing")
#print(best_state)
print(hc_fitness)


#annealing
schedule = mlrose.ExpDecay()
#schedule = mlrose.GeomDecay()
#schedule = mlrose.ArithDecay()
start = time.time()
best_state, sa_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts=300, 
			max_iters=3000, init_state=init_state,random_state=random_seed)
end = time.time()

   
print("annealing")
#print(best_state)
print(sa_fitness)


#genetic algorithms
start = time.time()
best_state,ga_fitness = mlrose.genetic_alg(problem, pop_size =200, mutation_prob = .1,
		 max_attempts=10, max_iters=10000,random_state=random_seed)
end = time.time()

print("genetic algorithms")
#print(best_state)
print(ga_fitness)


#mimic
start = time.time()
best_state,mimic_fitness = mlrose.mimic(problem, pop_size = 200, keep_pct = .3,
		 max_attempts=10,max_iters = 10000, random_state=random_seed)
print("mimic")
#print(best_state)
print(mimic_fitness)

