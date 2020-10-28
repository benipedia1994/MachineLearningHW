import six
import sys
sys.modules['sklearn.externals.six']=six
import numpy as np
import mlrose
import math

edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
fitness = mlrose.MaxKColor(edges)
#state = np.array([0, 1, 0, 1, 1])
#print(fitness.evaluate(state))
#state = np.array([1,1,1,1,1])
#print(fitness.evaluate(state))
#setup
problem = mlrose.DiscreteOpt(length = 5, fitness_fn = fitness, maximize = True, max_val = 2)
init_state = np.array([1,0,0,1,0])
best_state,best_fitness = mlrose.random_hill_climb(problem, max_attempts=5,
	max_iters=16,init_state=init_state,random_state=1)
print("hill climbing")
print(best_state)
print(best_fitness)


#annealing
schedule = mlrose.ExpDecay()
best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts=5, 
	max_iters=16, init_state=init_state,random_state=1)
print("annealing")
print(best_state)
print(best_fitness)


#genetic algorithm

best_state,best_fitness = mlrose.genetic_alg(problem, pop_size = 1, mutation_prob = .1,
 max_attempts=5, max_iters=32,random_state=1)
print("genetic alg")
print(best_state)
print(best_fitness)


#mimic

best_state,best_fitness = mlrose.mimic(problem, pop_size = 100, keep_pct = .2,
 max_attempts=5, max_iters=16,random_state=1)
print("mimic")
print(best_state)
print(best_fitness)

  