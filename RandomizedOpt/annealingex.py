import six
import sys
sys.modules['sklearn.externals.six']=six
import numpy as np
import mlrose
import math

def binary_conv(binaryarray):
	total = 0
	arraylen = len(binaryarray)
	for i in range(0,arraylen):
		if(binaryarray[arraylen-i-1]!=1 and binaryarray[arraylen-i-1] !=0):
			print("array not binary")
			return
		total = total + (2**i)*binaryarray[arraylen-i-1]

	#print("total is:" + str(total))
	return total

def multipleHills(state):
	x = binary_conv(state)

	fitness = 10*math.sin((math.pi/8)*x)*math.exp(-(((x/8)-64)**2)/150)+20
	return fitness

def multipleHillsLarge(state): 
    x = binary_conv(state)

	
    fitness = 10*math.sin((math.pi/8)*x)*math.exp(-(((x/40)-1000)**2)/100000)+20
    return fitness

    
#setup
fitness_cust = mlrose.CustomFitness(multipleHills)
problem = mlrose.DiscreteOpt(length = 10, fitness_fn = fitness_cust, maximize = True, max_val = 2)
init_state = np.array([0,0,0,0,0,0,0,0,0,0])


#hill climbing
best_state,best_fitness = mlrose.random_hill_climb(problem, max_attempts=10,
	max_iters=300,init_state=init_state,random_state=1)
print("hill climbing")
print(best_state)
print(best_fitness)
print(str(binary_conv(best_state)))

#annealing
schedule = mlrose.ExpDecay()
best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts=10, 
	max_iters=300, init_state=init_state,random_state=1)
print("annealing")
print(best_state)
print(best_fitness)
print(str(binary_conv(best_state)))

#genetic algorithm

best_state,best_fitness = mlrose.genetic_alg(problem, pop_size = 100, mutation_prob = .1,
 max_attempts=10, max_iters=1000,random_state=1)
print("genetic alg")
print(best_state)
print(best_fitness)
print(str(binary_conv(best_state)))

#mimic

best_state,best_fitness = mlrose.mimic(problem, pop_size = 100, keep_pct = .2,
 max_attempts=10, max_iters=1000,random_state=1)
print("mimic")
print(best_state)
print(best_fitness)
print(str(binary_conv(best_state)))



#large function
fitness_cust = mlrose.CustomFitness(multipleHillsLarge)
problem = mlrose.DiscreteOpt(length = 17, fitness_fn = fitness_cust, maximize = True, max_val = 2)
init_state = np.zeros(17)


#hill climbing
best_state,best_fitness = mlrose.random_hill_climb(problem, max_attempts=10,
	max_iters=15000,init_state=init_state,random_state=1)
print("hill climbing")
print(best_state)
print(best_fitness)
print(str(binary_conv(best_state)))

#annealing
schedule = mlrose.ExpDecay()
best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts=10, 
	max_iters=15000, init_state=init_state,random_state=1)
print("annealing")
print(best_state)
print(best_fitness)
print(str(binary_conv(best_state)))

#genetic algorithm

best_state,best_fitness = mlrose.genetic_alg(problem, pop_size = 100, mutation_prob = .1,
 max_attempts=10, max_iters=15000,random_state=1)
print("genetic alg")
print(best_state)
print(best_fitness)
print(str(binary_conv(best_state)))

#mimic

best_state,best_fitness = mlrose.mimic(problem, pop_size = 10000, keep_pct = .2,
 max_attempts=10, max_iters=10000,random_state=1)
print("mimic")
print(best_state)
print(best_fitness)
print(str(binary_conv(best_state)))
  