import six
import sys
sys.modules['sklearn.externals.six']=six
import numpy as np
import mlrose
import math
import matplotlib.pyplot as plt
import time

import pandas as pd






fitness = mlrose.OneMax()

random_seed = 50

lengths = [20,40,60,80,100]





rhc_fit = []
sa_fit = []
ga_fit = []
mimic_fit = []

rhc_time = []
sa_time = []
ga_time = []
mimic_time = []

for arrlength in lengths:
    problem = mlrose.DiscreteOpt(length = arrlength, fitness_fn = fitness, maximize = True)
    init_state = np.zeros(arrlength)
    #hill climbing
    start = time.time()
    best_state,rhc_fitness = mlrose.random_hill_climb(problem, max_attempts=50, restarts =20,
    		max_iters=1000,init_state=init_state,random_state=random_seed)
    end = time.time()
    
    rhc_time.append(end-start)
    
    #print("hill climbing")
    #print(best_state)
    #print(hc_fitness)
    
    
    #annealing
    schedule = mlrose.ExpDecay()
    #schedule = mlrose.GeomDecay()
    #schedule = mlrose.ArithDecay()
    start = time.time()
    best_state, sa_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts=50, 
    			max_iters=1000, init_state=init_state,random_state=random_seed)
    end = time.time()
    
    sa_time.append(end-start)
    
       
    #print("annealing")
    #print(best_state)
    #print(sa_fitness)
    
    
    #genetic algorithms
    start = time.time()
    best_state,ga_fitness = mlrose.genetic_alg(problem, pop_size =200, mutation_prob = .1,
    		 max_attempts=10, max_iters=10000,random_state=random_seed)
    end = time.time()
    ga_time.append(end-start)
    
    #print("genetic algorithms")
    #print(best_state)
    #print(ga_fitness)
    
    
    #mimic
    start = time.time()
    best_state,mimic_fitness = mlrose.mimic(problem, pop_size = 200, keep_pct = .3,
    		 max_attempts=10,max_iters = 10000, random_state=random_seed)
    end = time.time()
    mimic_time.append(end-start)
    #print("mimic")
    #print(best_state)
    #print(mimic_fitness)
    
    maximum_fitness = max(rhc_fitness, sa_fitness, ga_fitness, mimic_fitness)
    rhc_fit.append(rhc_fitness/maximum_fitness)
    sa_fit.append(sa_fitness/maximum_fitness)
    ga_fit.append(ga_fitness/maximum_fitness)
    mimic_fit.append(mimic_fitness/maximum_fitness)

fitness_df = pd.DataFrame({'size':lengths, 'rhc_fit':rhc_fit,'sa_fit':sa_fit,'ga_fit':ga_fit,'mimic_fit':mimic_fit})
fitness_df.plot(x='size',y=['rhc_fit','sa_fit','ga_fit','mimic_fit'])

time_df = pd.DataFrame({'size':lengths, 'rhc_time':rhc_time, 'sa_time':sa_time,'ga_time':ga_time, 'mimic_time':mimic_time})

time_df.plot(x = 'size', y = ['rhc_time','sa_time','ga_time','mimic_time'],logy=True)


#iterations_arr = [1000,2000,4000,6000,8000,10000]
iterations_arr = [200,400,600,800,1000]

arrlength =120
problem = mlrose.DiscreteOpt(length = arrlength, fitness_fn = fitness, maximize = True)
init_state = np.zeros(arrlength)

rhc_fit_it = []
sa_fit_it = []
ga_fit_it = []
mimic_fit_it = []


for iterations in iterations_arr:
    
    #hill climbing
    best_state,rhc_fitness = mlrose.random_hill_climb(problem, max_attempts=30, restarts =20,
    		max_iters=iterations,init_state=init_state,random_state=random_seed)
    
    
    
    #annealing
    schedule = mlrose.ExpDecay()
    #schedule = mlrose.GeomDecay()
    #schedule = mlrose.ArithDecay()
    best_state, sa_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts=100, 
    			max_iters=iterations, init_state=init_state,random_state=random_seed)

    

    
    #genetic algorithms
    best_state,ga_fitness = mlrose.genetic_alg(problem, pop_size =200, mutation_prob = .1,
    		 max_attempts=10, max_iters=iterations,random_state=random_seed)
    
    
    
    #mimic
    
    best_state,mimic_fitness = mlrose.mimic(problem, pop_size = 200, keep_pct = .3,
    		 max_attempts=10,max_iters = iterations, random_state=random_seed)
    end = time.time()
    mimic_time.append(end-start)
   
    
    maximum_fitness = max(rhc_fitness, sa_fitness, ga_fitness, mimic_fitness)
    rhc_fit_it.append(rhc_fitness/maximum_fitness)
    sa_fit_it.append(sa_fitness/maximum_fitness)
    ga_fit_it.append(ga_fitness/maximum_fitness)
    mimic_fit_it.append(mimic_fitness/maximum_fitness)

func_calls_df = pd.DataFrame({'iterations':iterations_arr, 'rhc_fit':rhc_fit_it,'sa_fit':sa_fit_it,'ga_fit':ga_fit_it,'mimic_fit':mimic_fit_it})
func_calls_df.plot(x = 'iterations', y = ['rhc_fit','sa_fit','ga_fit','mimic_fit'])
    
    
    
