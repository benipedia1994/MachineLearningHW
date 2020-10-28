import six
import sys
sys.modules['sklearn.externals.six']=six
import numpy as np
import mlrose
import math
import matplotlib.pyplot as plt
import time

import pandas as pd


random_seed = 50
lengths = [5,20,50,100]
rhc_fit = []
sa_fit = []
ga_fit = []
mimic_fit = []

rhc_time = []
sa_time = []
ga_time = []
mimic_time = []

for arrlength in lengths:
    rhc_it_fit = []
    sa_it_fit = []
    ga_it_fit = []
    mimic_it_fit = []
    
    rhc_it_time = []
    sa_it_time = []
    ga_it_time = []
    mimic_it_time = []
    
    for i in range(0,5):

		#arrlength = 100
        weights = np.random.randint(1,20,arrlength)
        values = np.random.randint(1,20,arrlength)
		#weights = [10,5,2,8,15]
		#values = [1,2,3,4,5]
        max_weight_pct = .6
        fitness = mlrose.Knapsack(weights,values, max_weight_pct)
        problem = mlrose.DiscreteOpt(length = arrlength, fitness_fn = fitness, maximize = True)
        init_state = np.zeros(arrlength)

        start = time.time()
        best_state,hc_fitness = mlrose.random_hill_climb(problem, max_attempts=50, restarts = 20,
			max_iters=10000,init_state=init_state,random_state=random_seed)
        end = time.time()
        rhc_it_time.append(end-start)

        '''
		print("hill climbing")
		print(best_state)
		print(best_fitness)
		'''


		#annealing
        schedule = mlrose.ExpDecay()
        start = time.time()
        best_state, sa_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts=50, 
			max_iters=10000, init_state=init_state,random_state=random_seed)
        end = time.time()
        sa_it_time.append(end-start)
        

        '''
		print("annealing")
		print(best_state)
		print(best_fitness)
		'''


		#genetic algorithm
        start = time.time()
        best_state,ga_fitness = mlrose.genetic_alg(problem, pop_size =100, mutation_prob = .1,
		 max_attempts=100, max_iters=10000,random_state=random_seed)
        end = time.time()
        ga_it_time.append(end-start)
        
        '''
        print("genetic alg")
        nt(best_state)
        nt(best_fitness)
        '''

		#mimic
        start = time.time()
        best_state,mimic_fitness = mlrose.mimic(problem, pop_size = 200, keep_pct = .2,
		 max_attempts=10,random_state=random_seed)
        '''
        best_state,mimic_fitness = mlrose.mimic(problem, pop_size = 200, keep_pct = .2,
		 max_attempts=10, max_iters=50,random_state=random_seed)
        '''
        end = time.time()
        mimic_it_time.append(end-start)
        '''
        print("mimic")
        print(best_state)
		print(best_fitness)
		'''
        maxfitness = max(hc_fitness,sa_fitness,ga_fitness,mimic_fitness)
        rhc_it_fit.append(hc_fitness/maxfitness)
        sa_it_fit.append(sa_fitness/maxfitness)
        ga_it_fit.append(ga_fitness/maxfitness)
        mimic_it_fit.append(mimic_fitness/maxfitness)
        
    rhc_fit.append(np.mean(rhc_it_fit))
    sa_fit.append(np.mean(sa_it_fit))
    ga_fit.append(np.mean(ga_it_fit))
    mimic_fit.append(np.mean(mimic_it_fit))
    
    rhc_time.append(np.mean(rhc_it_time))
    sa_time.append(np.mean(sa_it_time))
    ga_time.append(np.mean(ga_it_time))
    mimic_time.append(np.mean(mimic_it_time))
    
    

fitness_df = pd.DataFrame({'size':lengths, 'rhc_fit':rhc_fit,'sa_fit':sa_fit,'ga_fit':ga_fit,'mimic_fit':mimic_fit})

fitness_df.plot(x='size',y=['rhc_fit','sa_fit','ga_fit','mimic_fit'])

time_df = pd.DataFrame({'size':lengths, 'rhc_time':rhc_time, 'sa_time':sa_time,'ga_time':ga_time, 'mimic_time':mimic_time})

time_df.plot(x = 'size', y = ['rhc_time','sa_time','ga_time','mimic_time'],logy=True)

iterations = [50,100,500,1000,5000,20000]
arrlength = 20
hc_fit_calls = []
sa_fit_calls = []
ga_fit_calls = [] 
mimic_fit_calls = []



for fitness_calls in iterations:
     weights = np.random.randint(1,20,arrlength)
     values = np.random.randint(1,20,arrlength)
     max_weight_pct = .6
     fitness = mlrose.Knapsack(weights,values, max_weight_pct)
     problem = mlrose.DiscreteOpt(length = arrlength, fitness_fn = fitness, maximize = True)
     init_state = np.zeros(arrlength)
     
     best_state,hc_fitness = mlrose.random_hill_climb(problem, max_attempts=5, restarts = 20,
			max_iters=fitness_calls,init_state=init_state,random_state=random_seed)
     
     best_state, sa_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts=5, 
			max_iters=fitness_calls, init_state=init_state,random_state=random_seed)
     
     best_state,ga_fitness = mlrose.genetic_alg(problem, pop_size =10, mutation_prob = .1,
		 max_attempts=10, max_iters=fitness_calls,random_state=random_seed)
     
     best_state,mimic_fitness = mlrose.mimic(problem, pop_size = fitness_calls//3, keep_pct = .2,
		 max_attempts=10,random_state=random_seed,max_iters =fitness_calls -(fitness_calls//3) )
     
     maxfitness = max(hc_fitness,sa_fitness,ga_fitness,mimic_fitness)
     hc_fit_calls.append(hc_fitness/maxfitness)
     sa_fit_calls.append(sa_fitness/maxfitness)
     ga_fit_calls.append(ga_fitness/maxfitness)
     mimic_fit_calls.append(mimic_fitness/maxfitness)
     
func_calls_df = pd.DataFrame({'iterations':iterations, 'hc_calls':hc_fit_calls,'sa_calls':sa_fit_calls,'ga_calls':ga_fit_calls,'mimic_calls':mimic_fit_calls})
func_calls_df.plot(x = 'iterations', y = ['hc_calls','sa_calls','ga_calls','mimic_calls'])

     
     
     
    

