import six
import sys
sys.modules['sklearn.externals.six']=six
import numpy as np
import mlrose
import math
import matplotlib.pyplot as plt
import time

import pandas as pd





arrlength = 120
#fitness = mlrose.FourPeaks(.30)
fitness = mlrose.OneMax()
bigchungus = [0,1]*100
random_seed = 50



problem = mlrose.DiscreteOpt(length = arrlength, fitness_fn = fitness, maximize = True)
#init_state = bigchungus[0:arrlength]
init_state = np.zeros(arrlength)
    
attempts = [10,50,100,200]
max_iters = [100,1000,10000,20000]
restarts = [10,20,50,100]
record = {'attempt':0,'iters':0,'restarts':0}
max_fit = 0

for attempt in attempts:
    for iters in max_iters:
        for restart in restarts:
            #hill climbing
            start = time.time()
            best_state,hc_fitness = mlrose.random_hill_climb(problem, max_attempts=attempt, restarts =restart,
            		max_iters=iters,init_state=init_state,random_state=random_seed)
            end = time.time()
            
            if hc_fitness > max_fit:
                max_fit = hc_fitness
                record['attempt'] = attempt
                record['iters'] = iters
                record['restarts'] = restart
            
            #print("hill climbing")
            #print(best_state)
            #print(hc_fitness)
print("hill climbing")
print(max_fit)
print(record)



#annealing

schedules = [mlrose.ExpDecay(),mlrose.GeomDecay(),mlrose.ArithDecay()]
record = {'attempt':0,'iters':0}
max_fit=0

for attempt in attempts:
    for iters in max_iters:
        for schedule in schedules:
            #schedule = mlrose.ExpDecay()
            #schedule = mlrose.GeomDecay()
            #schedule = mlrose.ArithDecay()
            start = time.time()
            best_state, sa_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts=attempt, 
            			max_iters=iters, init_state=init_state,random_state=random_seed)
            end = time.time()
            
            if sa_fitness > max_fit:
                max_fit = sa_fitness
                record['attempt'] = attempt
                record['iters'] = iters
                


   
print("annealing")
#print(best_state)
#print(sa_fitness)
print(max_fit)
print(record)   

pop_sizes = [10,50,100,500,1000]
mutation_probs=[.0001,.001,.005,.01,.05,.1]
record = {'attempt':0,'iters':0,'pop':0,'prob':0}
max_fit=0
for attempt in attempts:
    for iters in max_iters:
        for pop in pop_sizes:
            for prob in mutation_probs:
            #genetic algorithms
                best_state,ga_fitness = mlrose.genetic_alg(problem, pop_size =pop, mutation_prob = prob,
                		 max_attempts=attempt, max_iters=iters,random_state=random_seed)
                if ga_fitness > max_fit:
                    max_fit = ga_fitness
                    record['attempt'] = attempt
                    record['iters'] = iters
                    record['pop'] = pop
                    record['prob']=prob
                    print('hey I did one')
print("ga")
print(max_fit)
print(record)
                

#print("genetic algorithms")
#print(best_state)
#print(ga_fitness)


#mimic
pop_sizes = [10,50,100,500,1000]
keep_pct = [.01,.05,.1,.2,.5]
record = {'attempt':0,'iters':0,'pop':0,'pct':0}
max_fit=0
for attempt in attempts:
    for iters in max_iters:
        for pop in pop_sizes:
            for pct in keep_pct:
                best_state,mimic_fitness = mlrose.mimic(problem, pop_size = pop, keep_pct =pct,
                max_attempts = attempt ,random_state=random_seed,max_iters= iters)
                if mimic_fitness > max_fit:
                    max_fit = mimic_fitness
                    record['attempt'] = attempt
                    record['iters'] = iters
                    record['pop'] = pop
                    record['pct']=pct
                    #print('hey, I did one')
print("mimic")
print(max_fit)
print(record)