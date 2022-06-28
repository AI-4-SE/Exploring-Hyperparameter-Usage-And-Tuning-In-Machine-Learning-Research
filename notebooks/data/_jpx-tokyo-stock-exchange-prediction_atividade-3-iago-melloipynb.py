#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## IMPORTAR BIBLIOTECAS
import random
import datetime
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import configparser
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math


from deap import base, creator
from deap import tools


# In[ ]:


## DEFINIR QUAISQUER PARAMETROS NECESSARIOS
x1_min = 0
x1_max = 100
x2_min = 0
x2_max = 100

pop_init = 100
gen_init = 50

show_summary = 'True'


# In[ ]:


## DEFINIR FUNCOES APTIDAO E PENALIDADE
def evaluate(individual):
    #evaluate deve conter a F.O. + Penalidades
    #individual é o vetor contendo todas as variaveis de decisao
    x1 = individual[0]
    x2 = individual[1]
    
    #F.O.:
    FO_value = 100*x1 + 80*x2
    #penalidade
    penalty_value = penalty(individual)
    
    #aptidao final
    fitness_value = FO_value - penalty_value # menos, pois FO é de max
    #print(FO_value, penalty_value)
    
    return fitness_value, #IMPORTANTE, NO DEAP O VALOR DA FUNÇÃO EVALUATE SEMPRE TEM QUE TER DUAS VARIAVEIS DENTRO


# In[ ]:


def penalty(individual):
    #penalidade deve conter valores grandes para as restrições
    x1 = individual[0]
    x2 = individual[1]
    
    #rest 1: x1+x2 <=100
    if  (x1+x2) <= 100:
        pen0 = 0
    
    else:
        pen0 = 1000000
        
    #rest 2: 3*x1 + 1.5*x2 <= 240
    if (3*x1 + 1.5*x2) <= 240:
        pen1 = 0
    elif 240 < (3*x1 + 1.5*x2) <= 250:
        pen1 = 500
    elif 250 < (3*x1 + 1.5*x2) <= 260:
        pen1 = 10000
    else:
        pen1 = 100000
   
    #rest 3: x1 >=0
    if 0 <= x1 <= 100:
        pen2 = 0
    else:
        pen2 = 10000
    
    #rest 4: x2 >=0    
    if 0 <= x2 <= 100:
        pen3 = 0
    else:
        pen3 = 10000
    
           
    pen_total = pen0 + pen1 + pen2 + pen3
    #print(pen0, pen1, pen2, pen3)
    return pen_total


# In[ ]:


## GA SETUP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("x1", random.uniform, x1_min, x1_max) 
toolbox.register("x2", random.uniform, x2_min, x2_max)

toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.x1, toolbox.x2))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


# In[ ]:


## RODAR GA
def rodar_algoritmo_genetico():
    random.seed()
    pop = toolbox.population(n=pop_init)
    #print(pop)
    cx_prob, mut_prob, ngen = 0.7, 0.2, gen_init
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("Evaluated {} individuals".format(len(pop)))
    
    contador = 0
    for g in range(ngen):
        print("=========================")
        print("Generation {}".format(g))
        
        # choose the next generation
        offspring = toolbox.select(pop, len(pop))

        # clone selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # compute the crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # compute the mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("Evaluated {} individuals".format (len(invalid_ind)))

        # offsprint complete replacement strategy
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        nse = max(fits)
        best_ind = tools.selBest(pop, 1, fit_attr='fitness')[0]
        if show_summary == 'True':
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            print("  => Best_ind:", best_ind)
            print("  => Best_fit:", best_ind.fitness.values)
                    
        #stop criteria:
        #std <=1
        #if std <= 1:
            #break
        
    best_ind = tools.selBest(pop, 1)[0]
    
    print("Done!")    
    print("Best_ind:", best_ind, "Best_fit:", best_ind.fitness.values)


# In[ ]:


if __name__ == "__main__":
    inicio = datetime.datetime.now()
    print(inicio)
    rodar_algoritmo_genetico()
    fim = datetime.datetime.now()
    print ('Elapsed time: ',fim - inicio)  


# In[ ]:


evaluate([0.97, 0.22])

