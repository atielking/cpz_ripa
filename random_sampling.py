import numpy as np
import pandas as pd

#load the data
data = pd.read_csv('data/statistical_significance_data.csv')

#assign counts
exp, cont = data.condition.value_counts()

#total population
pop = data.shape[0]
#number of trials
n = 200000
#probability of getting either control/experiment group
p = 0.5

#run the simulation
samples = np.random.binomial(pop, p, n)

#store extreme cases here
flag = 0

# make note of outliers
for x in samples:
    if x>=exp or x <=cont:
        flag+=1
#report results        
print(flag/n)
