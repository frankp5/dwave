import datetime
import pandas as pd
import math
import neal
import dimod
import numpy 
from types import SimpleNamespace
from pyqubo import Array, Placeholder, solve_qubo, Constraint, Sum, Model, Mul
import networkx as nx

# Initialize parameters 
N=10 # number of containers 
M=12 # number tracks 
 
# Costs 
c_b = [2 ,7 ,1 ,6 ,2 ,4 ,8 ,7 ,7 ,10 ] 
c_t = [23 ,25 ,23 ,17 ,24 ,22 ,19 ,16 ,21 ,17 ] 
 
# Track capacity 
v = {} 
v = [5 ,5 ,5 ,5 ,5 ,5 ,5 ,5 ,5 ,5 ,5 ,5 ] 
 
routes = {} # route of containers 
routes[0] = [1,0,1,0,0,1,0,1,0,0,0,0] 
routes[1] = [1,0,1,0,1,0,1,0,0,0,0,0] 
routes[2] = [1,0,1,0,0,1,0,1,0,0,0,0] 
routes[3] = [1,0,0,0,0,0,1,0,1,1,0,0] 
routes[4] = [1,0,1,0,1,0,1,0,0,0,0,0]
routes[5] = [0,1,0,1,0,1,0,1,0,0,0,0] 
routes[6] = [1,0,1,0,1,0,1,0,0,0,0,0] 
routes[7] = [1,0,1,0,1,0,1,0,0,0,0,0] 
routes[8] = [1,0,1,0,1,0,1,0,0,0,0,0] 
routes[9] = [0,1,0,1,0,1,0,1,0,0,0,0] 
 
# Extra parameter 
K = 3 

# Initialize variable vector
size_of_variable_array = N + K*M

var = Array.create('vector', size_of_variable_array, 'BINARY')

# Defining constraints in the Model
minimize_costs = 0
minimize_costs += Constraint(Sum(0, N, lambda i: var[i]*(c_t[i]-c_b[i])+c_b[i]), label="minimize_transport_costs")

capacity_constraint = 0
for j in range(M):
    capacity_constraint += Constraint ( (Sum(0, N, lambda i: (1-var[i])*routes[i][j])
        + Sum(0, K, lambda i: var[N + j*K + i]*(2**(i))) - v[j])**2
        , label= "capacity_constraints"
    )

teller = 0

while teller<1:
    teller = teller + 1

    # parameter values
    A = 1
    B = 6
    Cs =240
    # Define Hamiltonian as a weighted sum of individual constraints
    H = A * minimize_costs +  B * capacity_constraint
 
    # Compile the model and generate QUBO
    model = H.compile()
    qubo, offset = model.to_qubo()
    print('chain_strength 1: '+str(max(qubo.values())))
 
    hulpmatrix=list(qubo.values())
    Cs2=sum(numpy.abs(hulpmatrix))
    print('chain_strength 2: '+str(Cs2))


    useQPU=True

    # Choose sampler and solve qubo
    if useQPU: 
        from dwave.system.samplers import DWaveSampler           # Library to interact with the QPU
        from dwave.system.composites import EmbeddingComposite   # Library to embed our problem onto the QPU physical graph
        from dwave.system import FixedEmbeddingComposite
        from dwave_qbsolv import QBSolv
        from minorminer import minorminer
        from dimod import qubo_to_ising
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(qubo, chain_strength=Cs, num_reads = 1000) #solver=DWaveSampler()) #, num_reads=50)   
    else:
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(qubo, num_sweeps=10000, num_reads=10)   

    # Postprocess solution
    sample  = response.first.sample

    obj = 0
    aantaly = 0
    for i in range(size_of_variable_array):
        if i<=N-1:
            obj += c_b[i]+ (c_t[i]-c_b[i])* sample['vector['+str(i)+']']

    print('objective: '+str(obj)) 