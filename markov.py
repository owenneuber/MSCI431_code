# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:06:41 2020

@author: Owen
"""

import numpy as np

# Data from the problem set For AHS 1 to 12
improve_rate = np.array([0, 0.00032, 0.001134, 0.00311, 0.006721, 0.010747, 0.014862, 0.024625, 0.033571, 0.040201, 0.030801, 0.032320, 0.00032, 0.001134, 0.00311, 0.006721, 0.010747, 0.014862, 0.024625, 0.033571, 0.040201, 0.030801, 0.03232])
deteriorate_rate = np.array([0.001013, 0.000961, 0.001473, 0.003199, 0.005537, 0.008135, 0.010858, 0.02387, 0.0313, 0.037877, 0.031978, 0])
death_rate = np.array([0.000123, 0.000172, 0.000245, 0.000319, 0.000417, 0.000564, 0.00076, 0.001029, 0.001372, 0.001862, 0.00245, 0.003553])
transplant_prob = np.array([0.0001, 0.0002, 0.0002, 0.0003, 0.0004, 0.0006, 0.0008, 0.0010, 0.0014, 0.0019, 0.0024, 0.0035])

def rate_to_prob(rate):
    return 1 - np.exp(-1*rate)

def apply_threshold(P, threshold_index):
    removed_prob = 0
    for row in range(0, threshold_index):
        removed_prob += P[row][-1]
        P[row][row] += P[row][-1] # the row's probs must always sum to 1 so we add the transplant prob to the prob of staying in the same state
        P[row][-1] = 0 
        
    basic_threshold = P[0:12,-1] # i.e. the updated last row
    allocation_factor = basic_threshold/sum(basic_threshold)
    for row in range(threshold_index, 12):
        P[row][-1] += allocation_factor[row]*removed_prob
        P[row][row] -= allocation_factor[row]*removed_prob # the row's probs must always sum to 1 so we subtract the added transplant prob from the prob of staying in the same state
        
    return P

##### Create the P matrix #####
deteriorate_prob = rate_to_prob(deteriorate_rate[0])
death_prob = rate_to_prob(death_rate[0])
# make top row
P = np.array([[1-deteriorate_prob-death_prob-transplant_prob[0], deteriorate_prob, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, death_prob, transplant_prob[0]]])

# do all rows between 2 and 11
for i in range(1,11): # note that python indexing starts at 0
    row = []
    num_leading_zeros = i-1
    num_trailing_zeros = 10-i
    row += [0]*num_leading_zeros
    improve_prob = rate_to_prob(improve_rate[i])
    deteriorate_prob = rate_to_prob(deteriorate_rate[i])
    death_prob = rate_to_prob(death_rate[i])
    row += [improve_prob, 1-improve_prob-deteriorate_prob-death_prob-transplant_prob[i], deteriorate_prob]
    row += [0]*num_trailing_zeros
    row += [death_prob, transplant_prob[i]]
    P = np.append(P, [row], axis=0) # update the matrix
    
# make the 12th row
improve_prob = rate_to_prob(improve_rate[11])
death_prob = rate_to_prob(death_rate[11])
row = [0]*10 + [improve_prob, 1-improve_prob-death_prob-transplant_prob[11], death_prob, transplant_prob[11]]
P = np.append(P, [row], axis=0)

# make the death and transplant rows
row = [0]*12 + [1] + [0]
P = np.append(P, [row], axis=0)
row = [0]*13 + [1]
P = np.append(P, [row], axis=0)
########### Matrix now made ###########  <--- Part (a)
# hard coded version of the matrix:
# M = np.array([[0.99876452, 0.001012487, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000122992, 0.0001], [0.000319949, 0.998347528, 0.000960538, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000171985, 0.0002], [0, 0.001133357, 0.996949757, 0.001471916, 0, 0, 0, 0, 0, 0, 0, 0, 0.00024497, 0.0002], [0, 0, 0.003105169, 0.993081993, 0.003193889, 0, 0, 0, 0, 0, 0, 0, 0.000318949, 0.0003], [0, 0, 0, 0.006698465, 0.986962923, 0.005521699, 0, 0, 0, 0, 0, 0, 0.000416913, 0.0004], [0, 0, 0, 0, 0.010689457, 0.980044701, 0.008102, 0, 0, 0, 0, 0, 0.000563841, 0.0006], [0, 0, 0, 0, 0, 0.014752106, 0.972888918, 0.010799265, 0, 0, 0, 0, 0.000759711, 0.0008], [0, 0, 0, 0, 0, 0, 0.024324278, 0.950059886, 0.023587365, 0, 0, 0, 0.001028471, 0.0010], [0, 0, 0, 0, 0, 0, 0, 0.033013747, 0.933399968, 0.030815226, 0, 0, 0.001371059, 0.0014], [0, 0, 0, 0, 0, 0, 0, 0, 0.03940366, 0.919667434, 0.037168638, 0, 0.001860268, 0.0019], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.030331482, 0.933349406, 0.031472111, 0.002447001, 0.0024], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.031811036, 0.961142268, 0.003546696, 0.0035], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

# Matrix multiplication example:
P_squared = P.dot(P)

############ calculate life expectancy from AHS state i (note python index starts at 0) ######### <--- Part (b)
AHS_starting_state = 5 -1 # i.e. AHS state 5
AHS_threshold = 5 -1 # i.e. only accept the liver if we are at AHS state 5+

P = apply_threshold(P, AHS_threshold)

Q = P[0:12,0:12]
R = P[0:12,12:14]
I = np.identity(12) # P[12:14,12:14]
transposed = np.linalg.inv(I-Q)
ones = np.ones((12,1))

patient_lifetime = transposed.dot(ones)
print(patient_lifetime[i]) # returns the lifetime pre-death/transplat for a given AHS


########### Calculate prob of transplat before death ####### <--- Part (c)

exit_states_from_starting_states = transposed.dot(R)