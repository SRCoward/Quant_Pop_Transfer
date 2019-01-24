from random import randint, uniform
import numpy as np

"""
Hamiltonian computes the hamiltionian with which we will evolve the system
"""


# Will uniformly at random select M unique states from a total set of 2^n states
# \param n the number of qubits
# \param M the number of marked states to be chosen
# \returns a list of M bit strings
def generate_marked_states(n, M):
    if M > 2**n:
        return "number of marked states must be less than total"
    marked = []
    while len(marked)<M:
        string = bin(randint(0,2**n-1))
        string = string[2:len(string)] # remove "0B" from binary string
        diff = n - len(string)
        string = '0'*diff + string  # pad with zeros so all strings length n
        if string not in marked:
            marked.append(string)
    return marked


# Compute the classical hamiltonian defined in ref [https://arxiv.org/pdf/1807.04792.pdf] eqn (4)
# \param n the number of qubits
# \param M the number of marked states to be chosen
# \param marked_states a list of M marked bit strings
# \param epsilon a list of M values distributed uniformly over [-W/2,W/2]
# returns a (2^n) vector containing the diagonal elements of the classical hamiltonian
def classical(n, M, marked_states, epsilon):
    ham = np.zeros((2**n))  # may run into memory issues here...
    for i in range(0, len(marked_states)):
        state_num = int(marked_states[i], 2)
        ham[state_num] = -n + epsilon[i]
    return ham


epsilon1 = []
for i in range(0,3):
    epsilon1.append(uniform(-0.1, 0.1))
print(epsilon1)

print(classical(2, 3, generate_marked_states(2, 3), epsilon1))
