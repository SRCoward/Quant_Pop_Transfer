import matplotlib.pyplot as plt
import numpy as np
# Get generator functions from other IB_hamiltonian.py file
from IB_hamiltonian import generate_marked_states, generate_epsilon, generate_classical_eigen

n = 6
M = 2**3
eigenvalues = np.zeros(2**n)
epsilon = generate_epsilon(M, 0.5)
marked_states = generate_marked_states(n, M)
# classical eigen is a matrix of dimension M x 2**n + 2
# the second column contains the corresponding eigenvalues of the marked states all others are zero
classical_eigen = generate_classical_eigen(n, M, marked_states, epsilon)
for i in range(0, M):
    state_num = int(classical_eigen[i][0])
    eigenvalues[state_num] = classical_eigen[i][1]

for i in range(0,2**n):
    print('('+str(i)+','+str(eigenvalues[i])+')')
plt.plot(range(2 ** n), eigenvalues)
plt.show()
