from random import randint, uniform

import numpy as np
from numpy import linalg
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
# Import Aer
from qiskit import BasicAer, compile
# Import Aqua
from qiskit_aqua.algorithms import EOH
from qiskit_aqua import Operator, QuantumInstance
from qiskit_aqua.components.initial_states import Zero

from qiskit.transpiler import PassManager
from qiskit_aqua import get_aer_backend
from qiskit.qobj._qobj import QobjConfig
#from test.common import QiskitAquaTestCase
from qiskit_aqua import Operator, QuantumInstance

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
    while len(marked) < M:
        string = bin(randint(0,2**n-1))
        string = string[2:len(string)] # remove "0B" from binary string
        diff = n - len(string)
        string = '0'*diff + string  # pad with zeros so all strings length n
        if string not in marked:
            marked.append(string)
    return marked


# Produce all bit strings of length n
# \param n length of bit strings
# returns a list of all bit strings
def generate_bit_strings(n):
    bit_strings=[]
    for i in range(0,2**n):
        string = bin(i)
        string = string[2:len(string)]  # remove "0B" from binary string
        diff = n - len(string)
        string = '0' * diff + string  # pad with zeros so all strings length n
        bit_strings.append(string)
    return bit_strings


# Generates M epsilon_i from a uniform dist [-W/2,W/2]
# \param M is number of marked states
# \param W is the width of the interval
def generate_epsilon(M, W):
    epsilon = []
    for i in range(0, M):
        epsilon.append(uniform(-W/2, W/2))
    return epsilon


# Compute the classical hamiltonian defined in ref [https://arxiv.org/pdf/1807.04792.pdf] eqn (4)
# \param n the number of qubits
# \param M the number of marked states to be chosen
# \param marked_states a list of M marked bit strings
# \param epsilon a list of M values distributed uniformly over [-W/2,W/2]
# \return a M x 2^n + 2 matrix of all evecs, corresponding eval in col one integer of bit string in col 0
def generate_classical_eigen(n, M, marked_states, epsilon):
    ham = np.zeros((M, 2 + 2**n))  # may run into memory issues here...
    for i in range(0, M):
        state_num = int(marked_states[i], 2)
        ham[i][0] = state_num
        ham[i][1] = -n + epsilon[i]
        ham[i][state_num+2] = 1

    return ham


# Generates the state vectors for the eigenstates of the driver hamiltonian
# \param n is number of bits
# \param bitstring is a list of all bit_strings of length n
# \return a 2^n x 2^n + 1 matrix of all eigenvectors with their corresponding eigenvalue in first column
def generate_driver_eigenstates(n, bit_strings):
    row = 0
    outputstate = np.zeros((2 ** n, 1 + 2 ** n)) # memory issues???
    for string in bit_strings:
        sum = 0
        q = QuantumRegister(n, 'q')
        circ = QuantumCircuit(q)
        # We generate a state e.g |+-++--+> of n qubits '0' -> |+>, '1' -> |->
        for i in range(0, n):

            if string[i] == '0':
                circ.h(q[i])
                sum += 1
            else:
                circ.x(q[i])
                circ.h(q[i])
                sum -= 1
        # Run the quantum circuit on a statevector simulator backend
        backend = BasicAer.get_backend('statevector_simulator')
        # Create a Quantum Program for execution
        job = execute(circ, backend)
        result = job.result()
        state = result.get_statevector(circ, decimals=3) # get the resulting state vector
        outputstate[row][0] = sum
        outputstate[row][1:2**n+1] = state.real # the row is an evec with eval sum

        row += 1

    return outputstate


# trotter_time_step computes one time step using the trotter suzuki decomposition to evolve the state
# \param n is number of bits
# \param M is number of marked states
# \param driver_eigenstates a 2^n x 2^n + 1 matrix of all eigenvectors with corresponding eigenvalue in first column
# \param classical_eigenstates a M x 2^n +1 matrix of evecs of driver ham with evals in col 0
# \param dt is the time step size
# \param initial_state is the state vector of the state we evolve for dt
# \return the evolved state
def trotter_time_step(n, M, dt, classical_eigenstates, driver_eigenstates,initial_state, field_strength):
    update = np.zeros(2**n,dtype=complex)
    state_nums = classical_eigenstates[0:4, 0]  # length M vector

    for i in range(0, 2**n):
        for k in range(0, 2**n):
            lambda_k_D = driver_eigenstates[k][0] # eval of kth estate of Driver

            vec_k_D = driver_eigenstates[k][1:2**n+1]
            # print("driver estate number k=",k, "state =", vec_k_D, "eval =", lambda_k_D)
            dot_product = np.dot(vec_k_D, initial_state)
            update[i] += np.exp(dt*complex(0, 1)*lambda_k_D*field_strength)*vec_k_D[i]*dot_product
        for k in range(0, len(state_nums)):

            if state_nums[k] == i:
                # print("classical eval", k, " value is", classical_eigenstates[k][1])
                update[i]*=np.exp(-dt*classical_eigenstates[k][1]*complex(0,1))
    return update

"""
lam_plus = (-1+np.sqrt(5))/2
lam_minus = (-1-np.sqrt(5))/2
output0 = ((1+0.25*lam_minus**2)**(-1))*np.exp(-complex(0,lam_minus))
output0 += ((1+0.25*lam_plus**2)**(-1))*np.exp(-complex(0,lam_plus))
print(output0)

output1 = [np.exp(complex(0,1))*np.cos(2), np.sin(2)]
print(output1)



"""