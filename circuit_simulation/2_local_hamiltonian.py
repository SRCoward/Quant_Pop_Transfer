import time
import numpy as np
import matplotlib.pyplot as plt
from random import randint, uniform
from itertools import combinations
from numpy import kron
# Import qiskit to compile/run
from qiskit import LegacySimulators
from qiskit.transpiler import PassManager

# Import qiskit_aqua classes etc
from qiskit_aqua import run_algorithm
from qiskit_aqua.operator import Operator, QuantumInstance
from qiskit_aqua.algorithms import EOH
from qiskit_aqua.components.initial_states import Custom
from qiskit_aqua.input import EnergyInput
from qiskit_aqua.components.initial_states import Zero
from qiskit.result.result import Result

"""
Here we aim to reproduce the results of https://arxiv.org/pdf/1807.04792.pdf section 4
We investigate the 2 local hamiltonian presented in the paper.
"""

"""
Generate a subset of n/2 marked bonds to be dimers
\param n is the number of quibits
"""
def generate_marked_bonds(n):
    marked_bonds = []
    subsets = list(combinations(range(n), 2))
    upper = len(subsets)


    while len(marked_bonds)< (n/2):
        index = randint(0,upper-1)
        if subsets[index] not in marked_bonds:
            marked_bonds.append(subsets[index])
    return marked_bonds


"""
We generate the random 2**n dim vector h and (2**n x 2**n) matrix J which are the coefficients in the hamiltonian
\param n is number of qubits
\param marked_bonds is subset of n/2 bonds which are dimers
"""
def generate_h_J(n,marked_bonds):
    h = []
    J = np.zeros((n,n))
    for i in range(0,n):
        h.append(round(uniform(-1,1),6))
        for j in range(i,n):
            if (i,j) in marked_bonds:
                J[i][j] = -4
            elif (j,i) in marked_bonds:
                J[i][j] = -4

            else:
                J[i][j] = round(uniform(-1,1),6)
            J[j][i] = J[i][j]
    return h, J


"""
Generate the classical portion of the hamiltonian ref section 4 of https://arxiv.org/pdf/1807.04792.pdf
\param n is number of qubits
\param h vector of coefficients 2**n dim
\param J matrix of coefficients 2**n x 2**n dim
"""
def generate_classical_ham(n, h, J):
    Z = [[1,0],[0,-1]]
    I = np.identity(2)
    classic = np.zeros((2**n,2**n))
    classic_2 = np.zeros((2 ** n, 2 ** n))
    # we sum operators of the form I x I x ... x Z x ... x I where the Z moves along each time
    for i in range(0, n):
        if i == 0:
            hold = Z
        else:
            hold = I
        for j in range(1,n):
            if j==i:
                hold = np.kron(hold, Z)
            else:
                hold = np.kron(hold, I)

        classic += h[i] * hold
        # Here we compute the J component of the classical Hamiltonian
        for j in range(0,n):
            # driver_2[i][j] = kron product of Z's and I's
            if i == j:
                classic_2 += J[i][j]*np.identity(2**n)
            else:
                if i == 0 or j == 0:
                    hold2 = Z
                else:
                    hold2 = I
                for k in range(1,n):
                    if k == i or k == j:
                        hold2 = np.kron(hold2,Z)
                    else:
                        hold2 = np.kron(hold2,I)
                classic_2 += J[i][j]*hold2


    classical_ham = classic + classic_2
    return classical_ham


"""
Generate the driver portion of the hamiltonian ref section 4 of https://arxiv.org/pdf/1807.04792.pdf
\param n is number of qubits
\param h vector of coefficients 2**n dim
\param J matrix of coefficients 2**n x 2**n dim
"""
def generate_driver_ham(n, h, J):
    X = [[0, 1], [1, 0]]
    I = np.identity(2)
    driver = np.zeros((2 ** n, 2 ** n))
    driver_2 = np.zeros((2 ** n, 2 ** n))
    # we sum operators of the form I x I x ... x Z x ... x I where the Z moves along each time
    for i in range(0, n):
        if i == 0:
            hold = X
        else:
            hold = I
        for j in range(1, n):
            if j == i:
                hold = np.kron(hold, X)
            else:
                hold = np.kron(hold, I)

        driver += (abs(h[i]) + 1) * hold
        for j in range(0,n):
            # driver_2[i][j] = kron product of X's and I's
            if i == j:
                driver_2 += (abs(J[i][j])+1)*np.identity(2**n)
            else:
                if i == 0 or j == 0:
                    hold2 = X
                else:
                    hold2 = I
                for k in range(1,n):
                    if k == i or k == j:
                        hold2 = np.kron(hold2, X)
                    else:
                        hold2 = np.kron(hold2, I)
                driver_2 += (abs(J[i][j])+1)*hold2
    driver_ham = driver + driver_2
    return driver_ham


"""
single_BF_steepest_descent - computes a steepest descent optimisation initialised from start_state
This locates the local minima which the starting state will lead to following single bit flips
\param classical_evals - vector of classical evals
\param start_state - state number of the initial state
"""
def single_BF_steepest_descent(ham, start_state):
    # We compute the local minima of the hamiltonian using steepest descent starting from this initial state
    start_statenum = int(start_state,2)
    start_energy = ham[start_statenum][start_statenum]
    final_energy = start_energy
    state = []
    final_state = start_state
    n = len(start_state)
    for i in range(0, n):
        if start_state[i] == '0':
            if i==0:
                state.append( '1'+start_state[1:n] )
            elif i==n-1:
                state.append( start_state[0:n-1]+'1' )
            else:
                state.append( start_state[0:i]+'1'+start_state[i+1:n] )
        else: #start_state[i] == '1'
            if i==0:
                state.append('0'+start_state[1:n])
            elif i==n-1:
                state.append(start_state[0:n-1]+'0')
            else:
                state.append(start_state[0:i]+'0'+start_state[i+1:n])
        # Now we've updated our bit string with one bit flip
        new_statenum = int(state[i],2)
        #print(new_statenum,state)
        if ham[new_statenum][new_statenum] < final_energy:
            final_energy = ham[new_statenum][new_statenum]
            final_state = state[i]
        #print('final_state = ',final_state, "final_energy =",final_energy,"start_energy = ",start_energy)
    if final_energy < start_energy:
        single_BF_steepest_descent(ham,final_state)

    return final_state


"""
generate_local_minia computes all local minima which correspond to an exact state number.
\param classical_evals - vector of eigenvalues of our classical Hamiltonian
\param n - number of qubits in the system
"""
def generate_local_minima(ham,n):
    minima = []
    for i in range(0,2**n):
        state = bin(i)
        state = state[2:len(state)]
        diff = n - len(state)
        state = '0' * diff + state
        state = single_BF_steepest_descent(ham,state)
        if state not in minima:
            minima.append(state)
    return minima


"""
evolve - generates the circuit for the evolution operator.
We generate a qasm script of the circuit and that gets passed to our quest simulator
We also have the ability to simulate our circuit using this model but the simulator is much slower
"""
def evolve(n,evo_time,num_time_slices,expansion_order):
    print("n=",n)
    # Problem setuo
    gamma = 0.2
    marked_bonds = generate_marked_bonds(n)
    #print(marked_bonds)
    h, J = generate_h_J(n, marked_bonds)
    classical_ham = generate_classical_ham(n, h, J)
    driver_ham = generate_driver_ham(n, h, J)
    local_min = generate_local_minima(classical_ham,n)
    minima =[]
    for string in local_min:
        minima.append(int(string,2))

    #print(minima)
    qubit_op = Operator(matrix=classical_ham) # create the classical operator, which we measure evals of
    # Construct the evolution operator object which we evolve with
    evo_op = Operator(matrix=(classical_ham+gamma*driver_ham)) # add to it the driver to form the operator we actually evolve with
    start_index = randint(0,len(local_min)-1)
    state_num = int(local_min[start_index],2)
    state = np.zeros(2**n)
    state[state_num] = 1
    print("initial state of the evolution =", state_num)
    # initialise the circuit
    initial_state = Custom(n,'uniform',state)


    # expansion order can be toggled in order to speed up calculations
    # Create the evolution of hamiltonian object
    eoh = EOH(qubit_op,initial_state,evo_op, 'paulis', evo_time, num_time_slices, expansion_mode='trotter',expansion_order=expansion_order)


    circtime = time.time()
    # construct the circuit
    circ = eoh.construct_circuit()
    circtime = time.time() - circtime
    qasmtime = time.time()
    # generate the qasm data
    qasm = circ.qasm()
    qasmtime = time.time() - qasmtime
    print("circuit construction time = ",circtime," qasm write time = ",qasmtime)

    file = open("qasm.txt",'w')
    file.write(str(state_num)+'\n')
    for i in range(0,2**n):
        energy_i = classical_ham[i][i]
        file.write(str(energy_i)+'\n')
    file.write(qasm)
    file.close()

    # Here is where we cam actually use the inbuilt qiskit simulator
    """
    backend = LegacySimulators.get_backend('statevector_simulator') # only the statevector_simulator works
    quantum_instance = QuantumInstance(shots=1024, backend = backend, pass_manager=PassManager())
    # Execute our particular circuit
    result = eoh.run(quantum_instance) # this is where all the time cost is!
    # result is a tuple of outputs one just the average of the outcome the other the whole output
    ret = result[1]
    print('The result is\n{}'.format(ret))
    result = result[0] # get the state vector from this entry
    result_data_vec = result.results[0]  # need to go through a lot of layers to get to the vector
    update = result_data_vec.data.statevector  # output state vector at the end of the evolution
    print("actual amplitudes = ", update)

    update = np.abs(update)
    update = update**2  # set update to be a vector of probabilities
    print("updated state vector = ", update)
    compare = np.zeros(2**n)
    for i in minima:
        compare[i]=1
    energies = []
    for i in range(0,2**n):
        energies.append(classical_ham[i][i])
    plt.bar(energies, compare)
    plt.bar(energies, update)

    plt.show()
    """

# Run the actual construction and time
num_qubits=4
timings = open("timing.csv","a")
start_time = time.time()
evolve(num_qubits,4,50,1)
end_time = time.time()
timings.write('('+str(num_qubits)+','+str(end_time-start_time)+')'+'\n')
timings.close()

