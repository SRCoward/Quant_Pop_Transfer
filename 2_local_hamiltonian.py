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
        h.append(uniform(-1,1))
        for j in range(i,n):
            if (i,j) in marked_bonds:
                J[i][j] = -4
            elif (j,i) in marked_bonds:
                J[i][j] = -4

            else:
                J[i][j] = uniform(-1,1)
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
    #print(driver)
    driver_ham = driver + driver_2
    return driver_ham


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


def evolve(n,evo_time,num_time_slices,expansion_order):
    #n = 4
    gamma = 0.2
    marked_bonds = generate_marked_bonds(n)
    #print(marked_bonds)
    h, J = generate_h_J(n, marked_bonds)

    #h=[0,0]
    #J=[[0,0],[0,0]]
    X = [[0, 1], [1, 0]]
    I = np.identity(2)
    XI = np.kron(X,I)
    IX = np.kron(I,X)
    XX = np.kron(X,X)
    classical_ham = generate_classical_ham(n, h, J)
    driver_ham = generate_driver_ham(n, h, J)
    #ans = (abs(J[0,0]) + abs(J[1,1]) + abs(J[2,2])+3)*kron(kron(I,I),I)
    #ans += (abs(J[1,0])+abs(J[0,1])+2)*kron(XX,I)
    #ans += (abs(J[0,2])+abs(J[2,0])+2)*kron(XI,X)
    #ans += (abs(J[1,2])+abs(J[2,1])+2)*kron(IX,X)


    #print(ans-driver_ham)



    #test = single_BF_steepest_descent(classical_ham, '10')
    local_min = generate_local_minima(classical_ham,n)
    minima =[]
    for string in local_min:
        minima.append(int(string,2))

    print(minima)
    for i in range(0,2**n):
        print(classical_ham[i,i])
    qubit_op = Operator(matrix=classical_ham)
    # Construct the evolution operator object which we evolve with
    evo_op = Operator(matrix=(classical_ham+gamma*driver_ham))
    # Construct the initial state (first marked state a good choice as must start in marked state)
    start_index = randint(0,len(local_min)-1)
    state_num = int(local_min[start_index],2)
    state = np.zeros(2**n)
    state[state_num] = 1
    print("initial state of the evolution =", state)
    initial_state = Custom(n,'uniform',state)
    #initial_state = Zero(n)
    #evo_time = 100 # evolution time needs to be set sufficiently large
    #num_time_slices = 50

    # expansion order can be toggled in order to speed up calculations
    # Compute the evolution circuit
    eoh = EOH(qubit_op,initial_state,evo_op, 'paulis', evo_time, num_time_slices, expansion_mode='suzuki',expansion_order=expansion_order)
    circ = eoh.construct_circuit()
    qasm = circ.qasm()
    #print(circ.draw())
    file = open("qasm.txt",'w')
    file.write(qasm)
    file.close()
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
#evolve(6,10,300,2)
#ans = (abs(h[0])+1)*XI + (abs(h[1])+1)*IX + (abs(J[0,0])+abs(J[1,1])+2)*np.identity(4) + (abs(J[0,1])+abs(J[1,0])+2)*XX
#print(driver_ham-ans)
update = [0.0003713,0.00352192,0.00194864,0.000729625,0.00171505,0.00194903,0.000600375,0.000734134,0.00267611,0.00132657,0.000428424,0.00175401,0.00211246,0.000868837,0.00637151,0.000419189,0.00131795,0.00206435,0.00270362,0.00618022,0.00506279,0.0148935,0.00721335,0.00664659,0.00157628,0.00241277,0.00123426,0.000625983,0.000348013,0.000609569,0.0037661,0.000775622,0.00538568,0.00937779,0.000411002,0.00416552,0.00194125,0.00506693,0.00176799,0.00525268,0.00218005,0.00193875,0.0015486,0.00445118,0.000281597,0.000736425,0.00289309,0.0021985,0.000935793,0.00405781,0.00043244,0.00204313,0.00174723,0.00247475,0.00924348,0.000106196,0.00236499,0.000939215,0.000104316,0.00341343,0.00484768,0.00116119,0.00229579,0.00374067,0.000993135,0.0162346,0.000479557,0.000537183,0.00189209,0.000103511,0.00194671,0.00209781,0.00102024,0.0127036,0.00277341,0.0026834,0.00421568,0.00202353,0.0010944,0.00239642,0.00274799,0.000549659,0.00599522,0.000734482,0.0019634,0.00401702,0.00397271,0.0019977,0.00208558,0.00074559,0.0033224,0.00114501,0.00207163,0.00207685,0.00266926,0.000508948,0.000460105,0.00850737,0.00064008,0.00662628,0.00257175,0.0054968,0.00291315,0.00105566,0.0031859,0.00520629,0.00215419,0.00374125,0.00310114,0.000341487,0.000308777,0.00917567,0.00221893,0.003754,0.00243675,0.0013427,0.0013134,0.000750959,0.000504394,0.236485,0.00239934,0.00303363,0.00152865,0.00657164,0.000851437,0.00257442,0.0059231,0.00128386,0.00179389,0.001863,0.000878635,0.00179779,0.000600988,0.000522654,0.000282525,0.000710775,0.112183,0.00148797,0.00165445,0.000454131,0.00317448,0.00196186,0.0020098,0.000606583,9.40057e-05,0.00125736,0.00143145,1.1995e-05,0.000104265,0.00178157,0.0111489,0.000748453,0.00337035,2.20363e-05,0.000343519,0.00293002,0.000887184,0.00066379,0.000979351,0.00249401,0.00188112,0.00133724,0.00090508,0.00246493,0.000132863,0.000461082,0.00663705,0.0121704,0.00229172,0.00256907,0.00281276,0.00113142,0.001234,0.00896071,0.00086311,0.00185146,0.00103901,0.00151852,0.00201418,0.00251133,0.00248117,0.00209994,0.00413471,0.011314,0.000850754,0.00202473,0.003999,0.000110852,0.000760397,0.00323338,0.0183095,0.00717053,9.63223e-05,0.00420743,0.00373063,0.00047691,0.0036836,0.000842174,0.0010488,0.0014008,0.000213247,0.00241942,0.000257981,0.00282013,0.00313836,0.00293322,0.000901059,0.000475643,0.00211862,0.000261958,0.00102902,0.00167324,0.00154167,0.00171575,0.00152661,0.000410581,0.00172561,0.00251899,0.00101915,1.47608e-05,0.000469595,0.000836132,0.0011606,0.00360421,0.00334783,0.00126735,0.000305135,0.000919185,0.000897538,0.000402568,0.00053954,0.010617,0.0103203,0.00103397,3.36113e-07,0.00072016,0.00514777,0.00457352,0.000614791,7.95844e-05,0.0015491,0.00075708,0.000531043,0.00385595,0.000515902,0.00467607,0.0137236,0.000674151,0.00246884,0.00069907,0.00064601,0.00470812,0.00329886,0.000497325,0.001187,0.00490662]
marked_states = [119, 19, 186, 167, 33, 14, 191, 73, 136, 148, 83, 21, 82, 233, 25, 190]

states = np.zeros(2**8)
for i in marked_states:
    states[i]=0.3
plt.bar(range(2**8),states)
plt.bar(range(2 ** 8), update)
plt.show()
"""
ans = (J[0][1]+J[1][0])*np.kron(np.kron(Z,Z),I)
ans += (J[0][2]+J[2][0])*np.kron(ZI,Z)
ans += (J[1][2]+J[2][1])*np.kron(IZ,Z)
"""
