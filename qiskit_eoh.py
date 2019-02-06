import time
import numpy as np
import matplotlib.pyplot as plt

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

# Get generator functions from other hamiltonian.py file
from IB_hamiltonian import generate_marked_states, generate_epsilon, generate_classical_eigen
"""
We use the qiskit eoh.py algorithm to evolve some given initial state with our hamiltonian
We measure with the classical hamiltonian component at the end of the evolution
The idea is to implement the Population Transfer protocol introduced in https://arxiv.org/pdf/1807.04792.pdf
In particular we investigate the impurity band model in this file. 
"""


# The driver hamiltonian is a sum of pauli X operators on each qubit
# \param n number of qubits
# returns a 2**n by 2**n matrix representing the driver hamiltonian
def generate_driver_ham(n):
    X = [[0,1],[1,0]]
    I = np.identity(2)
    driver = np.zeros((2**n,2**n))
    # we sum operators of the form I x I x ... x X x ... x I where the X moves along each time
    for i in range(0, n):
        if i == 0:
            hold = X
        else:
            hold = I
        for j in range(1,n):
            if j==i:
                hold = np.kron(hold,X)
            else:
                hold = np.kron(hold,I)
        driver += hold
    return driver


# The classical hamiltonian can be found in the introduction of https://arxiv.org/pdf/1807.04792.pdf
# Diagonal hamiltonian with only M non-zero diagonal entries which are approx equal to -n with some noise
# \param n is number of bits
# \param M is number of marked states
# \param classical_eigenstates a M x 2^n +1 matrix of evecs of driver ham with evals in col 0
def generate_classical_ham(n,M,classical_estates):
    classic = np.zeros((2**n,2**n))
    for i in range(0,M):
        index = int(classical_estates[i][0])
        classic[index][index] = classical_estates[i][1]
    return classic


# Here we implement the evolution and measurement with the hamiltonians constructed above
# \param n is number of bits
# \param M is number of marked states
# \param W is the width of the impurity ban
# \param driver_strength is the strength of the transverse field H = H_cl - driver_strength*H_d
# \param marked_states is a matrix of the states which have non-zero classical energies (ie in the impurity band)
def run_evolution(n, M, W, driver_strength, marked_states):
    epsilon = generate_epsilon(M, W) # generate the noise
    classical_estates = generate_classical_eigen(n,M,marked_states,epsilon)
    classical = generate_classical_ham(n,M,classical_estates)
    print(classical)
    driver = generate_driver_ham(n)
    # Construct the classical operator object which we measure with at the end
    qubit_op = Operator(matrix=classical)
    # Construct the evolution operator object which we evolve with
    evo_op = Operator(matrix=(classical-driver_strength*driver))
    # Construct the initial state (first marked state a good choice as must start in marked state)
    state = classical_estates[0][2:2+2**n]
    print("initial state of the evolution =", state)
    initial_state = Custom(n,'uniform',state)
    #initial_state = Zero(n)
    evo_time = 1 # evolution time needs to be set sufficiently large
    num_time_slices = 1

    # expansion order can be toggled in order to speed up calculations
    # Compute the evolution circuit
    eoh = EOH(qubit_op,initial_state,evo_op, 'paulis', evo_time,num_time_slices,expansion_mode='suzuki',expansion_order=2)
    circ = eoh.construct_circuit()
    qasm = circ.qasm()
    print(circ.draw())
    energies=[]
    file = open("qasm_n"+str(n)+"_evol"+str(evo_time)+"_num_slices"+str(num_time_slices)+".txt", 'w')
    for i in range(0,2**n):
        energy_i = classical[i][i]
        energies.append(energy_i)
        file.write(str(energy_i)+'\n')
    file.write(qasm)
    file.close()
    """
    backend = LegacySimulators.get_backend('statevector_simulator') # only the statevector_simulator works
    quantum_instance = QuantumInstance(shots=1,backend=backend, pass_manager=PassManager())
    # Execute our particular circuit
    result = eoh.run(quantum_instance) # this is where all the time cost is!
    # result is a tuple of outputs one just the average of the outcome the other the whole output
    ret = result[1]
    print('The result is\n{}'.format(ret))
    result = result[0] # get the state vector from this entry
    result_data_vec = result.results[0]  # need to go through a lot of layers to get to the vector
    update = result_data_vec.data.statevector  # output state vector at the end of the evolution
    print("updated state vector = ",update)
    norm = np.linalg.norm(update)
    if norm > 0:
        update = update / norm  # normalise if needed
    update = np.abs(update)
    update = update**2  # set update to be a vector of probabilities
    # compare just highlights all the marked states for comparison in a plot later produced
    compare = np.sum(classical_estates, axis=0)
    compare = compare[2:len(compare)]


    #plt.bar(range(2 ** n), compare)
    plt.bar(energies, update)
    plt.show()
    
    return update
    """

n=2
M=2
N=0
states = []
marked_states = generate_marked_states(n, M)
for state in marked_states:
    states.append(int(state,2))
print(states)
update = 0
start_evol = time.time()
update = run_evolution(n, M, 0.5, 5, marked_states)
end_evol = time.time()
#update = [0.00339019,0.000381006,0.0237346,0.0315199,0.0310795,0.018268,0.016974,0.0105108,0.06652,0.0445995,0.00181729,0.0143614,0.0125738,0.00659924,0.0110921,0.0011829,0.00663663,0.0744168,0.0091871,0.00392531,0.00206133,0.0656466,0.00981126,0.00360323,0.00304289,0.00272767,0.0169954,0.00415056,0.00333704,0.0657484,0.00161517,0.000373542,0.00197969,1.78023e-06,0.00245165,0.00279078,0.0434809,0.0347769,0.00259702,0.0183415,0.0124327,0.00379197,0.00490723,0.00117928,0.0144335,0.0337546,0.0404741,0.00111177,0.00265345,0.0629585,0.00132536,0.00366255,0.0399588,0.0223633,0.00055195,0.0183574,0.00460192,0.000730764,0.00686781,0.00184048,0.0145706,0.00521343,0.00871006,0.0192452]
#plt.bar(range(2 ** n), update)
#plt.show()
"""
for i in range(0, N):
    update += (run_evolution(n, M, 0.1, 2, marked_states))/N

print(" evol time = ",end_evol-start_evol)
compare = np.zeros(2**n)
for i in states:
    compare[i]=1
plt.bar(range(2 ** n), compare)
plt.bar(range(2 ** n), update)
plt.show()
"""
"""
AN ALTERNATIVE WAY TO INITIALISE THE ALGORITHM

params = {
    'problem': {
        'name': 'eoh'
    },
    'algorithm': {
        'name': 'EOH',
        'num_time_slices': 1
    },
    'initial_state': {
        'name': 'CUSTOM',
        'state': 'uniform'
    }
}
algo_input = EnergyInput(qubit_op)
algo_input.add_aux_op(evo_op)


ret = run_algorithm(params, algo_input, backend=backend)
print('The result is\n{}'.format(ret))
"""



"""

HERE IS SOME STUFF THAT I TRIED BUT IT DIDN'T REALLY WORK
PARAMS
start_setup = time.time()
n=4
M=9
W = 0.2
marked_states = generate_marked_states(n,M)
epsilon = generate_epsilon(M,W)
classical_estates = generate_classical_eigen(n,M,marked_states,epsilon)
classical = generate_classical_ham(n,M,classical_estates)
driver = generate_driver_ham(n)
print(marked_states)



SIZE = n

temp = np.random.random((2 ** SIZE, 2 ** SIZE))
h1 = temp + temp.T
qubit_op = Operator(matrix=h1)
state_in = Zero(n)


hamiltonian = classical-2*driver
evo_op = Operator(matrix=h1)
evo_time = 1
num_time_slices = 100

end_setup = time.time()




start_evol = time.time()
eoh = EOH(qubit_op, state_in, evo_op, 'paulis', evo_time, num_time_slices)

circ = eoh.construct_circuit()
end_evol = time.time()

start_execute = time.time()
#print(circ.draw())
#backend = get_aer_backend('statevector_simulator')
backend = BasicAer.get_backend('statevector_simulator')
#run_config = eoh.run(backend)
run_config = QobjConfig(shots=1,memory_slots=10, max_credits=10)
#backend.run(circ)
#quantum_instance = QuantumInstance(backend, run_config, pass_manager=PassManager())
#ret = eoh.run(quantum_instance, backend=backend)


qobj = compile(circ,backend)
job = backend.run(qobj)
result = job.result()
update = result.get_statevector()
end_execute = time.time()
norm = np.linalg.norm(update)
if norm > 0:
    update = update / norm
update = np.abs(update)
update=update**2
print(update)
compare = np.sum(classical_estates, axis=0)
compare = compare[2:len(compare)]
plt.bar(range(2 ** n), compare)
plt.bar(range(2 ** n), update)

plt.show()

print("import time = ",end_import-start_import,"setup time = ",end_setup-start_setup," evol time = ",end_evol-start_evol," execute time = ",end_execute-start_execute)
"""