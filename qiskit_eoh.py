import time


import numpy as np

#from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
#from qiskit import execute
# Import Aer
start_import = time.time()
from qiskit import BasicAer, compile
import matplotlib.pyplot as plt
#from qiskit.quantum_info.operators.pauli import Pauli

from qiskit_aqua.algorithms import EOH
end_import = time.time()
#from qiskit_aqua import Operator, QuantumInstance
from qiskit_aqua.components.initial_states import Zero

from qiskit.transpiler import PassManager
#from qiskit_aqua import get_aer_backend
from qiskit.qobj._qobj import QobjConfig
#from test.common import QiskitAquaTestCase
from qiskit_aqua import Operator, QuantumInstance

from hamiltonian import generate_classical_eigen, generate_epsilon, generate_marked_states


"""
Let's try a qiskit evolution
"""
# \param n number of qubits
def generate_driver_ham(n):

    X = [[0,1],[1,0]]
    I = np.identity(2)
    driver = np.zeros((2**n,2**n))
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
def generate_classical_ham(n,M,classical_estates):
    classic = np.zeros((2**n,2**n))
    for i in range(0,M):
        index = int(classical_estates[i][0])
        classic[index][index] = classical_estates[i][1]
    return classic




def generate_driver_ham_2(n):
    driver = np.zeros((2**n,2**n))
    for i in range(0,2**n):
        driver[(2**n)-i-1][i] = 1
    return driver

"""
for i in range(1,4):
    print(generate_driver_ham(i)-generate_driver_ham_2(i))


print(generate_driver_ham(3))
print(generate_driver_ham_2(3))

"""



"""
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