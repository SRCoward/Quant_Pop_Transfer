import numpy as np
from qiskit import LegacySimulators
from qiskit.transpiler import PassManager
from qiskit_aqua import run_algorithm
from qiskit_aqua.operator import Operator, QuantumInstance
from qiskit_aqua.algorithms import EOH
from qiskit_aqua.components.initial_states import Custom
from qiskit_aqua.input import EnergyInput
from qiskit_eoh import generate_marked_states, generate_epsilon, generate_classical_eigen, generate_classical_ham,generate_driver_ham_2, generate_driver_ham
from qiskit_aqua.components.initial_states import Zero
from qiskit.result.result import Result
import matplotlib.pyplot as plt


def run_evolution(n, M, W, driver_strength, marked_states):

    # marked_states = generate_marked_states(n,M)
    epsilon = generate_epsilon(M,W)
    classical_estates = generate_classical_eigen(n,M,marked_states,epsilon)
    classical = generate_classical_ham(n,M,classical_estates)
    driver = generate_driver_ham_2(n)
    print("classical hamiltonian ",classical)
    print("driver hamiltonian ",driver)


    qubit_op = Operator(matrix=classical)

    evo_op = Operator(matrix=(classical-driver_strength*driver))

    state = classical_estates[0][2:2+2**n]
    print(state)

    initial_state = Custom(state)
    initial_state = Zero(n)
    evo_time = 10
    num_time_slices = 100



    eoh = EOH(qubit_op,initial_state,evo_op, 'paulis', evo_time,num_time_slices,expansion_mode='suzuki',expansion_order=2)

    backend = LegacySimulators.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend, pass_manager=PassManager())

    result = eoh.run(quantum_instance)
    #print('The result is\n{}'.format(ret))
    eoh_circuit = eoh.construct_circuit()
    print(eoh_circuit.draw())
    ret = result[1]
    print('The result is\n{}'.format(ret))
    result = result[0]
    result_data_vec = result.results[1]
    statevector = result_data_vec.data.statevector




    update = statevector
    print(statevector)
    norm = np.linalg.norm(update)
    if norm > 0:
        update = update / norm
    update = np.abs(update)
    update=update**2

    compare = np.sum(classical_estates, axis=0)
    compare = compare[2:len(compare)]
    """
    plt.bar(range(2 ** n), compare)
    plt.bar(range(2 ** n), update)
    plt.show()
    """
    return update

n=2
M=2

marked_states = generate_marked_states(n,M)
print(marked_states)
update = 0
update = run_evolution(n, M, 0.1, 2, marked_states)
"""
for i in range(0, 10):
    update += (run_evolution(3, 3, 0.1, 2, marked_states))/10
"""
plt.bar(range(2 ** n), update)
plt.show()
"""
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
