import numpy as np
import hamiltonian as ham
import matplotlib.pyplot as plt


# evolves the initial state via trotter for num_steps at step size dt
# \param dt is the time step size
# \param n is number of bits
# \param M is number of marked states
# \param num_steps is the max number of step (integer)
# \param W is the width of the impurity band
def evolve_state(n, M, dt, num_steps, W, field_strength):
    marked_states = ham.generate_marked_states(n, M)
    epsilon = ham.generate_epsilon(M, W)
    classical_estates = ham.generate_classical_eigen(n, M, marked_states, epsilon)
    bit_strings = ham.generate_bit_strings(n)

    driver_estates = ham.generate_driver_eigenstates(n, bit_strings)
    initial_state = classical_estates[0]
    print("start state=", initial_state[0])
    initial_state = initial_state[2:len(initial_state)]
    print("marked states=", classical_estates[0:M,0])
    update = []
    for i in range(0, num_steps):
        update = ham.trotter_time_step(n, M, dt, classical_estates, driver_estates, initial_state, field_strength)
    norm = np.linalg.norm(update)
    if norm > 0:
        update = update / norm
    update = np.abs(update)
    compare = np.sum(classical_estates,axis=0)
    compare = compare[2:len(compare)]
    update=update**2
    #print(compare)
    plt.bar(range(2**n),compare)
    plt.bar(range(2 ** n), update)

    plt.show()

    return update


update = evolve_state(6, 16, 0.1, 100, 0.2, 5)
#print(update)

