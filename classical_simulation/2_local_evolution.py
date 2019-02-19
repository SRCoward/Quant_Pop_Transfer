import numpy as np
import matplotlib.pyplot as plt
from random import randint, uniform
from itertools import combinations
import time
from joblib import Parallel, delayed
import multiprocessing
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


def generate_classical_evals(n,h,J):
    classical_evals = np.zeros(2**n)
    for i in range(0,2**n):
        bit_str = bin(i)
        bit_str = bit_str[2:len(bit_str)]  # remove "0B" from binary string
        diff = n - len(bit_str)
        bit_str = '0' * diff + bit_str  # pad with zeros so all strings length n
        for j in range(0,n):
            bit_j = int(bit_str[j])
            classical_evals[i]+= h[j]*(-1)**bit_j
            for k in range(0,n):
                bit_k = int(bit_str[k])
                classical_evals[i] += J[j][k]*(-1)**(bit_j+bit_k)
    return classical_evals


def generate_driver_evals(n,h,J):
    driver_evals = np.zeros(2**n)
    for i in range(0,2**n):
        bit_str = bin(i)
        bit_str = bit_str[2:len(bit_str)]  # remove "0B" from binary string
        diff = n - len(bit_str)
        bit_str = '0' * diff + bit_str  # pad with zeros so all strings length n
        for j in range(0,n):
            bit_j = int(bit_str[j])
            driver_evals[i]+= (abs(h[j]) + 1)*(-1)**bit_j
            for k in range(0,n):
                bit_k = int(bit_str[k])
                driver_evals[i] += (1 + abs(J[j][k]))*(-1)**(bit_j+bit_k)
    return driver_evals

def single_BF_steepest_descent(classical_evals, start_state):
    # We compute the local minima of the hamiltonian using steepest descent starting from this initial state
    start_statenum = int(start_state,2)
    start_energy = classical_evals[start_statenum]
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
        if classical_evals[new_statenum] < final_energy:
            final_energy = classical_evals[new_statenum]
            final_state = state[i]
        #print('final_state = ',final_state, "final_energy =",final_energy,"start_energy = ",start_energy)
    if final_energy < start_energy:
        single_BF_steepest_descent(classical_evals,final_state)

    return final_state


def generate_local_minima(classical_evals,n):
    minima = []
    for i in range(0,2**n):
        state = bin(i)
        state = state[2:len(state)]
        diff = n - len(state)
        state = '0' * diff + state
        state = single_BF_steepest_descent(classical_evals,state)
        if state not in minima:
            minima.append(state)
    return minima

"""
Returns a version of the fast fourier transformed state vector but with plus and minus one instead of phases
statevec is the input state vector, of dim 2**n
ref: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
"""
def FFT_statevec(statevec):
    layer = 1
    while layer < len(statevec):
        for i in range(0, len(statevec), layer * 2):
            for j in range(i, i + layer):
                x = statevec[j]
                y = statevec[j + layer]
                statevec[j] = x + y
                statevec[j + layer] = x - y
        layer *= 2
    statevec = statevec/(np.sqrt(len(statevec)))
    return statevec

"""
Makes one trotter_suzuki style time step
n is number of qubits
statevec is the input vector
dt is the desired time step
evals are vectors of the corresponding hamiltonians
gamma is the factor which multiplies our driver hamiltonian
"""

def evolve_step(n,statevec,dt, classical_evals,driver_evals, gamma):
    statevec = FFT_statevec(statevec)
    #update = np.zeros(2**n,dtype=complex)
    #time_i_loop = 0
    exp_classical_evals = np.exp(-complex(0,1)*dt*classical_evals,dtype=complex128) #may be +ve
    exp_driver_evals = np.exp(-gamma*complex(0,1)*dt*driver_evals,dtype=complex128) #may be +ve
    file = open("exp_classical_evals.txt",'w')
    for i in range(0,2**n):
        data = exp_driver_evals[i]
        data = data.real
        file.write(str(exp_classical_evals[i].real)+','+str(exp_classical_evals[i].imag)+'\n')
    file.close()
    file = open("exp_driver_evals.txt",'w')
    for i in range(0,2**n):
        file.write(str(exp_driver_evals[i].real)+','+str(exp_driver_evals[i].imag)+'\n')
    file.close()
    inputs = range(0,2**n)

    num_cores = multiprocessing.cpu_count()
    update = Parallel(n_jobs=num_cores)(delayed(process_input)(i,statevec,exp_driver_evals,exp_classical_evals) for i in inputs)
    """
    for i in range(0,2**n):
        gap = time.time()
        classical_evec = np.zeros(2**n)
        classical_evec[i] = 1
        classical_evec = FFT_statevec(classical_evec)

        for k in range(0,2**n):
            update[i] += statevec[k]*exp_driver_evals[k]*classical_evec[k]
        update[i] *= exp_classical_evals[i]

        gap = time.time() - gap
        time_i_loop += gap
    print("time for i loop = ",time_i_loop)
    print("parallel results=",results-update)
    """
    return update


def process_input(i, statevec, exp_driver_evals, exp_classical_evals):
    update=0
    classical_evec = np.zeros(2 ** n)
    classical_evec[i] = 1
    classical_evec = FFT_statevec(classical_evec)
    for k in range(0, 2 ** n):
        update += statevec[k] * exp_driver_evals[k] * classical_evec[k]
    update *= exp_classical_evals[i]
    return update



n=16
bonds = generate_marked_bonds(n)
h,J = generate_h_J(n,bonds)
eval_time = time.time()
classical_evals = generate_classical_evals(n,h,J)
file = open("classical_evals.txt",'w')
for i in range(0,2**n):
	file.write(str(classical_evals[i])+'\n')
file.close()
driver_evals = generate_driver_evals(n,h,J)

minima = generate_local_minima(classical_evals,n)

eval_time=time.time()-eval_time
statevec = np.zeros((2**n))
statenum = int(minima[randint(0,len(minima)-1)],2)
statevec[statenum]=1
for string in minima:
    num = int(string,2)
    #print("state =",num," energy =",classical_evals[num])
print("starting state",statenum," with energy ",classical_evals[statenum])
print("core count =",multiprocessing.cpu_count())
update_time = time.time()
N=1

statevec = np.zeros((2 ** n))
statevec[statenum] = 1
#for i in range(0,100):
#    statevec = evolve_step(n,statevec,0.08,classical_evals,driver_evals,0.2)

update_time=time.time()-update_time
print("update =",update_time,"eval time =",eval_time)
dt = 0.08
gamma = 0.2
exp_classical_evals = np.exp(-complex(0,1)*dt*classical_evals) #may be +ve
exp_driver_evals = np.exp(-gamma*complex(0,1)*dt*driver_evals) #may be +ve
file = open("exp_classical_evals.txt",'w')
for i in range(0,2**n):
        file.write(str(exp_classical_evals[i].real)+','+str(exp_classical_evals[i].imag)+'\n')
file.close()
file = open("exp_driver_evals.txt",'w')
for i in range(0,2**n):
        file.write(str(exp_driver_evals[i].real)+','+str(exp_driver_evals[i].imag)+'\n')
file.close()
"""
plt.plot(range(0,2**n),classical_evals)

plt.show()
"""
#print(" updated state vector =",statevec)

statevec = np.abs(statevec)
statevec = statevec**2

