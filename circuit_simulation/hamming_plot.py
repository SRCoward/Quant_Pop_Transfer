import numpy as np
import matplotlib.pyplot as plt
# Load the data in format (energy,frequency)
data = np.loadtxt('results.csv',delimiter=',')

"""
We produce a plot of the hamming distance from our initial state against the frequency with which we measure these states
This is to demonstrate that we can measure states separated by large hamming distances from the initial state
"""

# computes the hamming distance between integers a and b
def hamming_distance(a,b):
    if a==b:
        return 0
    # include some debug
    if type(a)!=int:
        return "input a not an integer"
    if type(b)!=int:
        return "input b not an integer"
    # Compute bit strings for a and b which are of the same length
    a_bit = bin(a)
    b_bit = bin(b)
    a_bit = a_bit[2:] # remove leading bits
    b_bit = b_bit[2:] # remove leading bits
    n = len(b_bit)-len(a_bit)

    if n>0: # b_bit longer than a_bit pad a_bit with zeros
        a_bit = '0'*n + a_bit
    elif n<0: # a_bit longer than b_bit pad b_bit with zeros
        b_bit = '0' * (-n) + b_bit
    count = 0
    for i in range(0,len(a_bit)):
        if a_bit[i]!=b_bit[i]:
            count+=1
    return count

# these two lines get altered by a sed command
num_qubits=4
start_state=0
len_data = data.shape
len_data = len_data[0]
hamming_distances_frequencies = np.zeros(num_qubits+1)
# Compute the total frequency of measuring a state at each given hamming distance
for i in range(1, len_data):
    # the indexing is offset by one because we print the initial state twice once at the start of the results file.
    hamming_distances_frequencies[hamming_distance(i-1,start_state)] += data[i,1]
plt.bar(range(num_qubits+1), hamming_distances_frequencies)
plt.xlabel("Hamming Distance from initial state")
plt.ylabel("Frequency")
plt.show()
