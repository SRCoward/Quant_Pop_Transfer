import numpy as np
import matplotlib.pyplot as plt
# Load the data in format (energy,frequency)
data = np.loadtxt('results.csv',delimiter=',')
# plot bar graph of data
plt.bar(data[:,0],data[:,1],width=0.4)
plt.bar(data[0,0],data[0,1],width=0.4) # mark the initial state with a coloured bar.
plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.show()



