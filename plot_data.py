import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('results.csv',delimiter=',')
print(data[:,0])
plt.bar(data[:,0],data[:,1],width=0.05)
plt.show()