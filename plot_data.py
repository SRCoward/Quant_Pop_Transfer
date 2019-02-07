import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('results.csv',delimiter=',')
sum = 0

for i in range(0,len(data[:,0])):
    if data[i][0]==0:
        sum+=data[i][1]
data = np.append(data,[[0, sum]],axis=0)
#print(data[:,0])
plt.bar(data[:,0],data[:,1],width=0.4)
plt.show()
