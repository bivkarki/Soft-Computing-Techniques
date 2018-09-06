import numpy as np
from math import *
import matplotlib.pyplot as plt
x=np.linspace(-10,10,100) #create 100 points between -10 and 10
#x=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
y=x #linear
plt.figure()
plt.subplot(221)
plt.plot(x,y)
plt.title("Linear Activation Function")
plt.xlabel("X")
plt.ylabel("Y")


#threshold at zero
plt.subplot(222)
y=[]
i=0
while (i<len(x)):
    if x[i]<0:
        y.append(0)
    else:
        y.append(1)
    i+=1

plt.plot(x,y)
plt.title("Threshold Function at 0")
plt.xlabel("X")
plt.ylabel("Y")

#ramp function
plt.subplot(223)
y=[]
i=0
while (i<len(x)):
    if x[i]<0:
        y.append(0)
    elif x[i]>1:
        y.append(1)
    else:
        y.append(x[i])
    i+=1
plt.plot(x,y)
plt.title("Ramp function threshold at 1")
plt.xlabel("X")
plt.ylabel("Y")

#Log Sigmoid
plt.subplot(224)
y=[]
i=0
while(i<len(x)):
    y.append(1/(1+np.exp(-x[i])))
    i+=1
plt.plot(x,y)
plt.title("Log Sigmoidal function threshold at 1")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


    
        
