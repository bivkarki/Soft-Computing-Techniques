import numpy as np
import matplotlib.pyplot as plt

#evenly sampled time at 200ms interval
t=np.arange(0,5,0.2)
#red dotted line,blue square , green triangles
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()
