import numpy as np
import matplotlib.pyplot as plt
#fixing randm state for reproductability

np.random.seed(19680801)
mu,sigma = 100 , 15


x= mu +sigma * np.random.randn(10000)
#histogram of the data
n , bins ,patches = plt.hist(x,50,normed=1,facecolor='g',alpha=0.75)
#alpha => transparency


plt.xlabel("Smarts")
plt.ylabel("Probability")
plt.title("Histogram of IQ")
plt.text(60, .025 , r'$\mu=100,\ \sigma=15$')
# $\mu => mu
plt.axis([40,160,0,0.03])
plt.grid(True)
plt.show()
