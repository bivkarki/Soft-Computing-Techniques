import numpy as np
class Perceptron(object):
    def __init__(self,eta=0.01 , n_iter=10):#eta=>learning rate n_iter=>no. of iter
        self.eta=eta
        self.n_iter=n_iter
    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1]) #np.zeros()=>zero filled array
        for _ in range(self.n_iter):

            for xi, target in zip(X,y): #zip() returns iterator 
                error=target-self.predict(xi)
                """ In actual learning error can't be determined,depends on various factors """
                if error!=0:
                    """Here we are learning data"""
                    update = self.eta * (self.predict(xi))
                    self.w_[1:] += update * xi
                    self.w_[0] +=update

        return self
    
    def net_input(self,X):
        """Calculate net input"""
        return np.dot(X,self.w_[1:])+self.w_[0]
    def predict(self,X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0,1,-1)
#input of 8 x 3
X=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
print("Inputs:\n",X)
#target
y=np.array([-1,-1,-1,-1,1,1,1,1])
##call perception
ppn = Perceptron (eta=0.1,n_iter=10)
ppn.fit(X,y)
##predict outputs
print("Outputs",ppn.predict(X))
print("Output for [1,1,1] is",ppn.predict([1,1,1]))
print("Output for [0,0,0] is",ppn.predict([0,0,0]))
#predicting O/p for given


"""
Perceptron part of artificial Neural Network

Supervised learning

=> requirements:
training data (X)
and target data (y)

Learning rate => 0 to 1
we have set learning rate 0.1 (eta)
"""

        
