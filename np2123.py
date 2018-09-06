import numpy as np
a=np.array([1,2,3,4])
print(a)
print(a.shape)
b=np.zeros((3,4),dtype=np.int32)
print(b)
c=np.array([[1,2,3],[4,5,6]])
print(c)
print(type(c))
print(c.size,c.sum(),c.sum(axis=0),c.sum(axis=1))

d=np.ones((3,4),dtype=np.int32)
print(d)
print(np.arange(10,20,2)) #start,end,skip
print(d.T) #tranpose of a matrix
print(np.arange(10,20,2).reshape(5,1))
f=np.arange(20)
print(f.reshape(5,2,2))
print(f)
print(f.resize(5,4))
print(f)
print(np.linspace(10,20,15)) #to take 15 elements in between 10 t0 20
print(np.linspace(2.0, 3.0, num=5, retstep=True))
print(np.linspace(2.0, 3.0, num=5, endpoint=False))
print(np.dot(c,d))
print(c.dot(d))
a=np.random.random((2,3)) # random matrix of 2 * 3
print(a)

