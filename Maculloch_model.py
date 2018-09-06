import matplotlib.pyplot as plt
from math import *
import numpy  as np

def threshold(x):
    if x>0:
        return 1
    else:
        return 0
x=[]
neti=0
w=[]
n=int(input("enter size of inputs "))
for i in range(n):
    x.append(float(input("Enter Inputs ")))
    w.append(float(input("Enter Weights ")))
    neti+=x[i]*w[i]
b=float(input("enter Value of bias"))
neti+=b
out=threshold(neti)

print("output for McCulloch Pitts Neuron",out)
