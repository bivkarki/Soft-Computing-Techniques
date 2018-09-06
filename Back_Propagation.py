import numpy as np
#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])

#Sigmoid Function
def sigmoid(x):
    return 1/(1+ np.exp(-x))

#Derivative of Signmoid function
def derivatives_sigmoid(x):
    return x * (1-x)

#Variable initialisation
epoch=500  #setting trainiing iterations
lr=0.1 #setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons= 3 #number of hidden layer neurons
output_neurons = 1 #number of neuron at output layer

#weight and bias initialisation
wh = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh = np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))


for i in range(epoch):
   #Forward Propagation
    print('Epoch',i)
    hiddenlayer_input1=np.dot(X,wh)
    hiddenlayer_input=hiddenlayer_input1 + bh
    hiddenlayer_activations =  sigmoid(hiddenlayer_input)

    outputlayer_input1=np.dot(hiddenlayer_activations,wout)
    outputlayer_input= outputlayer_input1+bout
    output = sigmoid(outputlayer_input)

    #Back Propagation
    E= y - output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer

    Error_at_hidden_layer=d_output.dot(wout.T)

    d_hiddenlayer =  Error_at_hidden_layer * slope_hidden_layer

    wout=wout + hiddenlayer_activations.T.dot(d_output) * lr
    bout=bout + np.sum(d_output,axis=0,keepdims=True) * lr
    wh=wh+X.T.dot(d_hiddenlayer) * lr
    bh=bh+np.sum(d_hiddenlayer,axis=0,keepdims=True) * lr
print("Target Values")
print(y)
print("output Values")
print(output)
print('E= Target - Ouput')
print(E)
            
