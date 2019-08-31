import numpy as np

'''Actuation function'''

def sigmoid(x, deriv=False):
    if deriv == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def step(x):
    if x >= 0.5:
        return 1
    else:
        return 0

'''Inputs'''

x = np.array([[3,1.5],
              [2,1],
              [4,1.5],
              [3,1],
              [3.5,0.5],
              [2,0.5],
              [5.5,1],
              [1,1]])

'''Outputs'''

y = np.array([[1],
              [0],
              [1],
              [0],
              [1],
              [0],
              [1],
              [0]])

'''Nodes'''

input_nodes   = 2
hidden1_nodes = 4
hidden2_nodes = 4
output_node   = 1

'''Random seed'''

np.random.seed(1)

'''Synapses'''

syn0 = np.random.random((input_nodes, hidden1_nodes ))
syn1 = np.random.random((hidden1_nodes,hidden2_nodes))
syn2 = np.random.random((hidden2_nodes, output_node ))

'''Bias'''

bias_h1 = np.random.randn()
bias_h2 = np.random.randn()
bias_o  = np.random.randn()

'''Learning rate & Iteration'''

learning_rate = 0.1
iteration = 100000

'''Training loop'''

for i in range(iteration):

    # Layers and Propagation

    l0 = x
    l1 = sigmoid(np.dot(l0, syn0) + bias_h1)
    l2 = sigmoid(np.dot(l1, syn1) + bias_h2)
    l3 = sigmoid(np.dot(l2, syn2) + bias_o )

    # Error and Back propagation

    l3_error = y - l3
    if (i % (iteration/10)) == 0:
        print('Error: ' + str(np.mean(np.abs(l3_error * 100))) + ' % ')

    l3_delta = l3_error * sigmoid(l3, deriv=True) * learning_rate
    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error * sigmoid(l2, deriv=True) * learning_rate
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1, deriv=True) * learning_rate

    # Correction

    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
    syn2 += l2.T.dot(l3_delta)

'''Print result'''

print('-----------------')
print('Actual outputs: ' + str(y.T))
print('Output after training: ' + str(l3.T))


'''Inputs test'''

A = float(input('Inputs A : '))
B = float(input('Inputs B : '))

x = np.array([A,B])

'''Prediction'''

l0 = x
l1 = sigmoid(np.dot(l0, syn0) + bias_h1)
l2 = sigmoid(np.dot(l1, syn1) + bias_h2)
l3 = sigmoid(np.dot(l2, syn2) + bias_o )

output = step(l3)
print('Predicted output: ' + str(output))
