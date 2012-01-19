"""perceptron.py
Implements perceptron learning algorithm from Machine Learning by Stephen Marsland.

Written by Joe Reistetter
"""

import numpy as np

def import_data(path, digits=None):
   arr = np.genfromtxt(path, dtype="i8", delimiter = ",")
   
   filtered_arr = []
   
   if digits:
       for row in arr:
           if row[64] in digits:
               filtered_arr.append(list(row))
       
       filtered_arr = np.array(filtered_arr)
       last_col = filtered_arr.shape[1]
       answers = array(filtered_arr[:,last_col-1])
       filtered_arr[:,last_col-1] = -1
       return filtered_arr, answers
   
   return arr

def set_init_w(n_inputs):
    return np.random.rand(n_inputs + 1) * 0.1 - 0.1

def activation(input, weights):
    neuron_input = np.dot(input, weights)
    neuron_activation = np.where(neuron_input > 0, 1, 0)

    return neuron_activation

def adjust_weights(input, weights, activations, answers, rate):
    error = (answers - activations)
    weights += rate*error*input
    
