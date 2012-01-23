"""perceptron.py
Implements perceptron learning algorithm from Machine Learning by Stephen Marsland.

This implementation is only valid for a network with one neuron and will not function
for multiple neurons.

Written by Joe Reistetter
"""

import numpy as np

def import_data(path, digits=None):
    """Imports data formatted for the digit writing task.
    
    format: csv, first n-1 cols are features, last col is answer.
    
    params:
    path -- path to datafile
    digits -- list containing digits to distinguish
    
    returns:
    filtered_arr -- numpy array with features and -1 bias col appended
    answers --  vector of digits represented by associated features.
                recodes lower digit to 0, higher digit to 1.
    """
    arr = np.genfromtxt(path, dtype="i8", delimiter = ",")
    
    if digits:
        filtered_arr = []
        for row in arr:
            if row[-1] in digits:
                filtered_arr.append(list(row))
       
        filtered_arr = np.array(filtered_arr)
        n_col = filtered_arr.shape[1]
        answers = filtered_arr[:,n_col-1]
        answers = filter_digit(answers, digits)
        answers = np.array(answers)
        filtered_arr[:,n_col-1] = -1
        return filtered_arr, answers

    return arr

def filter_digit(answers, digits):
    """Goes through input answers and recodes lower digit to 0
    upper digit to 1

    params:
    answers -- 1-d numpy array of 2 digits produced by import_data
    digits -- two digits for training task

    returns:
    1-d numpy array, with the smaller digit coded to 0, the larger to 1."""
    
    smaller = min(digits)
    return [0 if val == smaller else 1 for val in answers]

def set_init_w(n_inputs):
    """Set the initial weights at small values. Code taken from
    Machine Learning by Stephen Marsland.

    params:
    n_inputs -- length of vector to produce

    returns:
    1-d numpy array of small random floats with length n_inputs"""
    
    return np.random.rand(n_inputs) * 0.1 - 0.05

def activation(input_vec, weights):
    """Use features and weights to calculate input to neuron
    and if it will fire

    params:
    input_vec -- 1-d numpy array of inputs to the neuron activation function
    weights -- 1-d numpy array of weights to apply to input vector

    returns:
    neuron_activation -- 1-d numpy array of neuron activations, coded 0/1 for not/activated
    """
    neuron_input = sum(input_vec * weights)
    if neuron_input >= 0:
        return 1
    
    return 0

    return neuron_activation

def update_weights(input_vec, weights, activation, answers, learning_rate):
    """Update the weights based on 2.3 on page 23

    params:
    input_vec -- 1-d numpy array of input to the neuron, produced by import_data
    weights -- 1-d numpy array of weights on edges between input nodes and neuron
    activation -- 1-d numpy array of activations from activation()
    answers -- 1-d numpy array of correct digit, produced by import_data
    learning_rate -- rate to control magnitude of weight adjustments

    returns:
    weights -- 1-d numpy array of updated weights
    """
    errors = answers - activation
    weights += learning_rate*errors*input_vec
    return weights

def confusion_matrix(predicted, answers):
    """Produce a confusion matrix from predicted and actual activations.

    params:
    predicted -- 1-d numpy array of predicted digits, coded 0/1
    answers -- 1-d numpy array of actual digits, coded 0/1

    returns:
    list -- list of lists corresponding to rows. Format of matrix is:
                            actual
                        upper   lower
    predicted   upper
                lower
    """
    lower_lower, lower_upper, upper_lower, upper_upper = (0,0,0,0)
    
    for i, prediction in enumerate(predicted):
        if prediction == 0 and answers[i] == 0:
            lower_lower += 1
        elif prediction == 0 and answers[i] == 1:
            lower_upper += 1
        elif prediction == 1 and answers[i] == 1:
            upper_upper += 1
        else:
            upper_lower += 1
        
    return [[lower_lower, lower_upper], [upper_lower, upper_upper]]


def train(data_path, digits, learning_rate, stop_threshold=None, n_epochs=None):
    """Train a neural network to discrminate between two digits
    
    params:
    data_path -- path to csv, first n-1 cols are features, last col is answer
    digits -- list of two digits to directly compare
    learning_rate -- rate to adjust weights
    stop_threshold -- threshold of accuracy improvement at which to halt learning
    n_epochs -- number of epochs to run training

    note: stop_threshold and n_epochs are mutually exclusive, if one is defined
    the other will not be used.
    
    returns:
    weights -- 1-d numpy array of the final weights from trained network
    accuracy -- accuracy of final weights on predicting the training dataset
    """
    features, answers = import_data(data_path, digits)
    
    weights = set_init_w(features.shape[1])
    
    training_output = np.zeros(features.shape[0])
    
    if stop_threshold:
        accuracy_change = 1
        n_epochs = 0
        prev_accuracy = 0
        
        while abs(accuracy_change) > stop_threshold:
        
            for i, input_vec in enumerate(features):
                activated = activation(input_vec, weights)
                training_output[i] = activated
                answer = answers[i]
                weights = update_weights(input_vec, weights, activated, answer, learning_rate)
                    
            accuracy = sum(training_output == answers) / float(len(answers))
            accuracy_change = accuracy - prev_accuracy
            
            prev_accuracy = accuracy
            
            
            n_epochs += 1
        
        return weights, n_epochs, accuracy
    
    if n_epochs:
        while n_epochs > 0:
        
            for i, input_vec in enumerate(features):
                activated = activation(input_vec, weights)
                training_output[i] = activated
                answer = answers[i]
                weights = update_weights(input_vec, weights, activated, answer, learning_rate)

            n_epochs -= 1
    	accuracy = sum(training_output == answers)/float(len(training_output))
        return weights, accuracy


def test_network(data_path, digits, weights):
    """Uses test data to determine accuracy of network generated by train
    
    params:
    data_path -- path to csv, first n-1 cols are features, last col is answer
    digits -- list of two digits to directly compare
    weights -- 1-d numpy array of weights generated by train
    
    returns:
    predictions -- 1-d numpy array of predicted neuron activations, coded 0/1
    answers -- 1-d numpy array of actual neuron activations, coded 0/1, from import_data
    
    """
    features, answers = import_data(data_path, digits)
    
    predictions = np.zeros(features.shape[0])
    
    for i, input_vec in enumerate(features):
        predictions[i] = activation(input_vec, weights)
    
    return predictions, answers
    
    
    
    
    
