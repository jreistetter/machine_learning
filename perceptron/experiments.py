"""
Script to import functions from perceptron module and perform homework 1 experiments.

Written by Joe Reistetter
"""

#Wouldn't normally import * but small module, forgive me!
from perceptron import *

train_dat = "../../data/digits/optdigits.tra.txt"
test_dat = "../../data/digits/optdigits.tes.txt"

#Experiment 1
learning_rate = 0.2

for i in range(8):
	digits = [i, 8]
	weights, epochs, train_accuracy = train(train_dat, digits, learning_rate, stop_threshold=0.01)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))
	print """_______________________________
	Training %s vs %s
	
	number of epochs trained: %s
	learning rate: %s
	training accuracy: %s
	training accuracy: %s
	
	confusion matrix:
	""" % (digits[0], digits[1], epochs, learning_rate, train_accuracy, test_accuracy)
	confusion_matrix(predicted, answers, digits)
	
	print "_______________________________"


#Experiment 2
epochs = 6
for i in range(8):
	digits = [i, 8]
	weights, train_accuracy = train(train_dat, digits, learning_rate, n_epochs = epochs)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))
	print """_______________________________
	Training %s vs %s
	
	number of epochs trained: %s
	learning rate: %s
	training accuracy: %s
	training accuracy: %s
	
	confusion matrix:
	""" % (digits[0], digits[1], epochs, learning_rate, train_accuracy, test_accuracy)
	confusion_matrix(predicted, answers, digits)
	
	print "_______________________________"

#Experiment 3
epochs = 1
for i in range(8):
	digits = [i, 8]
	weights, train_accuracy = train(train_dat, digits, learning_rate, n_epochs = epochs)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))
	print """_______________________________
	Training %s vs %s
	
	number of epochs trained: %s
	learning rate: %s
	training accuracy: %s
	training accuracy: %s
	
	confusion matrix:
	""" % (digits[0], digits[1], epochs, learning_rate, train_accuracy, test_accuracy)
	confusion_matrix(predicted, answers, digits)
	
	print "_______________________________"

#Experiment 4
learning_rate = 0.5

for i in range(8):
	digits = [i, 8]
	weights, epochs, train_accuracy = train(train_dat, digits, learning_rate, stop_threshold=0.01)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))
	print """_______________________________
	Training %s vs %s
	
	number of epochs trained: %s
	learning rate: %s
	training accuracy: %s
	training accuracy: %s
	
	confusion matrix:
	""" % (digits[0], digits[1], epochs, learning_rate, train_accuracy, test_accuracy)
	confusion_matrix(predicted, answers, digits)
	
	print "_______________________________"

#Experiment 5
learning_rate = 0.1

for i in range(8):
	digits = [i, 8]
	weights, epochs, train_accuracy = train(train_dat, digits, learning_rate, stop_threshold=0.01)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))
	print """_______________________________
	Training %s vs %s
	
	number of epochs trained: %s
	learning rate: %s
	training accuracy: %s
	training accuracy: %s
	
	confusion matrix:
	""" % (digits[0], digits[1], epochs, learning_rate, train_accuracy, test_accuracy)
	confusion_matrix(predicted, answers, digits)
	
	print "_______________________________"
