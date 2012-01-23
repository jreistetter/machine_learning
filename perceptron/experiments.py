"""
Script to import functions from perceptron module and perform homework 1 experiments.

command line usage:
python experiments.py <path to training data> <path to test data>

Script usage:
edit train_dat and test_dat variables to represent the paths to the respective data files.

Written by Joe Reistetter
"""

from perceptron import * #Not normally good practice to import * but small module
import sys

if sys.argv[1]:
	train_dat = sys.argv[1]
	test_dat = sys.argv[2]
else:
	train_dat = "../../data/digits/optdigits.tra.txt"
	test_dat = "../../data/digits/optdigits.tes.txt"

#Experiment 1
learning_rate = 0.2
results = open("hw_1_output.txt", "w")
results.write("Experiment 1\n")
results.write("digit\tn_epochs\tlearning rate\ttraining_acc\ttest_acc\n")
confusion_matrices = []
for i in range(8):
	digits = [i, 8]
	weights, epochs, train_accuracy = train(train_dat, digits, learning_rate, stop_threshold=0.01)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))
	
	vals = [digits[0], epochs, learning_rate, train_accuracy, test_accuracy]
	vals = [str(val) for val in vals]
	line = "\t".join(vals)
	results.write(line + "\n")
	
	conf_mat = confusion_matrix(predicted, answers)
	confusion_matrices.append(conf_mat)

for i in range(8):
	mat = confusion_matrices[i]
	results.write("Confusion matrix for %s \n" % i)
	results.write("\t".join([str(val) for val in mat[0]]) + "\n")
	results.write("\t".join([str(val) for val in mat[1]]) + "\n")


#Experiment 2
epochs = 6
confusion_matrices = []
results.write("Experiment 2\n")
results.write("digit\tn_epochs\tlearning rate\ttraining_acc\ttest_acc\n")
for i in range(8):
	digits = [i, 8]
	weights, train_accuracy = train(train_dat, digits, learning_rate, n_epochs = epochs)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))

	vals = [digits[0], epochs, learning_rate, train_accuracy, test_accuracy]
	vals = [str(val) for val in vals]
	line = "\t".join(vals)
	results.write(line + "\n")
	
	conf_mat = confusion_matrix(predicted, answers)
	confusion_matrices.append(conf_mat)

for i in range(8):
	mat = confusion_matrices[i]
	results.write("Confusion matrix for %s \n" % i)
	results.write("\t".join([str(val) for val in mat[0]]) + "\n")
	results.write("\t".join([str(val) for val in mat[1]]) + "\n")
	

#Experiment 3
epochs = 1
confusion_matrices = []
results.write("Experiment 3\n")
results.write("digit\tn_epochs\tlearning rate\ttraining_acc\ttest_acc\n")
for i in range(8):
	digits = [i, 8]
	weights, train_accuracy = train(train_dat, digits, learning_rate, n_epochs = epochs)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))
	
	vals = [digits[0], epochs, learning_rate, train_accuracy, test_accuracy]
	vals = [str(val) for val in vals]
	line = "\t".join(vals)
	results.write(line + "\n")
	
	conf_mat = confusion_matrix(predicted, answers)
	confusion_matrices.append(conf_mat)

for i in range(8):
	mat = confusion_matrices[i]
	results.write("Confusion matrix for %s \n" % i)
	results.write("\t".join([str(val) for val in mat[0]]) + "\n")
	results.write("\t".join([str(val) for val in mat[1]]) + "\n")

#Experiment 4
learning_rate = 0.5
confusion_matrices = []
results.write("Experiment 4\n")
results.write("digit\tn_epochs\tlearning rate\ttraining_acc\ttest_acc\n")

for i in range(8):
	digits = [i, 8]
	weights, epochs, train_accuracy = train(train_dat, digits, learning_rate, stop_threshold=0.01)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))
	
	vals = [digits[0], epochs, learning_rate, train_accuracy, test_accuracy]
	vals = [str(val) for val in vals]
	line = "\t".join(vals)
	results.write(line + "\n")
	
	conf_mat = confusion_matrix(predicted, answers)
	confusion_matrices.append(conf_mat)

for i in range(8):
	mat = confusion_matrices[i]
	results.write("Confusion matrix for %s \n" % i)
	results.write("\t".join([str(val) for val in mat[0]]) + "\n")
	results.write("\t".join([str(val) for val in mat[1]]) + "\n")

#Experiment 5
learning_rate = 0.1
confusion_matrices = []
results.write("Experiment 5\n")
results.write("digit\tn_epochs\tlearning rate\ttraining_acc\ttest_acc\n")
for i in range(8):
	digits = [i, 8]
	weights, epochs, train_accuracy = train(train_dat, digits, learning_rate, stop_threshold=0.01)
	predicted, answers = test_network(test_dat, digits, weights)
	test_accuracy = sum(predicted == answers) / float(len(predicted))
	
	vals = [digits[0], epochs, learning_rate, train_accuracy, test_accuracy]
	vals = [str(val) for val in vals]
	line = "\t".join(vals)
	results.write(line + "\n")
	
	conf_mat = confusion_matrix(predicted, answers)
	confusion_matrices.append(conf_mat)

for i in range(8):
	mat = confusion_matrices[i]
	results.write("Confusion matrix for %s \n" % i)
	results.write("\t".join([str(val) for val in mat[0]]) + "\n")
	results.write("\t".join([str(val) for val in mat[1]]) + "\n")

results.close()
