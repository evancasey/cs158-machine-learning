from learning import *
import pdb
import math
import random
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt


def testNeuralNetLearner(data, iterations=1000, learn_rate=0.5, momentum=0.1):

	if data == "XOR":
	    ds = DataSet(name='../data/XOR')
	    num_input = 4
	    num_output = 4
	    num_hl = 1 # number of hidden layers
	elif data == "Semeion":
	    ds = DataSet(name='../data/semeion')
	    num_input = 256
	    num_output = 10
	    num_hl = 1 # number of hidden layers
	
	 # transform the ds to matrix
	 # m = _ds_to_matrix(ds)
	 
	 # # create the learner with the matrix
	 # NNlearner = createNeuralNetLearner(m, num_input, num_output, num_hl, iterations, learn_rate, momentum)
	 
	 # test the learner (cross validation)
		 

def createNeuralNetLearner(m, num_input, num_output, num_hl, iterations, learn_rate, momentum):
 	# create a neural net learner

 	# create weights
 	

 	# call forward propogation
 	

 	# call backward propogation
 	

 	# update weights
 	# 

def sigmoid(a):
	return 1/(1 + math.pow(math.e, -a))

def forwardProp(N, w):
	""" takes a list of node values and the associated weights"""
	if len(N) != len(w):
		return "error"
	else:
		a = 0
		for i in range(len(N)):
			a += N[i]*w[i]

		nodeValue = sigmoid(a)

		return nodeValue

def _create_weights(m):
	pass


def _ds_to_matrix(dataset):
	# return a numpy matrix
 
	# FILL IN
	
	pass



if __name__ == "__main__":

	testNeuralNetLearner("Semeion", 1000,0.5,0.1)

	# some stuff
	# 
	# testNeuralNetLearner()

