from learning import *
import pdb
import math
import random
import os
import sys
import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt


def testNeuralNetLearner(data, iterations=1000, learn_rate=0.5, momentum=0.1):

	if data == "xor":
	    ds = DataSet(name='../data/xor')
	    num_input = 2
	    num_output = 2
	    num_hl = 1 # number of hidden layers
	elif data == "semeion":
	    ds = DataSet(name='../data/semeion')
	    num_input = 256
	    num_output = 10
	    num_hl = 1 # number of hidden layers
	 
	# create the learner with the matrix
	NNlearner = createNeuralNetLearner(ds, num_input, num_output, num_hl, iterations, learn_rate, momentum)
 
	# test the learner (cross validation)
		 

def createNeuralNetLearner(ds, num_input, num_output, num_hl, iterations, learn_rate, momentum):
 	# create a neural net learner

 	# transform the ds to matrix of inputs 	
 	obs = np.matrix(ds.examples)
 	inputs = np.delete(obs,len(ds.examples) - 2,1)
	mx = np.matrix(inputs)

 	# create input weights
 	mwi = np.random.random((num_input + 1, num_input + 1))

 	# create output weights
 	mwo = np.random.random((num_input + 1, num_output))
 	
	# call forward propogation
	for row in mx:
		# add in bias column of 1's
		row = np.append(1,row)
		forwardProp(np.matrix(row),np.matrix(mwi),np.matrix(mwo))

 	# call backward propogation
 	

 	# update weights
 	

def forwardProp(row, mwi, mwo):
	# takes a list of node values and the associated weights
	# returns a matrix of updated input and output activations 	

	# 1 dimensional input activation matrix
	ai = row * mwi

	# initialize matrix to store hidden node values
	mnh = ai

	# loop through all of ai and update with sigmoid
	for i in range(len(ai)):		
		mnh[i] = sigmoid(ai[0,i])

	# 1 dimensional output activation matrix
	ao = mnh * mwo

	# initialize matrix to store output node values
	mno = ai

	# loop through all of ao and update with sigmoid
	for i in range(len(ao[0])):	
		mno[i] = sigmoid(ao[0,i])

	return ai, mnh, mno

def backwardProp(mno,mnh):
	# for i in range(len(mno))
		# error = desired[i] - mno[i]
		# change in node = error * mno[i] * (1 - mno[i])
		# output_deltas[i] = change in node

	# for i in range(len(mnh))
		# sum = 0
		# for j in range(len(output_deltas))
			# sum += (mwo[i,j]*output_deltas[j])
		# change in node = mnh[i] * (1 - mnh[i]) * sum


def sigmoid(a):
	# takes in a value in the activation matrix
	# returns the result of the sigmoid function performed on it

	return 1/(1 + math.pow(math.e, -a))


def _create_weights(m):
	pass


if __name__ == "__main__":

	testNeuralNetLearner("xor",1000,0.5,0.1)

	# some stuff
	# 
	# testNeuralNetLearner()

