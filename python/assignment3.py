from learning import *
import pdb
import math
import random
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def testNeuralNetLearner(data, rounds):

		if data == "XOR":
		    ds = DataSet(name='../data/restaurant', attrnames=attributeNames)
		    num_input = 4
		    num_output = 4
		    num_hl = 1 # number of hidden layers
		elif data == "Semion":
		    ds = DataSet(name='../breast-cancer/breast-cancer', attrnames=attributeNames)
		    num_input = 256
		    num_output = 10
		    num_hl = 1 # number of hidden layers
		
		 # transform the ds to matrix
		 

		 # create the learner
		 
		 
		 # test the learner (cross validation)
		 

def createNeuralNetLearner(m):
	 	# create a neural net learner


	 	# create weights


def _ds_to_matrix(dataset):
		# return a numpy matrix
	 
		# FILL IN
		
		pass



if __name__ == "__main__":

		# some stuff
		# 
		# testNeuralNetLearner()

