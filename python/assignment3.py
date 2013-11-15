from learning import *
import pdb
import math
import random
import os
import sys
import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt


def testNeuralNetLearner(data, iterations=1000, learn_rate=0.1, momentum=0.1):

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
 
    for i in range(20):

        # create the learner with the matrix
        NNlearner = createNeuralNetLearner(ds, num_input, num_output, num_hl, iterations, learn_rate, momentum)
     
        # create a test observation
        obs = np.matrix(ds.examples)[0]

        # test the learner (cross validation)
        classification = NNlearner(obs)

        print classification
         

def createNeuralNetLearner(ds, num_input, num_output, num_hl, iterations, learn_rate, momentum):
    # create a neural net learner

    # transform the ds to matrix of inputs  
    obs = np.matrix(ds.examples)
    inputs = np.delete(obs,len(ds.examples) - 2,1)
    target = obs[:,num_input]
    mxi = np.matrix(inputs)
    mxt = np.matrix(target)

    # create input weights
    mwi = np.random.random((num_input + 1, num_input + 1))

    # create output weights
    mwo = np.random.random((num_input + 1, num_output))

    for i in range(iterations):             
    
        for j in range(len(mxi)):            

            # add in bias column of 1's
            input_row_with_bias = np.append(1.0,mxi[j])   

            # call forward prop
            ai, mnh, mno = forwardProp(np.matrix(input_row_with_bias),np.matrix(mwi),np.matrix(mwo))

            # grab target values
            target_value = mxt[j]

            # call backward prop
            output_deltas, hidden_deltas = backwardProp(target_value, mno, mnh, mwo, num_output, num_input + 1)

            # update weights
            mwi, mwo = updateWeights(learn_rate, mwi, mwo, hidden_deltas, output_deltas, input_row_with_bias, mnh, mno)

            # print "mwi: ", mwi, "\n"


    def predict(row):
        ai, mnh, mno = forwardProp(row, mwi, mwo)
        return mno

    return predict
    

def forwardProp(row, mwi, mwo):
    # takes a list of node values and the associated weights
    # returns a matrix of updated input and output activations  

    # print "ROW: ", row, "\n"

    # 1 dimensional input activation matrix
    ai = row * mwi

    # print "AI: ", ai, "\n"

    # initialize matrix to store hidden node values
    mnh = np.matrix([1.0] * ai.shape[1])

    # loop through all of ai and update with sigmoid
    for i in range(1, ai.shape[1]):            
        mnh[0,i] = sigmoid(ai[0,i])

    # print "MNH: ", mnh, "\n"

    # 1 dimensional output activation matrix
    ao = mnh * mwo

    # print "AO: ", ao, "\n"

    # initialize matrix to store output node values
    mno = np.matrix(ao)

    # loop through all of ao and update with sigmoid
    for i in range(ao.shape[1]):    
        mno[0,i] = sigmoid(ao[0,i])

    # print "MNO: ", mno, "\n"

    return ai, mnh, mno

def backwardProp(target, mno, mnh, mwo, num_output, num_hidden):

    # create array of all 0's except 1 for target value
    desired = [0.0] * num_output
    desired[target] = 1

    # instantiate array for output deltas
    output_deltas = [0.0] * num_output

    # calculating the output deltas
    for i in range(mno.shape[1]):
        error = desired[i] - mno[0,i]
        delta = error * mno[0,i] * (1 - mno[0,i])
        output_deltas[i] = delta

    # instantiate array for hidden deltas
    hidden_deltas = [0.0] * num_hidden

    # calculating the hidden layer deltas
    for i in range(mnh.shape[1]):
        weighted_output_sum = 0
        # sum(weight to output node j * delta of output node j)
        for j in range(len(output_deltas)):
            weighted_output_sum += (mwo[i,j]*output_deltas[j])

        hidden_deltas[i] = mnh[0,i] * (1 - mnh[0,i]) * weighted_output_sum

    return output_deltas, hidden_deltas

def updateWeights(learn_rate, mwi, mwo, hidden_deltas, output_deltas, input_row_with_bias, mnh, mno):
    # return updated input and output weights
    
    # update input weights
    for i in range(len(input_row_with_bias)):       
        for j in range(mnh.shape[1]):
            mwi[i,j] = mwi[i,j] + (learn_rate * input_row_with_bias[i] * hidden_deltas[j])

    # update output weights
    for i in range(len(mnh)):       
        for j in range(mno.shape[1]):
            mwo[i,j] = mwo[i,j] + (learn_rate * mnh[0,i] * output_deltas[j])

    return mwi, mwo


def sigmoid(a):
    # takes in a value in the activation matrix
    # returns the result of the sigmoid function performed on it

    return 1/(1 + math.pow(math.e, -a))


if __name__ == "__main__":

    testNeuralNetLearner("xor",10000,0.1,0.1)

    # do some stuff with the results

