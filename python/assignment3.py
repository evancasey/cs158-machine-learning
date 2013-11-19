from learning import *
import pdb
import math
import random
import os
import sys
import numpy as np


def testNeuralNetLearner(data, iterations=1000, learn_rate=0.1, momentum=0.1):

    if data == "xor":
        ds = DataSet(name='../data/xor')
        num_input = 2
        num_output = 1
        num_hl = 1 # number of hidden layers
    elif data == "semeion":
        ds = DataSet(name='../data/semeion')
        num_input = 256
        num_output = 10
        num_hl = 1 # number of hidden layers

    for x in range(20):

        # create the learner with the matrix
        NNlearner = createNeuralNetLearner(ds, num_input, num_output, num_hl, iterations, learn_rate, momentum)
     
        # create a test observation
        obs = np.matrix(ds.examples)[1]

        # test the learner (cross validation)
        classification = NNlearner(obs)

        print classification
         

def createNeuralNetLearner(ds, num_input, num_output, num_hl, iterations, learn_rate, momentum):
    # create a neural net learner

    # transform the ds to matrix of inputs  
    obs = np.matrix(ds.examples)
    mx_inputs = np.matrix(np.delete(obs,len(ds.examples) - 2,1))
    mx_target = obs[:,num_input]

    # create input weights
    mw_input_hidden= np.matrix(np.random.random((num_input + 1, num_input)))

    # create output weights
    mw_hidden_output = np.matrix(np.random.random((num_input + 1, num_output)))

    # repeat for num iterations
    for i in range(iterations):             
    
        # go through each observation
        for j in range(len(mx_inputs)):            

            # add in bias column of 1's
            input_row_with_bias = np.matrix(np.append(1.0,mx_inputs[j]))            

            # call forward prop
            mn_hidden, mn_output = forwardProp(input_row_with_bias,mw_input_hidden,mw_hidden_output)            

            # call backward prop
            hidden_deltas, output_deltas = backwardProp(mx_target[j,0], mn_hidden, mn_output, mw_hidden_output, num_input, num_output)

            # update weights
            mwi, mwo = updateWeights(learn_rate, mwi, mwo, hidden_deltas, output_deltas, input_row_with_bias, mnh, mno)

            # print "mwi: ", mwi, "\n"

    def predict(row):
        ai, mnh, mno = forwardProp(row, mwi, mwo)
        return mno

    return predict
    

def forwardProp(row, mw_input_hidden, mw_hidden_output):
    # takes a list of node values and the associated weights
    # returns a matrix of updated input and output activations  

    # 1 dimensional input activation matrix
    a_hidden = row * mw_input_hidden

    # initialize matrix to store hidden node values
    mn_hidden = np.matrix([1.0] * a_hidden.shape[1])    

    # loop through all of hidden activations and update with sigmoid
    for i in range(a_hidden.shape[1]):            
        mn_hidden[0,i] = sigmoid(a_hidden[0,i])

    # add on bias node to hidden nodes
    mn_hidden = np.matrix(np.append(1.0,mn_hidden[0]))    

    # 1 dimensional output activation matrix
    a_output = mn_hidden * mw_hidden_output

    # initialize matrix to store output node values
    mn_output = np.matrix([1.0])

    # loop through all of output activations and update with sigmoid
    for i in range(a_output.shape[1]):    
        mn_output[0,i] = sigmoid(a_output[0,i]) 

    return mn_hidden, mn_output

def backwardProp(target, mn_hidden, mn_output, mw_hidden_output, num_input, num_output):    

    # create array to represent desired outcome
    desired = create_desired(target, num_output)

    # instantiate array for output deltas
    output_deltas = [0.0] * num_output

    # calculating the output deltas
    for i in range(mn_output.shape[1]):        
        error = desired[i] - mn_output[0,i]
        delta = error * mn_output[0,i] * (1 - mn_output[0,i])
        output_deltas[i] = delta    

    # instantiate array for hidden deltas
    hidden_deltas = [0.0] * (num_input + 1)

    # calculating the hidden layer deltas
    for i in range(mn_hidden.shape[1]):
        weighted_output_sum = 0.0
        # sum(weight to output node j * delta of output node j)
        for j in range(len(output_deltas)):
            weighted_output_sum += (mw_hidden_output[i,j] * output_deltas[j])

        hidden_deltas[i] = mn_hidden[0,i] * (1 - mn_hidden[0,i]) * weighted_output_sum 

    # take out the delta for the bias node
    hidden_deltas = hidden_deltas[1:]

    return hidden_deltas, output_deltas

def updateWeights(learn_rate, mwi, mwo, hidden_deltas, output_deltas, input_row_with_bias, mnh, mno):
    # return updated input and output weights

    # update input weights
    for i in range(len(input_row_with_bias)):       
        for j in range(mnh.shape[1] - 1):
            # pdb.set_trace()
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

def create_desired(target, num_output):
    # creates an array of binary values to represent 
    # desired values of the learner

    desired = [0.0] * num_output

    if num_output == 1:
        # xor
        desired[0] = target
    else:
        desired[target] = 1

    return desired


if __name__ == "__main__":

    testNeuralNetLearner("xor",10000,0.1,0.1)

    # do some stuff with the results

