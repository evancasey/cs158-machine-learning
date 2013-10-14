from learning import *
import pdb
import math
import random
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def testBinaryLearner(data, rounds):
    if data == "restaurant":
        attributeNames = "Alternate, Bar, Fri/Sat, Hungry, Patrons, Price, Raining, Reservation, Type, WaitEstimate, WillWait"
        ds = DataSet(name='../data/restaurant', attrnames=attributeNames)
    elif data == "cancer":
        attributeNames = "Class, age, menopause, tumor-size, inv-nodes, node-caps, deg-malig, breast, breast-quad, irradiat"
        ds = DataSet(name='../breast-cancer/breast-cancer', attrnames=attributeNames)
    elif data == "new_bands":
        attributeNames = "timestamp, cylinder_number, customer, job_number, paper_type, ink_type, paper_mill_location, plating_tank, proof_cut, viscosity, caliper, ink_temperature, humifity, roughness, blade_pressure, varnish_pct, press_speed, ink_pct, band_type"
        ds = DataSet(name='../data/trimmed_bands', attrnames=attributeNames)
        
    results = {}

    results['k'] = rounds


    ''' Non-boosted or bagged '''

    # make the decision tree
    dtLearner = BinaryDecisionTreeLearner(ds)
    #dtLearner.display()

    # normal training error
    results['normal_training_error'] = test(BinaryDecisionTreeLearner(ds),ds)
    
    # normal testing error
    results['normal_testing_error'] = cross_validation(BinaryDecisionTreeLearner, ds)

    print "Running cross validation for BinaryNormalLearner... %s" % str(results['normal_testing_error'])


    ''' Boosted '''

    #returns train
    adaLearner = AdaBoost(WeightedLearner(BinaryDecisionTreeLearner), rounds)

    # boosted training error
    results['boosted_training_error'] = test(adaLearner(ds),ds)

    # boosted testing error
    results['boosted_testing_error'] = cross_validation(adaLearner, ds)
    
    print "Running cross validation for BinaryAdaBoostNormalLearner... %s" % str(results['boosted_testing_error'])


    ''' Bagged '''

    # make decision tree
    baggedLearner = create_bagging_learners(BinaryDecisionTreeLearner, ds, rounds, len(ds.examples))

    results['bagged_training_error'] = test(baggedLearner(ds), ds)

    results['bagged_testing_error'] = cross_validation(baggedLearner, ds)

    print "Running cross validation for BinaryBaggedNormalLearner... %s" % str(results['bagged_testing_error'])


    return results


def testBinaryStumpLearner(data, rounds):
    if data == "restaurant":
        attributeNames = "Alternate, Bar, Fri/Sat, Hungry, Patrons, Price, Raining, Reservation, Type, WaitEstimate, WillWait"
        ds = DataSet(name='../data/restaurant', attrnames=attributeNames)
    elif data == "cancer":
        attributeNames = "Class, age, menopause, tumor-size, inv-nodes, node-caps, deg-malig, breast, breast-quad, irradiat"
        ds = DataSet(name='../breast-cancer/breast-cancer', attrnames=attributeNames)
    elif data == "new_bands":
        attributeNames = "timestamp, cylinder_number, customer, job_number, paper_type, ink_type, paper_mill_location, plating_tank, proof_cut, viscosity, caliper, ink_temperature, humifity, roughness, blade_pressure, varnish_pct, press_speed, ink_pct, band_type"
        ds = DataSet(name='../data/trimmed_bands', attrnames=attributeNames)
        

    ''' Non-boosted or bagged '''

    # make the decision tree
    stumpDTLearner = BinaryDecisionTreeStump(ds)
    #dtLearner.display()

    results = {}

    results['k'] = rounds

    # normal training error
    results['normal_training_error'] = test(BinaryDecisionTreeStump(ds),ds)
    
    # normal testing error
    results['normal_testing_error'] = cross_validation(BinaryDecisionTreeStump, ds)

    print "Running cross validation for BinaryStumpLearner... %s" % str(results['normal_testing_error'])


    ''' Boosted '''

    #returns train
    adaLearner = AdaBoost(WeightedLearner(BinaryDecisionTreeStump), rounds)

    # boosted training error
    results['boosted_training_error'] = test(adaLearner(ds),ds)

    # boosted testing error
    results['boosted_testing_error'] = cross_validation(adaLearner, ds)
    
    print "Running cross validation for BinaryAdaBoostStumpLearner... %s" % str(results['boosted_testing_error'])


    ''' Bagged '''

    # make decision tree
    baggedLearner = create_bagging_learners(BinaryDecisionTreeStump, ds, rounds, len(ds.examples))

    results['bagged_training_error'] = test(baggedLearner(ds), ds)

    results['bagged_testing_error'] = cross_validation(baggedLearner, ds)

    print "Running cross validation for BinaryBaggedStumpLearner... %s \n" % str(results['bagged_testing_error'])


    return results

    
#___________________________________________________________________________________________________


def testNaryLearner(data, rounds):
    if data == "restaurant":
        attributeNames = "Alternate, Bar, Fri/Sat, Hungry, Patrons, Price, Raining, Reservation, Type, WaitEstimate, WillWait"
        ds = DataSet(name='../data/restaurant', attrnames=attributeNames)
    elif data == "cancer":
        attributeNames = "Class, age, menopause, tumor-size, inv-nodes, node-caps, deg-malig, breast, breast-quad, irradiat"
        ds = DataSet(name='../breast-cancer/breast-cancer', attrnames=attributeNames)
    elif data == "new_bands":
        attributeNames = "timestamp, cylinder_number, customer, job_number, paper_type, ink_type, paper_mill_location, plating_tank, proof_cut, viscosity, caliper, ink_temperature, humifity, roughness, blade_pressure, varnish_pct, press_speed, ink_pct, band_type"
        ds = DataSet(name='../data/trimmed_bands', attrnames=attributeNames)
        
    results = {}

    results['k'] = rounds


    ''' Non-boosted or bagged '''

    # make the decision tree
    dtLearner = DecisionTreeLearner(ds)
    #dtLearner.display()

    # normal training error
    results['normal_training_error'] = test(DecisionTreeLearner(ds),ds)
    
    # normal testing error
    results['normal_testing_error'] = cross_validation(DecisionTreeLearner, ds)

    print "Running cross validation for NaryNormalLearner... %s" % str(results['normal_testing_error'])


    ''' Boosted '''

    #returns train
    adaLearner = AdaBoost(WeightedLearner(DecisionTreeLearner), rounds)

    # boosted training error
    results['boosted_training_error'] = test(adaLearner(ds),ds)

    # boosted testing error
    results['boosted_testing_error'] = cross_validation(adaLearner, ds)
    
    print "Running cross validation for NaryAdaBoostNormalLearner... %s" % str(results['boosted_testing_error'])


    ''' Bagged '''

    # make decision tree
    baggedLearner = create_bagging_learners(DecisionTreeLearner, ds, rounds, len(ds.examples))

    results['bagged_training_error'] = test(baggedLearner(ds), ds)

    results['bagged_testing_error'] = cross_validation(baggedLearner, ds)

    print "Running cross validation for NaryBaggedNormalLearner... %s" % str(results['bagged_testing_error'])


    return results
    

def testNaryStumpLearner(data, rounds):
    if data == "restaurant":
        attributeNames = "Alternate, Bar, Fri/Sat, Hungry, Patrons, Price, Raining, Reservation, Type, WaitEstimate, WillWait"
        ds = DataSet(name='../data/restaurant', attrnames=attributeNames)
    elif data == "cancer":
        attributeNames = "Class, age, menopause, tumor-size, inv-nodes, node-caps, deg-malig, breast, breast-quad, irradiat"
        ds = DataSet(name='../breast-cancer/breast-cancer', attrnames=attributeNames)
    elif data == "new_bands":
        attributeNames = "timestamp, cylinder_number, customer, job_number, paper_type, ink_type, paper_mill_location, plating_tank, proof_cut, viscosity, caliper, ink_temperature, humifity, roughness, blade_pressure, varnish_pct, press_speed, ink_pct, band_type"
        ds = DataSet(name='../data/trimmed_bands', attrnames=attributeNames)
        

    ''' Non-boosted or bagged '''

    # make the decision tree
    stumpDTLearner = DecisionTreeStump(ds)
    #dtLearner.display()

    results = {}

    results['k'] = rounds

    # normal training error
    results['normal_training_error'] = test(DecisionTreeStump(ds),ds)
    
    # normal testing error
    results['normal_testing_error'] = cross_validation(DecisionTreeStump, ds)

    print "Running cross validation for NaryStumpLearner... %s" % str(results['normal_testing_error'])


    ''' Boosted '''

    #returns train
    adaLearner = AdaBoost(WeightedLearner(DecisionTreeStump), rounds)

    # boosted training error
    results['boosted_training_error'] = test(adaLearner(ds),ds)

    # boosted testing error
    results['boosted_testing_error'] = cross_validation(adaLearner, ds)
    
    print "Running cross validation for NaryAdaBoostStumpLearner... %s" % str(results['boosted_testing_error'])


    ''' Bagged '''

    # make decision tree
    baggedLearner = create_bagging_learners(DecisionTreeStump, ds, rounds, len(ds.examples))

    results['bagged_training_error'] = test(baggedLearner(ds), ds)

    results['bagged_testing_error'] = cross_validation(baggedLearner, ds)

    print "Running cross validation for NaryBaggedStumpLearner... %s \n" % str(results['bagged_testing_error'])


    return results

#___________________________________________________________________________________________________

def create_bagging_learners(learner, dataset, rounds, n):
    # Take in a learner, dataset, num_samples, sample size
    # Returns list of learners
    
    def train(dataset):
        bagged_learners = []

        for r in range(rounds):
            sample = []
            for i in range(n):            
                sample.append(random.choice(dataset.examples))
            dataset.examples = sample

            # passed modified dataset examples    
            bagged_learners.append(learner(dataset))

        return process_learners(bagged_learners, dataset)

    return train
    

def process_learners(bagged_learners, dataset):
    # Takes in a list of learners
    # Iterates through obs, for each obs takes majority vote from each learner in list
    
    def predict(example):
        return _get_mode(predictor(example) for predictor in bagged_learners)

    return predict

#___________________________________________________________________________________________________

def _get_mode(values):
    # Takes a list of classifications
    # Returns majority
    
    totals = defaultdict(int)
    for v in values:
        totals[v] += 1
    return max(totals.keys(), key=totals.get)

    
def _average_results(results_dict):
    # returns list of avg results for a given num_rounds
    
    normal_testing_avg = sum(results_dict[j]['normal_testing_error'] for j in range(num_trials)) / num_trials
    boosted_testing_avg = sum(results_dict[j]['boosted_testing_error'] for j in range(num_trials)) / num_trials
    bagged_testing_avg = sum(results_dict[j]['bagged_testing_error'] for j in range(num_trials)) / num_trials
    normal_training_avg = sum(results_dict[j]['normal_training_error'] for j in range(num_trials)) / num_trials
    boosted_training_avg = sum(results_dict[j]['boosted_training_error'] for j in range(num_trials)) / num_trials
    bagged_training_avg = sum(results_dict[j]['bagged_training_error'] for j in range(num_trials)) / num_trials
    
    return [normal_testing_avg, boosted_testing_avg, bagged_testing_avg, normal_training_avg, boosted_training_avg, bagged_training_avg]


def _compose_results(avg_results_dict):
    # returns dict where each result type is key and values are a list of each kth result
    
    plot_dict = {'normal_testing_avg': [],
                 'boosted_testing_avg': [],
                 'bagged_testing_avg' : [],
                 'normal_training_avg': [],
                 'boosted_training_avg': [],
                 'bagged_training_avg': []}

    for k,v in avg_results_dict.items():
        plot_dict['normal_testing_avg'].append(v[0])
        plot_dict['boosted_testing_avg'].append(v[1])
        plot_dict['bagged_testing_avg'].append(v[2])
        plot_dict['normal_training_avg'].append(v[3])
        plot_dict['boosted_training_avg'].append(v[4])
        plot_dict['bagged_training_avg'].append(v[5])

    return plot_dict

def _plot_results(num_rounds, results_dict):
    # takes in list of num_rounds (x-coords), dict of results (y-coords and labels)
    # makes a matplotlib line chart with legend

    one_minus = lambda l: [1-x for x in l]

    plot0, = plt.plot(num_rounds, one_minus(results_dict['normal_testing_avg']))
    plot1, = plt.plot(num_rounds, one_minus(results_dict['boosted_testing_avg']))
    plot2, = plt.plot(num_rounds, one_minus(results_dict['bagged_testing_avg']))
    plot3, = plt.plot(num_rounds, one_minus(results_dict['normal_training_avg']))
    plot4, = plt.plot(num_rounds, one_minus(results_dict['boosted_training_avg']))
    plot5, = plt.plot(num_rounds, one_minus(results_dict['bagged_training_avg']))

    labels = ['normal_testing_avg', 'boosted_testing_avg', 'bagged_testing_avg', 
                'normal_training_avg', 'boosted_training_avg', 'bagged_training_avg']

    plt.legend([plot0,plot1,plot2,plot3,plot4,plot5],labels)

    plt.show()


if __name__ == "__main__":

    # number of trials used to take results average
    num_trials = 1

    # number of iterations for boosting/bagging algos
    num_rounds = [1, 2]

    # stores the averaged results
    rest_nary_tree_avg_results = {}
    rest_nary_stump_avg_results = {}
    canc_nary_tree_avg_results = {}
    canc_nary_stump_avg_results = {}

    rest_binary_tree_avg_results = {}
    rest_binary_stump_avg_results = {}
    canc_binary_tree_avg_results = {}
    canc_binary_stump_avg_results = {}

    for k in num_rounds:

        print "K = %d" % k

        # stores the results for each j in 0-9
        rest_nary_tree_results = {}
        rest_nary_stump_results = {}
        canc_nary_tree_results = {}
        canc_nary_stump_results = {}

        rest_binary_tree_results = {}
        rest_binary_stump_results = {}
        canc_binary_tree_results = {}
        canc_binary_stump_results = {}


        for j in range(num_trials):

            print "\nRESTAURANT: \n"
            rest_nary_tree_results[j] = testNaryLearner("restaurant", k)
            rest_nary_stump_results[j] = testNaryStumpLearner("restaurant", k)
            rest_binary_tree_results[j] = testBinaryLearner("restaurant", k)
            rest_binary_stump_results[j] = testBinaryStumpLearner("restaurant", k)

            print "\nCANCER: \n"
            canc_nary_tree_results[j] = testNaryLearner("cancer", k)
            canc_nary_stump_results[j] = testNaryStumpLearner("cancer", k)
            canc_binary_tree_results[j] = testBinaryLearner("cancer", k)
            canc_binary_stump_results[j] = testBinaryStumpLearner("cancer", k)


        # averages results for this n-round experiment
        rest_nary_tree_avg_results[k] = _average_results(rest_nary_tree_results)
        rest_nary_stump_avg_results[k] = _average_results(rest_nary_stump_results)
        rest_binary_tree_avg_results[k] = _average_results(rest_binary_tree_results)
        rest_binary_stump_avg_results[k] = _average_results(rest_binary_stump_results)

        canc_nary_tree_avg_results[k] = _average_results(canc_nary_tree_results)
        canc_nary_stump_avg_results[k] = _average_results(canc_nary_stump_results)
        canc_binary_tree_avg_results[k] = _average_results(canc_binary_tree_results)
        canc_binary_stump_avg_results[k] = _average_results(canc_binary_stump_results)


    # compose results into plot format
    rest_nary_tree_plot_dict = _compose_results(rest_nary_tree_avg_results)
    rest_nary_stump_plot_dict = _compose_results(rest_nary_stump_avg_results)
    rest_binary_tree_plot_dict = _compose_results(rest_binary_tree_avg_results)
    rest_binary_stump_plot_dict = _compose_results(rest_binary_stump_avg_results)

    canc_nary_tree_plot_dict = _compose_results(canc_nary_tree_avg_results)
    canc_nary_stump_plot_dict = _compose_results(canc_nary_stump_avg_results)
    canc_binary_tree_plot_dict = _compose_results(canc_binary_tree_avg_results)
    canc_binary_stump_plot_dict = _compose_results(canc_binary_stump_avg_results)

    # plot averages
    _plot_results(num_rounds, canc_nary_tree_plot_dict)
