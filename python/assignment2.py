from learning import *
import pdb
import math
import random
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt

class BinaryDecisionFork:
    """A fork of a decision tree holds an attribute to test, and a dict 
    of branches, one for each of the attribute's values."""

    def __init__(self, attr, attrname=None, branches=None):
        "Initialize by saying what attribute this node tests."
        self.attr_map = {}
        update(self, attr=attr, attrname=attrname or attr,
               branches=branches or {})

    def __call__(self, example):
        "Given an example, classify it using the attribute and the branches."
        attrvalue = example[self.attr_map[self.attr].keys()[0]]
        
        if attrvalue == self.attr_map[self.attr].values()[0]:
            return self.branches['Yes'](example)
        else:
            return self.branches['No'](example)

    def add(self, val, subtree):
        "Add a branch.  If self.attr = val, go to the given subtree."
        self.branches[val] = subtree
        
    def update_attr_map(self, attr_map):
        self.attr_map = attr_map

    def display(self, indent=0):
        name = self.attrname
        print 'Test', name
        for (val, subtree) in self.branches.items():
            print ' '*4*indent, name, ': [is', self.attr_map[self.attr].values()[0], ']=', val, '==>',
            subtree.display(indent+1)

    def __repr__(self):
        return ('DecisionFork(%r, %r, %r)'
                % (self.attr, self.attrname, self.branches))

def BinaryDecisionTreeLearner(dataset):
    "[Fig. 18.5]"
    
    # Construct new attributes list and accompanying mapping

    target, values = dataset.target, dataset.values
    
    #Create our new data structures
    orig_attrs = dataset.inputs
    new_attrs = []
    attr_map = dict()
    
    num_count = 0
    for a in dataset.inputs:
        for v in values[a]:
            attr_map[num_count] = {a:v} #update({count,2})
            new_attrs.append(num_count)
            num_count+=1
            
    #print "original attrs: %s \n\n new attrs: %s \n\n attrmap:%s\n\n" % (orig_attrs, new_attrs, attr_map)

    def decision_tree_learning(examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return plurality_value(parent_examples)
        elif all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        elif len(attrs) == 0:
            return plurality_value(examples)
        else:
            A = choose_attribute(attrs, examples)
            tree = BinaryDecisionFork(A, dataset.attrnames[attr_map[A].keys()[0]])
            tree.update_attr_map(attr_map)
            for (v_k, exs) in split_by(A, examples):
                subtree = decision_tree_learning(
                    exs, removeall(A, attrs), examples)
                tree.add(v_k, subtree)
            return tree

    def plurality_value(examples):
        """Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality.)"""
        popular = argmax_random_tie(values[target],
                                    lambda v: count(target, v, examples))
        return DecisionLeaf(popular)

    def count(attr, val, examples):
        #print "attr: %s \n\n val: %s \n\n values: %s \n\n \n\n\n\n" % (attr, val, attr_map[attr].values()[0])
        if (val == "Yes"): # corresponds to 'Yes' branch - does value fit boolean attribute?
            return count_if(lambda e: e[attr_map[attr].keys()[0]] == attr_map[attr].values()[0], examples)
        else:
            return count_if(lambda e: e[attr_map[attr].keys()[0]] != attr_map[attr].values()[0], examples)

    def all_same_class(examples):
        "Are all these examples in the same target class?"
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        "Choose the attribute with the highest information gain."
        return argmax_random_tie(attrs,
                                 lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        "Return the expected reduction in entropy from splitting by attr."
        def I(examples):
            return information_content([count(target, v, examples)
                                        for v in ['Yes','No']])
        N = float(len(examples))
        remainder = sum((len(examples_i) / N) * I(examples_i) # = (% of examples with that value * I(all those examples)
                        for (v, examples_i) in split_by(attr, examples))
        return I(examples) - remainder

    def split_by(attr, examples):
        
        "Return a list of (val, examples) pairs for each val of attr."
        
        yes_list = []
        no_list = []
        for x in examples:
            if x[attr_map[attr].keys()[0]] == attr_map[attr].values()[0]:
                yes_list.append(x)
            else:
                no_list.append(x)
            
        list = []
        list.append(('Yes',yes_list))
        list.append(('No',no_list))

        return list
    
    return decision_tree_learning(dataset.examples, new_attrs, attr_map)

def information_content(values):
    "Number of bits to represent the probability distribution in values."
    probabilities = normalize(removeall(0, values))
    return sum(-p * log2(p) for p in probabilities)

#______________________________________________________________________________

def ContinuousBinaryDecisionTreeLearner(dataset):
    "[Fig. 18.5]"
    
    # Construct new attributes list and accompanying mapping

    target, values = dataset.target, dataset.values
    
    #Create our new data structures
    orig_attrs = dataset.inputs
    new_attrs = []
    attr_map = dict()
    
    num_count = 0
    for a in dataset.inputs:
        #get rid of any ?s
        values[a] = filter(lambda b: b != '?', values[a])
        #check for type
        if isinstance(values[a][0], str):
            pass #print "STRING!"
        elif isinstance(values[a][0], int):
            values[a] = _create_intervals(map(float, values[a]))
        elif isinstance(values[a][0], float):
            values[a] = _create_intervals(values[a])
        for v in values[a]:
            attr_map[num_count] = {a:v} 
            new_attrs.append(num_count)
            num_count+=1
                
    def decision_tree_learning(examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return plurality_value(parent_examples)
        elif all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        elif len(attrs) == 0:
            return plurality_value(examples)
        else:
            A = choose_attribute(attrs, examples)
            tree = BinaryDecisionFork(A, dataset.attrnames[attr_map[A].keys()[0]])
            tree.update_attr_map(attr_map)
            for (v_k, exs) in split_by(A, examples):
                subtree = decision_tree_learning(
                    exs, removeall(A, attrs), examples)
                tree.add(v_k, subtree)
            return tree

    def plurality_value(examples):
        """Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality.)"""
        popular = argmax_random_tie(values[target],
                                    lambda v: count(target, v, examples))
        return DecisionLeaf(popular)

    def count(attr, val, examples):
        if (val == "Yes"): # corresponds to 'Yes' branch - does value fit boolean attribute?
            return count_if(lambda e: e[attr_map[attr].keys()[0]] == attr_map[attr].values()[0], examples)
        else:
            return count_if(lambda e: e[attr_map[attr].keys()[0]] != attr_map[attr].values()[0], examples)

    def all_same_class(examples):
        "Are all these examples in the same target class?"
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        "Choose the attribute with the highest information gain."
        return argmax_random_tie(attrs,
                                 lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        "Return the expected reduction in entropy from splitting by attr."
        def I(examples):
            return information_content([count(target, v, examples)
                                        for v in ['Yes','No']])
        N = float(len(examples))
        remainder = sum((len(examples_i) / N) * I(examples_i) # = (% of examples with that value * I(all those examples)
                        for (v, examples_i) in split_by(attr, examples))
        return I(examples) - remainder

    def split_by(attr, examples):
        
        "Return a list of (val, examples) pairs for each val of attr."
        
        yes_list = []
        no_list = []
        for x in examples:
            #first type check
            if isinstance(attr_map[attr].values()[0], dict):
                #then check if is interval
                max = attr_map[attr].values()[0]['max']
                min = attr_map[attr].values()[0]['min']
                if x[attr_map[attr].keys()[0]] <= max and x[attr_map[attr].keys()[0]] > min:
                    yes_list.append(x)
                else:
                    no_list.append(x)
            else:
                #check that string value matches
                if x[attr_map[attr].keys()[0]] == attr_map[attr].values()[0]:
                    yes_list.append(x)
                else:
                    no_list.append(x)
        list = []
        list.append(('Yes',yes_list))
        list.append(('No',no_list))

        return list
    
    return decision_tree_learning(dataset.examples, new_attrs, attr_map)

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


    ''' Non-boosted or bagged'''

    # make the decision tree
    dtLearner = DecisionTreeLearner(ds)
    #dtLearner.display()

    # normal training error
    results['normal_training_error'] = test(DecisionTreeLearner(ds),ds)
    
    # normal testing error
    results['normal_testing_error'] = cross_validation(DecisionTreeLearner, ds)

    print "Running cross validation for NormalLearner... %s" % str(results['normal_testing_error'])


    ''' Boosted '''

    #returns train
    adaLearner = AdaBoost(WeightedLearner(DecisionTreeLearner), rounds)

    # boosted training error
    results['boosted_training_error'] = test(adaLearner(ds),ds)

    # boosted testing error
    results['boosted_testing_error'] = cross_validation(adaLearner, ds)
    
    print "Running cross validation for AdaBoostNormalLearner... %s \n" % str(results['boosted_testing_error'])


    ''' Bagged '''

    # make decision tree
    baggedLearner = create_bagging_learners(DecisionTreeLearner, ds, rounds, len(ds.examples))

    results['bagged_training_error'] = test(baggedLearner(ds), ds)

    results['bagged_testing_error'] = cross_validation(baggedLearner, ds)

    print "Running cross validation for NormalBaggedLearner... %s \n" % str(results['bagged_testing_error'])

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
        

    ''' Normal'''

    # make the decision tree
    stumpDTLearner = DecisionTreeStump(ds)
    #dtLearner.display()

    results = {}

    results['k'] = rounds

    # normal training error
    results['normal_training_error'] = test(DecisionTreeStump(ds),ds)
    
    # normal testing error
    results['normal_testing_error'] = cross_validation(DecisionTreeStump, ds)

    print "Running cross validation for NormalStumpLearner... %s" % str(results['normal_testing_error'])


    ''' Boosted '''

    #returns train
    adaLearner = AdaBoost(WeightedLearner(DecisionTreeStump), rounds)

    # boosted training error
    results['boosted_training_error'] = test(adaLearner(ds),ds)

    # boosted testing error
    results['boosted_testing_error'] = cross_validation(adaLearner, ds)
    
    print "Running cross validation for AdaBoostNormalStumpLearner... %s \n" % str(results['boosted_testing_error'])

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
        return get_mode(predictor(example) for predictor in bagged_learners)

    return predict

    

def get_mode(values):
    # Takes a list of classifications
    # Returns majority
    
    totals = defaultdict(int)
    for v in values:
        totals[v] += 1
    return max(totals.keys(), key=totals.get)



# def testBinaryLearner(data, rounds):
#     if data == "restaurant": 
#         attributeNames = "Alternate, Bar, Fri/Sat, Hungry, Patrons, Price, Raining, Reservation, Type, WaitEstimate, WillWait"
#         ds = DataSet(name='../data/restaurant', attrnames=attributeNames)
#     elif data == "cancer":
#         attributeNames = "Class, age, menopause, tumor-size, inv-nodes, node-caps, deg-malig, breast, breast-quad, irradiat"
#         ds = DataSet(name='../breast-cancer/breast-cancer', attrnames=attributeNames)
#     elif data == "new_bands":
#         attributeNames = "timestamp, cylinder_number, customer, job_number, paper_type, ink_type, paper_mill_location, plating_tank, proof_cut, viscosity, caliper, ink_temperature, humifity, roughness, blade_pressure, varnish_pct, press_speed, ink_pct, band_type"
#         ds = DataSet(name='../data/trimmed_bands', attrnames=attributeNames)
    
#     # make the decision tree
#     dtLearner = BinaryDecisionTreeLearner(ds)
#     #dtLearner.display()
    
#     print "\n Training Error: "
#     print test(BinaryDecisionTreeLearner(ds),ds)

#     cross_validation_result = cross_validation(BinaryDecisionTreeLearner, ds)
    
#     print "Running cross validation for BinaryLearner... %s" % str(cross_validation_result)

#     #returns train
#     adaLearner = AdaBoost(WeightedLearner(BinaryDecisionTreeLearner), rounds)

#     print "\n Training Error: "
#     print test(adaLearner(ds),ds)

#     print "\n Test Error:"

#     cross_validation_result = cross_validation(adaLearner, ds)

#     print "\n"
    
#     print "Running cross validation for AdaBoostBinaryLearner... %s \n" % str(cross_validation_result)
    
# def testContinuousBinaryLearner(data):
#     if data == "restaurant": 
#         attributeNames = "Alternate, Bar, Fri/Sat, Hungry, Patrons, Price, Raining, Reservation, Type, WaitEstimate, WillWait"
#         ds = DataSet(name='../data/restaurant', attrnames=attributeNames)
#     elif data == "cancer":
#         attributeNames = "Class, age, menopause, tumor-size, inv-nodes, node-caps, deg-malig, breast, breast-quad, irradiat"
#         ds = DataSet(name='../breast-cancer/breast-cancer', attrnames=attributeNames)
#     elif data == "bands":
#         attributeNames = "timestamp, cylinder_number, customer, job_number, grain_screened, ink_color, proof_on_ctd_ink, blade_mfg, cylinder_division, paper_type, ink_type, direct_steam, solvent_type, type_on_cylinder, press_type, press, unit_number, cylinder_size, paper_mill_location, plating_tank, proof_cut, viscosity, caliper, ink_temperature, humifity, roughness, blade_pressure, varnish_pct, press_speed, ink_pct, solvent_pct, ESA_Voltage, ESA_Amperage, wax, hardener, roller_durometer, current_density, anode_space_ratio, chrome_content, band_type"
#         ds = DataSet(name='../data/bands', attrnames=attributeNames)
#     elif data == "new_bands":
#         attributeNames = "timestamp, cylinder_number, customer, job_number, paper_type, ink_type, paper_mill_location, plating_tank, proof_cut, viscosity, caliper, ink_temperature, humifity, roughness, blade_pressure, varnish_pct, press_speed, ink_pct, band_type"
#         ds = DataSet(name='../data/trimmed_bands', attrnames=attributeNames)
    
#     dtLearner = ContinuousBinaryDecisionTreeLearner(ds)
#     #dtLearner.display()
    
#     cross_validation_result = cross_validation(ContinuousBinaryDecisionTreeLearner, ds) 
#     print "Running cross validation for ContinuousBinaryLearner... %s" % str(cross_validation_result)


# def _create_intervals(values_list):
    
#     intervals = []
#     values_list = sorted(values_list)
#     num_intervals = 12
#     if len(values_list) < num_intervals:
#         last = float("-inf")
#         for v in values_list:
#             intervals.append({'min':last, 'max':v})
#             last = v
#         intervals.append({'min':v, 'max':float("inf")})
#     else:
#         last = values_list[0]
#         intervals.append({'min':float("-inf"), 'max':last})
#         for i in range(0, num_intervals):
#             next = values_list[int(math.ceil((len(values_list) - 1) / num_intervals) * (i + 1))]
#             intervals.append({'min':last, 'max':next})
#             last = next
#             # may have to refine intervals
            
#         intervals.append({'min':last, 'max':float("inf")})
        
#     #print "values_list" + str(values_list)
#     #print "intervals" + str(intervals)
#     return intervals
    
def average_results(results_dict):
    # returns list of avg results for a given num_rounds
    
    normal_testing_avg = sum(results_dict[j]['normal_testing_error'] for j in range(10)) / 10
    boosted_testing_avg = sum(results_dict[j]['boosted_testing_error'] for j in range(10)) / 10
    normal_training_avg = sum(results_dict[j]['normal_training_error'] for j in range(10)) / 10
    boosted_training_avg = sum(results_dict[j]['boosted_training_error'] for j in range(10)) / 10

    return [normal_testing_avg, boosted_testing_avg, normal_training_avg, boosted_training_avg]


def compose_results(avg_results_dict):
    # returns dict where each result type is key and values are a list of each kth result
    
    plot_dict = {'normal_testing_avg': [],
                 'boosted_testing_avg': [],
                 'normal_training_avg': [],
                 'boosted_training_avg': []}

    for k,v in avg_results_dict.items():
        plot_dict['normal_testing_avg'].append(v[0])
        plot_dict['boosted_testing_avg'].append(v[1])
        plot_dict['normal_training_avg'].append(v[2])
        plot_dict['boosted_training_avg'].append(v[3])

    return plot_dict
    
if __name__ == "__main__":

    num_rounds = [1,2,3,5,10,15,20,25,30,40,50]

    # stores the averaged results
    restaurant_avg_results = {}
    restaurant_stump_avg_results = {}
    cancer_avg_results = {}
    cancer_stump_avg_results = {}

    for k in num_rounds:

        print "K = %d" % k

        # stores the results for each j in 0-9
        restaurant_results = {}
        restaurant_stump_results = {}
        cancer_results = {}
        cancer_stump_results = {}

        for j in range(10):

            print "\nRESTAURANT: \n"
            restaurant_results[j] = testNaryLearner("restaurant", int(k))
            restaurant_stump_results[j] = testNaryStumpLearner("restaurant", k)
            # testBinaryLearner("restaurant", int(i))

            print "\nCANCER: \n"
            cancer_results[j] = testNaryLearner("cancer", int(k))
            cancer_stump_results[j] = testNaryStumpLearner("cancer", k)
            # testBinaryLearner("cancer", int(i))
        

        # averages results for this n-round experiment
        restaurant_avg_results[k] = average_results(restaurant_results)
        restaurant_stump_avg_results[k] = average_results(restaurant_stump_results)
        cancer_avg_results[k] = average_results(cancer_results)
        cancer_stump_avg_results[k] = average_results(cancer_stump_results)

    # compose results into plot format
    rest_plot_dict = compose_results(restaurant_avg_results)
    rest_stump_plot_dict = compose_results(restaurant_stump_avg_results)
    canc_plot_dict = compose_results(cancer_avg_results)
    canc_stump_plot_dict = compose_results(cancer_stump_avg_results)

    print "rest_plot_dict: ", rest_plot_dict
    print "canc_plot_dict: ", canc_plot_dict

    def one_minus(L): return [1-x for x in L]

    # plot averages
    X = num_rounds
    for Y in canc_plot_dict.values():
        plt.plot(X, Y)
    plt.show()

    

