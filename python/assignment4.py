from time import time
import numpy as np
import csv
import pdb

from learning import *
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import normalize
import pdb


def fetchData():
    ''' Read in the dataset into a ds object '''

    ds = DataSet(name="../data/semeion")
    return ds


def createKMeansLearner(ds,num_clusters=10):
    ''' Return a kmeans learner trained on ds'''

    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    examples = [row[:-1] for row in ds.examples]
    norm_examples = normalize(examples).tolist()    
    kmeans.fit(norm_examples)

    cluster_output_map = getClusterOutputMap(ds,kmeans)

    def predict(obs):        r
        cluster = kmeans.predict(obs[0:256])[0]   
        return cluster_output_map[cluster]

    return predict

def getClusterOutputMap(ds,learner):
    ''' Call predict on each row in ds, and return a dictionary
     to map cluster to digit value'''

    cluster_output_map = {}
    for row in ds.examples:
        cluster = learner.predict(row[:-1])[0]            
        cluster_output_map[cluster] = row[-1]

    return cluster_output_map


def testKMeansLearner(ds):
    ''' Takes a kmeans learner and test data and 
        runs cross validation '''

    data = fetchData()    
    return cross_validation(createKMeansLearner, ds)    
    

if __name__ == '__main__':

    data = fetchData()
    results = testKMeansLearner(data)

    print "10-Fold Cross Validation: ", results


