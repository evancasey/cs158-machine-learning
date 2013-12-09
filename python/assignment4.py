from time import time
import numpy as np
import csv
import pdb

from learning import *
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import pdb


def fetchData():
    ''' Read in the dataset into a ds object '''

    ds = DataSet(name="../data/semeion")
    return ds


def createKMeansLearner(ds,num_clusters=10):
    ''' Return a kmeans learner trained on ds'''

    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    examples = [row[:-1] for row in ds.examples]
    norm_examples = scale(examples).tolist()    
    kmeans.fit(norm_examples)

    cluster_output_map = getClusterOutputMap(ds,kmeans)

    def predict(obs):        
        cluster = kmeans.predict(obs[:-1])[0]   
        return cluster_output_map[cluster]

    return predict

def createPCAKMeansLearner(ds,num_clusters=10):
    ''' Return a kmeans learner with dimensions reduced to 2 '''

    #take out output column and scale
    examples = [row[:-1] for row in ds.examples]
    norm_examples = scale(examples).tolist()    

    #reduce dims with PCA and fit our learner
    reduced_data = PCA(n_components=2).fit_transform(norm_examples)
    pca_kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    pca_kmeans.fit(reduced_data)

    cluster_output_map = getClusterOutputMap(ds,pca_kmeans)

    def predict(obs):
        cluster = pca_kmeans.predict(obs[:-1])[0]   
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


def testKMeansLearner(ds,learner):
    ''' Takes a kmeans learner and test data and 
        runs cross validation '''

    if learner == createPCAKMeansLearner:
        
        examples = [row[:-1] for row in ds.examples]
        norm_examples = scale(examples).tolist()    
        reduced_data = PCA(n_components=2).fit_transform(norm_examples)

        reduced_data_with_output = [reduced_data.tolist()[i]+[row[-1]] for i,row in enumerate(ds.examples)]        

        ds.examples = reduced_data_with_output
        ds.target = 2
        ds.inputs = range(0,2)

    return cross_validation(learner, ds) 


def plotPCAKMeansLearner(ds,learner):
    ''' Takes a kmeans learner and ds, outputs
        a virinoi diagram with each input point plotted '''

    #dimension reduction steps
    examples = [row[:-1] for row in ds.examples]
    norm_examples = scale(examples).tolist()    
    reduced_data = PCA(n_components=2).fit_transform(norm_examples)
    reduced_data_with_output = [reduced_data.tolist()[i]+[row[-1]] for i,row in enumerate(ds.examples)]        

    #instantiate kmeans object
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    #step size of the mesh
    h = .02     

    #plot the decision boundary
    x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    #obtain labels for each point in mesh. Use last trained model
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    #put result into color plot
    Z = Z.reshape(xx.shape)
    pl.figure(1)
    pl.clf()
    pl.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap=pl.cm.Paired,
              aspect='auto', origin='lower')

    pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    pl.scatter(centroids[:, 0], centroids[:, 1],
               marker='x', s=169, linewidths=3,
               color='w', zorder=10)
    pl.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
             'Centroids are marked with white cross')
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.xticks(())
    pl.yticks(())
    pl.show()


if __name__ == '__main__':

    data = fetchData()

    # results = testKMeansLearner(data,createKMeansLearner)
    # print "Normal Kmeans: ", results

    results = testKMeansLearner(data,createPCAKMeansLearner)
    print "PCA Kmeans: ", results

    plotPCAKMeansLearner(data,createPCAKMeansLearner)

