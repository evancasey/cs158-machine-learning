from time import time
import numpy as np
import csv
import pdb

from learning import *
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import random
import operator
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import matplotlib.offsetbox as osb
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

    cluster_output_map = defaultdict(dict)
    for row in ds.examples:        
        cluster = learner.predict(row[:-1])[0]       
        count = cluster_output_map[cluster].setdefault(row[-1],0)
        cluster_output_map[cluster][row[-1]] = count + 1        
    
    for k,v in cluster_output_map.items():
        cluster_output_map[k] = max(v.iteritems(), key=operator.itemgetter(1))[0]

    return cluster_output_map


def testKMeansLearner(ds,learner):
    ''' Takes a kmeans learner and test data and 
        runs cross validation '''

    #make copy of original ds
    pca_ds = copy.deepcopy(ds)

    if learner == createPCAKMeansLearner:
        
        examples = [row[:-1] for row in pca_ds.examples]
        norm_examples = scale(examples).tolist()    
        reduced_data = PCA(n_components=2).fit_transform(norm_examples)

        reduced_data_with_output = [reduced_data.tolist()[i]+[row[-1]] for i,row in enumerate(pca_ds.examples)]        

        pca_ds.examples = reduced_data_with_output
        pca_ds.target = 2
        pca_ds.inputs = range(0,2)

    return cross_validation(learner, pca_ds) 


def plotPCAKMeansLearner(ds,num_clusters=10):
    ''' Takes a kmeans learner and ds, outputs
        a virinoi diagram with each input point plotted '''

    #dimension reduction steps
    examples = [row[:-1] for row in ds.examples]
    norm_examples = scale(examples).tolist()    
    reduced_data = PCA(n_components=2).fit_transform(norm_examples)
    reduced_data_with_output = [reduced_data.tolist()[i]+[row[-1]] for i,row in enumerate(ds.examples)]        

    #instantiate and fit kmeans object
    pca_kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    pca_kmeans.fit(reduced_data)

    #get the boundaries of our grid
    x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1

    #create our background grid
    plot = pl.subplot(111)
    plot.patch.set_facecolor('black')
    pl.xlim([x_min,x_max])
    pl.ylim([y_min,y_max])

    #create a dictionary of the cluster to img_list
    img_dict = defaultdict(list)
    for i,row in enumerate(ds.examples):

        #get the cluster
        cluster = pca_kmeans.labels_[i]

        #get the img and img coordinates
        coordinates = reduced_data[i]
        img = np.array(ds.examples[i][:-1]).reshape((16,16))

        #for each cluster append the img to its value
        img_dict[cluster].append((img,coordinates))


    color_map = {0: [1.0,0.0,0.0],
                 1: [0.93,0.57,0.13],
                 2: [1.0,0.8431,0.0],
                 3: [.1961,.8039,.1961],
                 4: [0,1.0,0],
                 5: [0.0,0.8078,.8196],
                 6: [.098,.098,.4392],
                 7: [.6275,.1255,.9412],
                 8: [1.0,.4118,.7059],
                 9: [1.0,1.0,1.0] }

    #plot a subset of images for each cluster
    for cluster,v in img_dict.items():

        #sample 10 images
        img_list = random.sample(v,10)

        #plot each subset 
        for i,img_and_coord in enumerate(v):            
            if i < 10:                

                color_img = img_list[i][0].tolist()

                #change colors
                for j,row in enumerate(color_img):
                    for k,val in enumerate(row):
                        if val == 1.0:
                            color_img[j][k] = color_map[cluster]
                        else:
                            color_img[j][k] = [0.0,0.0,0.0]
                
                osb_img = osb.OffsetImage(np.array(color_img),zoom=1.5)
                xy = img_list[i][1].tolist()
                ab = osb.AnnotationBbox(osb_img,xy,pad=0,xycoords='data')                                  
                plot.add_artist(ab)

    #draw and show the plot
    pl.draw()
    pl.show()

def plotKMeansGrid(ds,num_clusters=10):
    ''' Create a kmeans learner and output a grid containing
        a rep. image and a sample of correct/incorrect images '''

    #fit the learner
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    examples = [row[:-1] for row in ds.examples]
    norm_examples = scale(examples).tolist()       
    kmeans.fit(norm_examples)

    cluster_output_map = getClusterOutputMap(ds,kmeans)

    #create a dict with key as digit and value as images
    img_dict = defaultdict(list)
    for i,row in enumerate(ds.examples):

        #get the cluster
        cluster = kmeans.labels_[i]        

        #for each cluster append the img to its value
        img_dict[cluster].append(np.array(ds.examples[i]))

    cum_img_list = []

    #for each cluster, draw a row on the grid
    for k,v in img_dict.items():
        img_list = []

        #cluster/digit metadata
        img_transform = kmeans.transform([img[:-1] for img in v]).T
        rep_digit = cluster_output_map[k]

        #find rep imgs
        cum_img_list.append(v[min(enumerate(img_transform[k]), key=operator.itemgetter(1))[0]][:-1])

        cor_img_list = []
        incor_img_list = []

        for img in v:
            #find cor imgs
            if int(img[-1]) == rep_digit and len(cor_img_list) < 5:                
                cor_img_list.append(img[:-1])
            #find incor imgs
            elif int(img[-1]) != rep_digit and len(incor_img_list) < 5:
                incor_img_list.append(img[:-1])

        cum_img_list += cor_img_list + incor_img_list
        
    for i,img in enumerate(cum_img_list):

        #create a new subplot for each image
        plot = pl.subplot(10,11,i)
        plot.xaxis.set_visible(False)
        plot.yaxis.set_visible(False)
        pl.xlim([0,16])
        pl.ylim([0,16])    

        #plot rep img
        img_matrix = np.array(np.rot90(img.reshape((16,16)).T))
        plot.imshow(img_matrix)

    #draw and show the plot
    pl.draw()
    pl.show()

if __name__ == '__main__':

    data = fetchData()

    # results = testKMeansLearner(data,createKMeansLearner)
    # print "Normal Kmeans: ", results

    # results = testKMeansLearner(data,createPCAKMeansLearner)
    # print "PCA Kmeans: ", results

    # plotPCAKMeansLearner(data,createPCAKMeansLearner)
    
    plotKMeansGrid(data)

