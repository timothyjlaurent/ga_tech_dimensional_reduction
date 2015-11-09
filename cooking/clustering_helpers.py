from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import time

def run_k_means(X, n_clusters=20):  
    k_means = KMeans(n_clusters=20, n_jobs=8)
    k_means.fit(X)
    return k_means.labels_, kmeans

def run_k_means_ar(ar, n_clusters=20):
    out = []
    for x in ar:
        start = time.time()
        clusters, kmeans = run_k_means(x, n_clusters=20 )
        e = {'prediction': clusters ,
             'model': kmeans,
             'time': time.time() - start}
        print(e)
        out.append( e )
    return out
    
    
def run_maxlikelyhood_ar(ar, n_clusters=20):
    out = []
    for x in ar:
        start = time.time()
        prediction, model = maxlikelyhood_transform(x, n_components=20 )
        e = {'prediction': prediction ,
             'model' : model,
             'time': time.time() - start}
        print(e)
        out.append( e )
    return out
    

def maxlikelyhood_transform(X, n_components=20 ):
    max_likelyhood = GMM(n_components=n_components)
    max_likelyhood.fit(X)
    return max_likelyhood.predict(X), max_likelyhood

def enrich_with_cluster(datasets, clusters):
    out = []
    lb = LabelBinarizer()
    for i, dataset in enumerate(datasets):
        cluster = clusters[i]
        print('dataset')
        print(dataset)
        print(dataset.shape)
        
        print('cluster')
        print(cluster)
        print(cluster.shape)
        cluster_bin = lb.fit_transform(cluster)
        print('cluster-bin')
        print(cluster_bin)
        print(cluster_bin.shape)
        e = np.hstack((dataset, cluster_bin))
        print('output')
        print(e)
        print(e.shape)
        out.append(e)
    return out

def get_predictions(kmeans_ar):
    return [ kmean['prediction'] for kmean in kmeans_ar]


from sknn.mlp import Classifier, Layer  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedKFold, cross_val_score
import time

def run_nn_cv(X, y,n_iter=1, units=300):
    start = time.time()
#     print X.shape[1]
    features = X.shape[1]
    pipeline = Pipeline([
            ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
            ('neural network', Classifier(layers=[Layer("Rectifier", units=units),
                                                  Layer("Rectifier", units=units),
                                                  Layer('Softmax')],
                                         n_iter=n_iter))
            
        ])
    cv = cross_val_score(pipeline, X, y, n_jobs=-1)
    out = {'cv': cv, 'time': time.time()- start}
    print(out)
    return out