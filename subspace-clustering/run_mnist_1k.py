import statistics
import numpy as np
import sys
import pandas as pd
import scipy.io as sio
from pathlib import Path
import time

from sklearn.metrics import adjusted_mutual_info_score

from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from gen_union_of_subspaces import gen_union_of_subspaces
from metrics.cluster.accuracy import clustering_accuracy
from sklearn import cluster

# =================================================
# Generate dataset where data is drawn from a union of subspaces
# =================================================
ambient_dim = 9
subspace_dim = 5
num_subspaces = 10
num_points_per_subspace = 50

curr_dir = str(Path(__file__).parent)
# load face images and labels
data = sio.loadmat(f'{curr_dir}/../ODSC/Data/mnist-original.mat')

Img = data['data']
Label = data['label']

# # print(Label.shape)
# Img = np.reshape(Img,(Img.shape[0],28,28,1))
# print(Img.shape)
Imgsub = np.zeros([784,1000])
Labelsub = np.zeros([1000])
sub_n = -1
#Create MNIST sub dataset
tmp = 1
for class_n in range(0,10):
    for i in range(0,Label.shape[1]):
        class_num = int(Label[:,i])
        # print(class_num)
        # print(class_num,class_n)
        if class_num == class_n:
            sub_n = sub_n+1
            # print(sub_n, class_n)
            Imgsub[:,sub_n] = Img[:,i]
            Labelsub[sub_n] = Label[:,i] 
            tmp = 0
        if tmp == 0 and (sub_n+1) % 100 == 0 and sub_n!=0:
            tmp=1
            break

Img = Imgsub
Label = Labelsub
Img = np.transpose(Img)
print(Img.shape)

# =================================================
# Create cluster objects
# =================================================

# Baseline: non-subspace clustering methods
model_kmeans = cluster.KMeans(n_clusters=num_subspaces)  # k-means as baseline
model_spectral = cluster.SpectralClustering(n_clusters=num_subspaces,affinity='nearest_neighbors',n_neighbors=5)  # spectral clustering as baseline

# Elastic net subspace clustering with a scalable active support elastic net solver
# You et al., Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
model_ensc = ElasticNetSubspaceClustering(n_clusters=num_subspaces,algorithm='spams',active_support=True,gamma=200,tau=0.9)

# Sparse subspace clusterign by orthogonal matching pursuit (SSC-OMP)
# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=num_subspaces,n_nonzero=subspace_dim,thr=1e-5)

clustering_algorithms = (
    ('KMeans', model_kmeans),
    ('Spectral Clustering', model_spectral),
    ('EnSC', model_ensc),
    ('SSC-OMP', model_ssc_omp)
)
for name, algorithm in clustering_algorithms:
    acc, ami = [], []
    for _ in range(10):
        t_begin = time.time()
        algorithm.fit(Img)
        t_end = time.time()
        pred_labs = algorithm.labels_
        accuracy = clustering_accuracy(Label, pred_labs)
        ami_score = adjusted_mutual_info_score(Label, pred_labs)
        acc.append(accuracy)
        ami.append(ami_score)
    print(f'Algorithm: {name}. Clustering accuracy: {round(statistics.mean(acc), 4)} \u00B1 {round(statistics.stdev(acc), 4)}, \
        AMI : {round(statistics.mean(ami), 4)} \u00B1 {round(statistics.stdev(ami), 4)}. Running time: {t_end - t_begin}')
    # print('Algorithm: {}. Clustering accuracy: {}. Running time: {}'.format(name, accuracy, t_end - t_begin))