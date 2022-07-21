import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def eucliDist(A,B): 
    return np.sqrt(sum(np.power((A - B), 2)))


data = np.load('data_n_feature.npy')
data_n = np.load('data_n.npy')


# Balanced dataset

def data_k_means(n):
    kmeans = KMeans(n_clusters=n, random_state=123).fit(data)

    index = np.array([])
    for i in range(n):
        D = []
        for x in np.where(kmeans.labels_ == i)[0]:
            d = eucliDist(kmeans.cluster_centers_[i], data[x])
            D.append(d)
        D = np.array(D)
        u = np.concatenate((np.where(kmeans.labels_ == i)[0][:,np.newaxis],D[:,np.newaxis]),axis=1)
        v = u[np.argsort(u[:,1])][:int(u.shape[0]/47), 0]

        index = np.concatenate((index,v), axis=0)

    data_n11 = []
    for i in index:
        data_n11.append(data_n[int(i)].tolist())

    np.save('data_n_{}_11'.format(str(n)),np.array(data_n11).T)


# Imbalanced dataset

def data_k_means(n, r):
    kmeans = KMeans(n_clusters=n, random_state=123).fit(data)

    index = np.array([])
    for i in range(n):
        D = []
        for x in np.where(kmeans.labels_ == i)[0]:
            d = eucliDist(kmeans.cluster_centers_[i], data[x]) #距离d
            D.append(d)
        D = np.array(D)
        u = np.concatenate((np.where(kmeans.labels_ == i)[0][:,np.newaxis],D[:,np.newaxis]),axis=1)
        v = u[np.argsort(u[:,1])][:int(u.shape[0]/r), 0]

        index = np.concatenate((index,v), axis=0)

    data_n11 = []
    for i in index:
        data_n11.append(data_n[int(i)].tolist())

    np.save('data_n_{}_{}'.format(str(n),str(r)),np.array(data_n11).T)