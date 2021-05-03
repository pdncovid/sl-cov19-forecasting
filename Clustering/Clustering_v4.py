#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[73]:


import os, sys
import time

# machine learning
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

# data manipulation and signal processing

import time
import warnings
import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.manifold import TSNE

from sklearn.preprocessing import LabelEncoder
from sklearn import cluster

import math
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import scipy.stats as ss
import argparse

# plots
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import folium

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
from sklearn.decomposition import PCA 
import matplotlib

font = {'size'   : 11}
matplotlib.rc('font', **font)

# path = "/content/drive/Shareddrives/covid.eng.pdn.ac.lk/COVID-AI (PG)/spatio_temporal/Covid19_DL_Forecasting_Codes"
# os.chdir(path) 
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from functions import split_into_pieces_inorder,split_into_pieces_random,create_dataset_random, distance, convert_lon_lat_to_adjacency_matrix, plot_prediction
from data_loader import load_data, per_million, get_daily
from smoothing_functions import O_LPF,NO_LPF,O_NDA,NO_NDA


parser = argparse.ArgumentParser(description='COVID-AI')
parser.add_argument('--dataset', metavar='D', type=str, default="Sri Lanka", help='Dataset that is going to load (Sri Lanka/Texas/USA/Global)')
parser.add_argument('--ds_type', metavar='T', type=str, default="cumulative", help='Dataset type (daily/cumulative)')
parser.add_argument('--scaler', metavar='S', type=str, default="original", help='scaler type (original/standard/minmax)')
parser.add_argument('--seglen', metavar='L', type=int, default=30, help='Segment length')
parser.add_argument('--monte', metavar='M', type=int, default=50, help='Number of monte-carlo simulations')

parser.add_argument('--regionwise', default=False, action='store_true', help="Scale data region wise if True. Otherwise use global scaling")
parser.add_argument('--per_million', default=False, action='store_true', help="Use cases per million")
parser.add_argument('--view', default=False, action='store_true', help="View plots")
args = parser.parse_args()


is_daily_data = args.ds_type == "daily"
is_per_million = args.per_million
is_regionwise = args.regionwise
DATASET = args.dataset
scaler_type = args.scaler
seg_length = args.seglen
view = args.view
MC_all = args.monte

foldername = f"{args.ds_type}_{scaler_type}_{'permil' if is_per_million else 'notpermil'}_{'regionwise' if is_regionwise else 'global'}_{seg_length}_{MC_all}"
print(foldername)

algo_params = {'quantile': .4,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 4,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

def finalize_figure(figname):
    if view:
        plt.show()
    else:
        from pathlib import Path
        Path(f"logs/{foldername}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"logs/{foldername}/{figname}")
        plt.close('all')

def fixed_aspect_ratio(ax,ratio):
    '''
    Set a fixed aspect ratio on matplotlib plots 
    regardless of axis units
    '''
    xvals,yvals = ax.get_xlim(),ax.get_ylim()

    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    ax.set_aspect(ratio*(xrange/yrange), adjustable='box')


def preprocess(X, population):
    # Xf = O_LPF(X, datatype='daily', order=1, R_cons=1, EIG_cons=1, corr = True)
    # Xf = O_LPF(X, datatype='daily', order=3, R_weight=1.0, EIG_weight=1, corr = True, region_names=region_names, plot_freq=1000)
    # Xf = NO_LPF(X)
    # Xf = O_NDA(X)
    # Xf = NO_NDA(X)

    Xpm = per_million(X,population)
    # Xfpm = per_million(Xf,population)
    
    return X,Xpm #X, Xf, Xpm, Xfpm

def my_imshow(arr,ax=None):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(arr)
    ax.set_yticks(())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im, cax=cax)

def onehot(x, n_classes):
    b = np.zeros((x.size, n_classes+1))
    b[np.arange(x.size),x] = 1
    return b

def metric(yt,yp,n_classes):
    yt = (yt+1)/n_classes
    yp = (yp+1)/n_classes
    # return sum(yt!=yp)
    # return sum(-yt*np.log(yp))
    # return np.abs(yp - yt*np.log(yp))
    return np.mean((yt-yp)**2)

    yt = onehot(yt,n_classes)
    yp = onehot(yp,n_classes)
    kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE) # loss = y_true * log(y_true / y_pred)
    return kl(yt,yp).numpy().sum()
    # return yt * np.log(yt / yp)
    
def compare(lt,lp, n_clusters, permute_both=True):
    """
    lt : 1D array of labels (true)
    lp : 1D array of labels (predicted)
    n_clusters: number of clusters in true labels
    """
    lp = lp - np.min(lp)
    best = 1e100
    for mapper in list(itertools.permutations(np.arange(n_clusters))):
        cost = metric(np.array(mapper)[lt], lp, n_clusters)
        best = min(best, cost)
    if permute_both:
        for mapper in list(itertools.permutations(np.arange(n_clusters))):
            cost = metric(lt, np.array(mapper)[lp], n_clusters)
            best = min(best, cost)
    return best


d = load_data(DATASET,path="../Datasets")
region_names=d["region_names"] 
confirmed_cases=d["confirmed_cases"] 
daily_cases=d["daily_cases"] 
features=d["features"] 
START_DATE=d["START_DATE"] 
n_regions=d["n_regions"] 

lat = features["Lat"].values
lon = features["Lon"].values

features=features.drop(['Lat', 'Lon'], axis=1)

population = features["Population"]
for i in range(len(population)):
    print("{:.2f}%".format(confirmed_cases[i,:].max()/population[i]*100), region_names[i])

days = confirmed_cases.shape[1]
print(f"Total population {population.sum()/1e6:.2f}M, regions:{n_regions}, days:{days}")


if is_daily_data:
    ds, dspm = preprocess(daily_cases, population)
else:
    ds, dspm = preprocess(confirmed_cases, population)


if is_per_million:
    epicurve_data = np.copy(dspm)
else:
    epicurve_data = np.copy(ds)

segments = np.int(np.floor(epicurve_data.shape[1]/seg_length))
epicurve_seg = np.zeros([segments, epicurve_data.shape[0], seg_length])

for i in range(segments):
    epicurve_seg[i,:,:] = epicurve_data[:,i*seg_length:(i+1)*seg_length]

print(f"Number of segments: {segments}\n Segment length: {seg_length}\n Final shape: {epicurve_seg.shape}")



# ## Select a good scaler


if scaler_type == "original":
    scaler = None
if scaler_type == "standard":
    scaler = StandardScaler()
if scaler_type == "minmax":
    scaler = MinMaxScaler()

pca = PCA(n_components = 2) 



data_to_cluster_names = ["Dec", "Jan","Feb","Mar"]
data_to_cluster = []
for i in range(len(epicurve_seg)):
    data_to_cluster.append(epicurve_seg[i])
    
features_fillna = features.fillna(0).values
for i in range(features_fillna.shape[1]):
    data_to_cluster.append(features_fillna[:,i].reshape((-1,1)))
    data_to_cluster_names.append(features.columns[i])

X_original = data_to_cluster

X_scaled=[]
X_normalized=[]
X_two_PC = []

fig, axes = plt.subplots(nrows=len(X_original), ncols=4, figsize=(15, 25))
for i in range(len(X_original)):
    if scaler == None:
        X_scaled.append(X_original[i])  
    else:
        if is_regionwise:
            X_scaled.append(scaler.fit_transform(X_original[i].T).T)
        else:
            shape = X_original[i].shape
            X_scaled.append(scaler.fit_transform(X_original[i].reshape(-1,1)).reshape(shape))
    if is_regionwise:
        X_normalized.append(normalize(X_scaled[-1].T).T)
    else:
        shape = X_scaled[-1].shape
        X_normalized.append(normalize(X_scaled[-1].reshape(1,-1)).reshape(shape))

    
    if X_original[i].shape[1]!=1:
        X_principal = pca.fit_transform(X_normalized[-1]) 
        X_principal = pd.DataFrame(X_principal) 
        X_principal.columns = ['P1', 'P2'] 
        X_two_PC.append(X_principal)
    else:
        X_principal = pd.concat([pd.Series(X_original[i].reshape(-1)),pd.Series(X_original[i].reshape(-1))],axis=1) 
        X_principal.columns = ['P1', 'P2'] 
        X_two_PC.append(X_principal)
#     X_embedded = TSNE(n_components=2).fit_transform(X)

    ax = axes[i,0]
    ax.set_xlabel("Days", axes=ax),ax.set_ylabel("District", axes=ax),ax.set_title("Original")
    my_imshow(X_original[i], ax)

    ax = axes[i,1]
    ax.set_xlabel("Days", axes=ax),ax.set_ylabel("District", axes=ax),ax.set_title("Scaled")
    my_imshow(X_scaled[-1],ax)

    ax = axes[i,2]
    ax.set_xlabel("Days", axes=ax),ax.set_ylabel("District", axes=ax),ax.set_title("Scaled+Normalized")
    my_imshow(X_normalized[-1],ax)

    ax = axes[i,3]
    ax.set_xlabel("PC", axes=ax),ax.set_ylabel("District", axes=ax),ax.set_title("Original1st two PC (S+N)")
    my_imshow(X_two_PC[-1],ax)

finalize_figure("scaling.jpg")

X_cluster = X_scaled

# ============
# Set up cluster parameters
# ============




labels = []

for i_dataset, X_i in enumerate(X_cluster):
    # update parameters with dataset-specific values
    params = algo_params.copy()

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X_i, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X_i, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    km = cluster.KMeans(n_clusters = params['n_clusters'], init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    
    ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
    average_linkage = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage="average", affinity="cityblock", connectivity=connectivity)
    
    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    optics = cluster.OPTICS(min_samples=params['min_samples'], xi=params['xi'], min_cluster_size=params['min_cluster_size'])
    
    affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    
    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('KMeans', km),
        # ('MBKMeans', two_means),
#         ('AffinityProp', affinity_propagation),
#         ('MeanShift', ms),
        ('Spectral', spectral),
#         ('Ward', ward),
#         ('Agglomerative', average_linkage),
#         ('DBSCAN', dbscan),
#         ('OPTICS', optics),
#         ('Birch', birch),
#         ('GMM', gmm)
    )
    

    labels.append([])
    i_algo = -1
    for name, algorithm in clustering_algorithms:
        i_algo += 1
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X_i)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X_i)
        labels[-1].append(y_pred)
labels = np.array(labels)
n_months = labels.shape[0]-features_fillna.shape[1]



# ================= Plotting clusters created by several algorithms using covid 19 cases in each month

fig, axes = plt.subplots(nrows=n_months, ncols=len(clustering_algorithms), figsize=(15, 25))
plt.subplots_adjust(left=.02, right=.96, bottom=.001, top=.96, wspace=.00, hspace=.00)
plot_num = 1
for i in range(n_months):
    for a in range(len(clustering_algorithms)):
        # creating DataFrame from results to plot
        tmp = pd.DataFrame()
        tmp["Code"] = features.index.values
        tmp["Lon"] = lon
        tmp["Lat"] = lat
        tmp["Clu"] = labels[i][a]
        tmp["P1"] = X_two_PC[i]["P1"]
        tmp["P2"] = X_two_PC[i]["P2"]
        tmp.set_index("Code")
          
        if len(tmp["Clu"].unique()) == 1:
            palette = None
        else:
            palette = sns.color_palette("Spectral", as_cmap=True) 

        try:
            ax = axes[i,a]
            sns.scatterplot(data=tmp, x='Lon', y='Lat', hue='Clu',  legend="none", palette=palette,ax=ax) 
#             sns.scatterplot(data=tmp, x='P2', y='P1', hue='Clu',  legend="none", palette=palette,ax=ax)    
        except:
            pass
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlabel("")
        ax.set_ylabel("")
        fixed_aspect_ratio(ax,1.6)
        plot_num += 1

pad = 5
cols = [x[0] for x in clustering_algorithms]
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for ax, row in zip(axes[:,0], data_to_cluster_names[:n_months]):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
finalize_figure("clusters_months.jpg")

# ================= Plotting clusters created by several algorithms using demographics

fig, axes = plt.subplots(nrows=len(features.columns), ncols=len(clustering_algorithms), figsize=(15, 25))
plt.subplots_adjust(left=.02, right=.96, bottom=.001, top=.96, wspace=.00, hspace=.00)

for i in range(len(features.columns)):
    for a in range(len(clustering_algorithms)):
        # creating DataFrame from results to plot
        tmp = pd.DataFrame()
        tmp["Code"] = features.index.values
        tmp["Lon"] = lon
        tmp["Lat"] = lat
        tmp["Clu"] = labels[n_months+i][a]
        tmp["P1"] = X_two_PC[i]["P1"]
        tmp["P2"] = X_two_PC[i]["P2"]
        tmp.set_index("Code")
          
        if len(tmp["Clu"].unique()) == 1:
            palette = None
        else:
            palette = sns.color_palette("Spectral", as_cmap=True) 

        try:
            ax = axes[i,a]
            sns.scatterplot(data=tmp, x='Lon', y='Lat', hue='Clu',  legend="none", palette=palette,ax=ax) 
#             sns.scatterplot(data=tmp, x='P2', y='P1', hue='Clu',  legend="none", palette=palette,ax=ax)    
        except:
            pass
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlabel("")
        ax.set_ylabel("")
        fixed_aspect_ratio(ax,1.6)
        plot_num += 1

pad = 5
cols = [x[0] for x in clustering_algorithms]
for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
for ax, row in zip(axes[:,0], data_to_cluster_names[n_months:]):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
finalize_figure("clusters_demographics.jpg")





# # Compare labels

# ## Algorithms v Demographics

all_metrics = []
all_metrics_MC = []

for j in range(features_fillna.shape[1]):
    metrics = []
    for month_i in range(n_months):
        metrics.append([])
        for alg_i in range(labels.shape[1]):
            metrics[-1].append(compare(labels[-(features_fillna.shape[1])+j,alg_i], labels[month_i,alg_i], 4))

    metrics = np.array(metrics).squeeze()
    all_metrics.append(metrics)
    
    metrics = []
    for month_i in range(n_months):
        metrics.append([])
        for alg_i in range(labels.shape[1]):
            _metric = []
            for MC in range(MC_all):
                 _metric.append(compare(np.random.randint(0,4,25), labels[month_i,alg_i], 4, permute_both=False))
            metrics[-1].append(np.mean(_metric))
    
    metrics = np.array(metrics).squeeze()
    all_metrics_MC.append(metrics)
    
all_metrics_MC = np.array(all_metrics_MC)
all_metrics = np.array(all_metrics)

for i in  range(all_metrics.shape[1]):
    
    plt.subplots_adjust(left=.52, right=.88, bottom=.2, top=.95, wspace=.05, hspace=.01)
    
    plt.subplot(121)
    plt.imshow(all_metrics[:,i,:], vmin=0, vmax=0.3)
    
    plt.title(data_to_cluster_names[i])
    plt.xlabel('Algorithm')
    plt.ylabel('Feature')
    plt.yticks([i for i in range(len(features.columns))],features.columns)
    plt.xticks([i for i in range(len(clustering_algorithms))],[x[0] for x in clustering_algorithms], rotation=45, ha="right", rotation_mode="anchor")
    
    plt.subplot(122)
    plt.imshow(all_metrics_MC[:,i,:], vmin=0, vmax=0.3)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(data_to_cluster_names[i])
    plt.xlabel('Algorithm')
    
    plt.yticks([])
    plt.xticks([i for i in range(len(clustering_algorithms))],[x[0] for x in clustering_algorithms], rotation=45, ha="right", rotation_mode="anchor")    
    finalize_figure(f"alg_v_feat_{data_to_cluster_names[i]}.jpg")

fig = plt.figure()
ax = fig.add_subplot(111)
plt.subplots_adjust(left=.12, right=.98, bottom=.35, top=.95, wspace=.05, hspace=.01)
X = np.arange(len(features.columns))
w = 0.2
for i in  range(all_metrics.shape[1]):
    ax.bar(X + i*w, all_metrics[:,i,1], width = w, label=data_to_cluster_names[i])
    ax.plot(all_metrics_MC[:,i,1], linestyle='--')
# plt.axhline(y=2, color='r', linestyle='--')
ax.set_title("Clustering Error using Spectral Clustering")
ax.set_xlabel('Feature')
ax.set_ylabel('Error')
ax.set_xticks(X+0.4)
ax.set_xticklabels(features.columns)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.legend()
finalize_figure('clustering error.jpg')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.subplots_adjust(left=.12, right=.98, bottom=.35, top=.95, wspace=.05, hspace=.01)
X = np.arange(len(features.columns))
w = 0.2
for i in  range(all_metrics.shape[1]):
    ax.bar(X + i*w, all_metrics_MC[:,i,1]-all_metrics[:,i,1], width = w, label=data_to_cluster_names[i])
ax.set_title("Clustering Error difference using Spectral Clustering with random labeling")
ax.set_xlabel('Feature')
ax.set_ylabel('Error difference')
ax.set_xticks(X+0.4)
ax.set_xticklabels(features.columns)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.legend()
finalize_figure('clustering error diff.jpg')

# ## Months v Months

# In[71]:


all_metrics = []
all_metrics_MC = []

for month_j in range(n_months):
    metrics = []
    for month_i in range(labels.shape[0]-features_fillna.shape[1]):
        metrics.append([])
        for alg_i in range(labels.shape[1]):
            metrics[-1].append(compare(labels[month_j,alg_i], labels[month_i,alg_i], 4))

    metrics = np.array(metrics).squeeze()
    all_metrics.append(metrics)
    
    metrics = []
    for month_i in range(n_months):
        metrics.append([])
        for alg_i in range(labels.shape[1]):
            _metric = []
            for MC in range(MC_all):
                 _metric.append(compare(np.random.randint(0,4,25), labels[month_i,alg_i], 4))
            metrics[-1].append(np.mean(_metric))
    
    metrics = np.array(metrics).squeeze()
    all_metrics_MC.append(metrics)
    
all_metrics_MC = np.array(all_metrics_MC)
all_metrics = np.array(all_metrics)


# In[72]:

font = {'size'   : 16}
matplotlib.rc('font', **font)

for i in  range(all_metrics.shape[2]):
    
#     plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    
    # plt.subplot(121)
    plt.imshow(all_metrics[:,:,i], vmin=0, vmax=0.3)
    
    plt.title(data_to_cluster_names[i])
    plt.xlabel('Month')
    plt.ylabel('Month')
    plt.title(clustering_algorithms[i][0])
    plt.yticks([i for i in range(n_months)],data_to_cluster_names[:n_months])
    plt.xticks([i for i in range(n_months)],data_to_cluster_names[:n_months], rotation=90)
    
#     plt.subplot(122)
#     plt.imshow(all_metrics_MC[:,:,i], vmin=0, vmax=0.3)
#     plt.xlabel('Month')
# #     plt.ylabel('Month')
#     plt.yticks([])
# #     plt.yticks([i for i in range(n_months)],data_to_cluster_names[:n_months])
#     plt.xticks([i for i in range(n_months)],data_to_cluster_names[:n_months], rotation=90)

    plt.title(clustering_algorithms[i][0])
    plt.colorbar(fraction=0.046, pad=0.04)
    finalize_figure(f"mon_v_mon_{clustering_algorithms[i][0]}.jpg")
    
font = {'size'   : 11}
matplotlib.rc('font', **font)

# # Number of points in each cluster

def get_count(arr):
    tmp = []
    for i in np.unique(arr):
        tmp.append(np.sum(arr==i))
    return tmp



for i in range(n_months):
    fig,ax = plt.subplots()
    count = []
    for a in range(labels.shape[1]):
        count.append(get_count(labels[i,a,:]))

    width = 0
    for tmp in count:
        width = max(width, len(tmp))
    for a in range(len(count)):
        while len(count[a])!=width:
            count[a].append(0)
    count = np.array(count)
    im = ax.imshow(count)

    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(len(clustering_algorithms)))
    ax.set_xticklabels([str(x) for x in np.arange(width)])
    ax.set_yticklabels([x[0] for x in clustering_algorithms])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for x in range(count.shape[0]):
        for y in range(count.shape[1]):
            text = ax.text(y, x, count[x, y], ha="center", va="center", color="r")

    ax.set_title(f"Cluster item count for month: {data_to_cluster_names[i]}")
    fig.tight_layout()
    
    finalize_figure(f"count_{data_to_cluster_names[i]}.jpg")
    



# ========================================================================================== MAP
import branca.colormap as cmp
import io
from PIL import Image

map_labels = labels[:,1,:]
print(map_labels)
n_clusters = 4
for month_i in range(1,n_months):
    for month_j in range(month_i):
        
        
        best = 1e100
        perm = None
        for mapper in list(itertools.permutations(np.arange(n_clusters))):
            lp = map_labels[month_i]
            lt = map_labels[month_j]
            cost = metric(np.array(mapper)[lt], lp, n_clusters)
            if cost < best:
                best = cost
                perm = mapper

        map_labels[month_j] = np.array(perm)[map_labels[month_j]]

for i in range(n_months):

    dpmc = pd.Series(map_labels[i], name="Labels")
    names = pd.Series(features.index, name="District")
    state_data = pd.concat([names, dpmc], 1)
    print(state_data)


    state_geo = "maps/LKA_electrol_districts.geojson"
    step = cmp.StepColormap(
         ['black', 'green', 'blue','red'],
         vmin=0, vmax=3,
         index=[0, 0.5, 1.5, 2.5],  #for change in the colors, not used fr linear
         caption='Color Scale for Map'    #Caption for Color scale or Legend
        )
    m = folium.Map(location=[8,81], zoom_start=7.3)

    label_dict = state_data.set_index('District')['Labels']
    folium.GeoJson(state_geo, name="geojson",
        style_function=lambda feature: {
            'fillColor': step(label_dict[feature['properties']['electoralDistrictCode']]),
            'fillOpacity':0.8,
            'color': 'black',       #border color for the color fills
            'weight': 1,            #how thick the border has to be
            'dashArray': '5, 3'  #dashed lines length,space between them
    }).add_to(m)
    
    # folium.Choropleth(
    #     geo_data=state_geo,
    #     name="choropleth",
    #     data=state_data,
    #     columns=["District", "Labels"],
    #     key_on="properties.electoralDistrictCode",
    #     fill_color="YlGn",
    #     fill_opacity=0.7,
    #     line_opacity=0.2,
    #     legend_name="Mean Covid-19 Cases",
        
    # ).add_to(m)
    
    step.add_to(m)     #adds colorscale or legend

    folium.LayerControl().add_to(m)

    

    img_data = m._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    img.save(f'logs/{foldername}/{data_to_cluster_names[i]}_image.png')

# ===================================================================================================
