import os, glob, shutil, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def min_max_norm(x):

    return (x - x.min() + (1e-30)) / (x.max() - x.min() + (1e-30))

def plot_generation(dict_plot, savepath=""):

    list_key = list(dict_plot.keys())
    list_key.sort()

    plt.figure(figsize=(14, 3))

    for idx_key, name_key in enumerate(list_key):
        plt.subplot(2, 10, idx_key+1)
        plt.imshow(dict_plot[name_key]['y'], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 10, idx_key+1+10)
        plt.imshow(dict_plot[name_key]['y_hat'], cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(savepath, transparent=True)
    plt.close()

def plot_comparison(y, y_hat, savepath=""):

    plt.figure(figsize=(3, 2))

    plt.subplot(1, 2, 1)
    plt.title('Target')
    plt.imshow(y, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.title('Generation')
    plt.imshow(y_hat, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(savepath, transparent=True)
    plt.close()

def plot_scatter(z_0, z_k, savepath=""):

    if(z_0.shape[-1] > 2):
        pca = PCA(n_components=2)
        z_merge = np.append(z_0, z_k, axis=0)
        pca.fit(z_merge)
        z_0 = pca.transform(z_0)
        z_k = pca.transform(z_k)

    fig = plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title('$z_0$')
    plt.scatter(z_0[:, 0], z_0[:, 1])
    plt.xticks([])
    plt.yticks([])
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title('$z_k$')
    plt.scatter(z_k[:, 0], z_k[:, 1])
    plt.xticks([])
    plt.yticks([])
    plt.grid()

    plt.tight_layout()
    plt.savefig(savepath, transparent=True)
    plt.close()
