from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import sklearn.cluster as cluster
import hw1.utils as utils
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

rand = 10

class G_AgglomerativeClustering(AgglomerativeClustering):
    def predict(self, X):
        return self.fit_predict(X)

class G_DBSCAN(DBSCAN):
    def predict(self, X):
        return self.fit_predict(X)

class G_SpectralClustering(SpectralClustering):
    def predict(self, X):
        return self.fit_predict(X)

def km_grid(X :np.ndarray, y :np.ndarray, n_digits :int, n_iter :int = 1e6) -> RandomizedSearchCV:
    param_dict = {'init' : ['k-means++', 'random'], "algorithm" : ['full', 'elkan']}
    estimator = KMeans(n_clusters=n_digits, n_init=rand)
    grid = utils.grid_search_random(estimator, param_dict, X, y, utils.evaluate_items,n_iter, random_state=rand)

    return grid

def ap_grid(X :np.ndarray, y :np.ndarray, n_digits :int, n_iter :int = 2, max_iter = 10) -> RandomizedSearchCV:
    param_dict = {
        "damping" : np.arange(0.5,1.0,0.1),
        'convergence_iter' : np.linspace(1,30,6).astype(int),
    }
    estimator = AffinityPropagation(max_iter=max_iter)
    grid = utils.grid_search_random(estimator, param_dict, X, y, utils.evaluate_items, n_iter, random_state=10)
    return grid


def ms_grid(X :np.ndarray, y :np.ndarray, n_digits :int, n_iter :int = 2) -> RandomizedSearchCV:
    def bandwidth_list(): #calculate and generate bandwidths
        quantiles = np.linspace(0.1,1.,4)
        bandwidths = []
        for q in quantiles:
            bandwidths.append(cluster.estimate_bandwidth(X, quantile=q,n_samples=None, random_state=rand))

        mean= np.mean(bandwidths)
        bandwidths += [mean*0.1, mean*2, mean*5, mean*10] + list(np.linspace(np.maximum(0.01, mean-10), mean+10, 5))
        return bandwidths

    estimator = MeanShift()
    param_dict = {'bandwidth' : bandwidth_list()}
    grid = utils.grid_search_random(estimator, param_dict, X, y, utils.evaluate_items, n_iter, random_state=rand)
    return grid

def sc_grid(X :np.ndarray, y :np.ndarray, n_digits :int, n_iter :int = 10) -> RandomizedSearchCV:
    param_dict = {
        "assign_labels" : ['kmeans', 'discretize'],
        "affinity" : ['nearest_neighbors', 'rbf'],
        "n_neighbors" : [3,7,10,15,30,50]
    }
    estimator = G_SpectralClustering(n_clusters=n_digits, n_init=rand, random_state=rand)
    grid = utils.grid_search_random(estimator, param_dict, X, y, utils.evaluate_items, n_iter, random_state=rand)
    return grid

def wh_grid(X :np.ndarray, y :np.ndarray, n_digits :int, n_iter :int = 10) -> RandomizedSearchCV:
    param_dict = {"linkage" : ['ward']}
    estimator = G_AgglomerativeClustering(n_clusters=n_digits)
    grid = utils.grid_search_random(estimator, param_dict, X, y, utils.evaluate_items, n_iter=1, random_state=rand)
    return grid

def ag_grid(X :np.ndarray, y :np.ndarray, n_digits :int, n_iter :int = 10) -> RandomizedSearchCV:
    #this function does not include 'ward' of linkage
    param_dict = {
        "affinity" : ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
        "linkage" : ['complete', 'average', 'single']
    }
    estimator = G_AgglomerativeClustering(n_clusters=n_digits)
    grid = utils.grid_search_random(estimator, param_dict, X, y, utils.evaluate_items, n_iter, random_state=rand)
    return grid

def db_grid(X :np.ndarray, y :np.ndarray, n_digits :int, n_iter :int = 20) -> RandomizedSearchCV:
    param_dict = {
        'eps' : [0.2, 0.5, 1, 1.5, 3, 5, 10, 20, 50], # maximum distance between two neighbors
        'min_samples' : [2, 5, 7, 9, 10, 20, 50, 100], # minimal neighborhood number as a core point, include itself
        'metric' : ['euclidean', 'l1', 'l2']
    }

    estimator = G_DBSCAN()
    grid = utils.grid_search_random(estimator, param_dict, X, y, utils.evaluate_items, n_iter, random_state=rand)
    return grid

def gm_grid(X :np.ndarray, y :np.ndarray, n_digits :int, n_iter :int = 20) -> RandomizedSearchCV:
    param_dict = {
        "n_components" : [1,2,3,4,5,8,12],
        "covariance_type" : ['full', 'tied', 'diag', 'spherical'],
        'init_params' : ['kmeans', 'random']
    }
    estimator = GaussianMixture(random_state=rand)
    grid = utils.grid_search_random(estimator, param_dict, X, y, utils.evaluate_items, n_iter, random_state=rand)
    return grid