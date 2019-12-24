#%%
import warnings

warnings.filterwarnings('ignore')

import hw1.utils as utils
import hw1.cluster_method as cluster

if __name__ == '__main__':
    datasets = {
        'newsgroups' : utils.prepare_data_news,
        'digit' : utils.prepare_data_digits
    }

    methods = {
        'KMeans' : cluster.km_grid,
        'AffinityPropagation' : cluster.ap_grid,
        'MeanShift' : cluster.ms_grid,
        'SpectralClustering' : cluster.sc_grid,
        'AgglomerativeClustering' : cluster.ag_grid,
        'DBSCAN' : cluster.db_grid,
        'WardHierarchicalClustering' : cluster.wh_grid
    }

    for data_key in datasets.keys():
        data, labels, categories = datasets[data_key]()
        for cluster_key in methods.keys():
            grid = methods[cluster_key](data, labels, categories, n_iter = 20)
            utils.report_csv(grid, "./result", data_key+"_"+cluster_key)