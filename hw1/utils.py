import sklearn
import numpy as np
from typing import Tuple
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def refit_func(y_true, y_pred):
    result = list(evaluate(y_true, y_pred).values())
    value = np.mean(result)
    return value

evaluate_items = {'nm' : "normalized_mutual_info_score", "homo" : "homogeneity_score", "comp" : "completeness_score",
                  'refit' : sklearn.metrics.make_scorer(refit_func, greater_is_better=True)}


def evaluate(labels :np.ndarray, preds :np.ndarray) -> dict:
    '''
    This is an evaluation function for clustering method.Three metrics are evaluated which are :
     metrics.normalized_mutual_info_score, metrics.homogeneity_score, metrics.completeness_score.
    :param labels: true labels
    :param preds: predicted labels
    :return: three metrics tuple
    '''

    return{
        "normalized_mutual_info" : metrics.normalized_mutual_info_score(labels_true=labels, labels_pred=preds),
        "homogeneity" :metrics.homogeneity_score(labels, preds),
        "completeness" : metrics.completeness_score(labels, preds)
    }

def visualize_PCA(cluster, data :np.ndarray):
    reduced_data = PCA(n_components=2).fit_transform(data)
    cluster.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = cluster.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    '''
    centroids = cluster.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
              '''
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def prepare_data_digits() -> Tuple[np.ndarray, np.ndarray, int]:
    np.random.seed(42)

    digits = load_digits()
    data = scale(digits.data)
    n_digits = len(np.unique(digits.target))
    data, target = shuffle(data, digits.target)

    return data, target, n_digits

def prepare_data_iris() -> Tuple[np.ndarray, np.ndarray, int]:
    np.random.seed(42)

    digits = load_iris()
    data = scale(digits.data)
    categories = len(np.unique(digits.target))
    data, target = shuffle(data, digits.target)

    return data, target, categories

def prepare_data_news(n_feature :int = 4096) -> Tuple[np.ndarray, np.ndarray, int]:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
        'misc.forsale',
        'rec.autos'
    ]

    dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=42, categories=categories)

    y = dataset.target
    true_k = np.unique(y).shape[0]
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_feature, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(dataset.data).toarray()

    return X, y, true_k

def grid_search_random(estimator,param_dict :dict, X :np.ndarray, y :np.ndarray, scoring :dict,n_iter :int,
                       random_state :int, refit :str = 'refit') -> RandomizedSearchCV:

    rgrid = RandomizedSearchCV(estimator=estimator, param_distributions=param_dict, n_iter=n_iter,
                               scoring=scoring, refit=refit, random_state=random_state, )
    rgrid.fit(X=X, y=y)
    return rgrid

def report_csv(grid :RandomizedSearchCV, path :str, filename :str):
    cols = [str('param_' + key) for key in list(grid.param_distributions.keys())]
    cols += ['mean_test_refit', 'mean_test_nm', 'mean_test_comp', 'mean_test_homo', 'mean_fit_time']

    df :pd.DataFrame = pd.DataFrame(grid.cv_results_).sort_values(by='rank_test_refit', axis=0).loc[:, cols]

    file = path+"/"+filename+".csv"

    df.to_csv(file, index=False)

    print("{} has saved".format(file))
