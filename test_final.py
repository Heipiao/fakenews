import sys
import numpy as np
import os
import re
import nltk
import numpy as np
import spacy
import random

from sklearn import feature_extraction
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from feature_engineering import gen_or_load_feats
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from Semantic_features import semantic
from entity import entity
from sklearn.neural_network import MLPClassifier
from word2vec import word2VecFeature
from glove import gloveFeature


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt




# import gensim
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
# _wnl = nltk.WordNetLemmatizer()
# import spacy
# nlp = spacy.load('en')


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X_entity = gen_or_load_feats(entity, h, b, "features/entity."+name+".npy")
    X_semantic = gen_or_load_feats(semantic, h, b, "features/semantic."+name+".npy")
    X_word2vec = gen_or_load_feats(word2VecFeature, h, b, "features/word2VecFeature."+name+".npy")
    X_gloveFeature = gen_or_load_feats(gloveFeature, h, b, "features/gloveFeature."+name+".npy")
  
    #X = gen_or_load_feats(word2VecFeature, h, b, "features/word2VecFeature."+name+".npy")
    #X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap,X_entity,X_semantic,X_word2vec,X_gloveFeature]
    
    

  
    return X,y





if __name__ == "__main__":
    d = DataSet()
    r = random.Random()
    r.seed(1489215)

    article_ids = list(d.articles.keys())  # get a list of article ids
    r.shuffle(article_ids)  # a

    stances_all = []
    for stance in d.stances:
        if stance['Body ID'] in article_ids:
            stances_all.append(stance)
    
    stances = stances_all 
    dataset = d 
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    Xs = dict()
    ys = dict()

        # Load/Precompute all features now
    # X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    # for fold in fold_stances:
    #     Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))

    Xi = []
    yi = []
    Xi,yi = generate_features(stances_all,d,"all")

    print(len(Xi))
    print(len(yi))
    # best_score = 0
    # best_fold = None


    # #     # Classifier for each fold
    # for fold in stances_all:
    #     ids = list(range(len(folds)))
    #     del ids[fold]

    X_train = np.vstack(tuple([Xi ]))
    y_train = np.hstack(tuple([yi ]))
    #     X_test = Xs[fold]
    #     y_test = ys[fold]
    X_train = np.nan_to_num(X_train)
    #y_train = np.nan_to_num(X_train)
    #     X_test = np.nan_to_num(X_train)
    #     #y_test = np.nan_to_num(X_train)
    #     X_holdout = np.nan_to_num(X_holdout)


    #     #clf = GradientBoostingClassifier(learning_rate = 0.8,n_estimators=1000, random_state=1418, verbose=True)
    #     #clf = RandomForestClassifier(n_estimators=200, random_state=14128, verbose=True)
    #clf = MLPClassifier(hidden_layer_sizes=(300,))
    #clf.fit(X_train, y_train)

   
    X, y = X_train, y_train


    title = "Learning Curves "
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    estimator = MLPClassifier(hidden_layer_sizes=(100,))
    #estimator = RandomForestClassifier(n_estimators=200, random_state=14128, verbose=True)
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)

      

    plt.show()
    #      ############################################
    #     predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
    #     actual = [LABELS[int(a)] for a in y_test]

    #     fold_score, _ = score_submission(actual, predicted)
    #     max_fold_score, _ = score_submission(actual, actual)

    #     score = fold_score/max_fold_score

    #     print("Score for fold "+ str(fold) + " was - " + str(score))
    #     if score > best_score:
    #         best_score = score
    #         best_fold = clf

    #     #Run on Holdout set and report the final score on the holdout set
    # predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    # actual = [LABELS[int(a)] for a in y_holdout]

    # report_score(actual,predicted)
