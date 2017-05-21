import sys
import numpy as np
import os
import re
import nltk
import numpy as np
import spacy

from sklearn import feature_extraction
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from feature_engineering import gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
_wnl = nltk.WordNetLemmatizer()
import spacy
nlp = spacy.load('en')

def semantic(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = makeFeatureVec(clean_headline,clean_body)
        X.append(features)
        print(len(X))
    return X

def makeFeatureVec(title , content):
    # Function to convert title and content in 300
    # dimension vector then get the average value
    # of the vector, the substract title vec and content
    # vec
    # Pre-initialize an empty numpy array (for speed)
    
    N_title_count = 1
    N_content_count = 1
    V_title_count = 1
    V_content_count = 1
    ADJ_title_count = 1
    ADJ_content_count = 1
    # ENT_title_count = 0
    # ENT_content_count = 0
    N_titleVec = np.zeros((300,),dtype="float64")
    N_contentVec = np.zeros((300,),dtype="float64")
    V_titleVec = np.zeros((300,),dtype="float64")
    V_contentVec = np.zeros((300,),dtype="float64")
    ADJ_titleVec = np.zeros((300,),dtype="float64")
    ADJ_contentVec = np.zeros((300,),dtype="float64")
    # ENT_titleVec = np.zeros((300,),dtype="float64")
    # ENT_contentVec = np.zeros((300,),dtype="float64")
    doc_title = nlp(" ".join(title))
    doc_content = nlp(" ".join(content))
    # Loop
    for t in doc_title:
        if(t.pos_ == "NOUN"):
            try:
                 N_titleVec = np.add(N_titleVec,model.word_vec(t.text))
            except KeyError as e:
                continue
            N_title_count += 1
        elif(t.pos_ == "VERB"):
            try:
                 V_titleVec = np.add(V_titleVec,model.word_vec(t.text))
            except KeyError as e:
                continue
            V_title_count += 1
        elif(t.pos_ == "ADJ"):
            try:
                 ADJ_titleVec = np.add(ADJ_titleVec,model.word_vec(t.text))
            except KeyError as e:
                continue
            ADJ_title_count += 1
        

       
    for c in doc_content:
        if(c.pos_ == "NOUN"):
            try:
                 N_contentVec = np.add(N_contentVec,model.word_vec(c.text))
            except KeyError as e:
                continue
            N_content_count += 1
        elif(c.pos_ == "VERB"):
            try:
                 V_contentVec = np.add(V_contentVec,model.word_vec(c.text))
            except KeyError as e:
                continue
            V_content_count += 1
        elif(c.pos_ == "ADJ"):
            try:
                 ADJ_contentVec = np.add(ADJ_contentVec,model.word_vec(c.text))
            except KeyError as e:
                continue
            ADJ_content_count += 1
        
        
    # Divide the result by the number of words to get the average
    #print ("is.inf=", np.where(np.isinf(content_count)))
    N_titleVec = np.divide(N_titleVec,N_title_count)
    N_contentVec = np.divide(N_contentVec,N_content_count)
    V_titleVec = np.divide(V_titleVec,V_title_count)
    V_contentVec = np.divide(V_contentVec,V_content_count)
    ADJ_titleVec = np.divide(ADJ_titleVec,ADJ_title_count)
    ADJ_contentVec = np.divide(ADJ_contentVec,ADJ_content_count)
    # get the feature
    N_featureVec = 1 - spatial.distance.cosine(N_titleVec,N_contentVec)
    V_featureVec = 1 - spatial.distance.cosine(V_titleVec,V_contentVec)
    ADJ_featureVec = cosine_similarity(ADJ_titleVec,ADJ_contentVec)
    featureVec = np.array([N_featureVec,V_featureVec,ADJ_featureVec])
    return featureVec

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    #X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    #X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    #X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    #X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    #X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    #X = gen_or_load_feats(word2VecFeature, h, b, "features/word2VecFeature."+name+".npy")
    X = gen_or_load_feats(semantic, h, b, "features/semantic."+name+".npy")
  
  
    return X,y

def normalize_word(w):
    return _wnl.lemmatize(w).lower()
def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]




if __name__ == "__main__":
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    stances = hold_out_stances 
    dataset = d 
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    Xs = dict()
    ys = dict()

        # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))

    best_score = 0
    best_fold = None


        # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))
        X_test = Xs[fold]
        y_test = ys[fold]
        X_train = np.nan_to_num(X_train)
        #y_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_train)
        #y_test = np.nan_to_num(X_train)
        X_holdout = np.nan_to_num(X_holdout)
            
        #clf = GradientBoostingClassifier(learning_rate = 0.8,n_estimators=1000, random_state=1418, verbose=True)
        clf = RandomForestClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

        #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    report_score(actual,predicted)