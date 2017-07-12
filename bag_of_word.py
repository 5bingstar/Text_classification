#!/usr/bin/python
#-*-coding:utf-8 -*-

import numpy as np
from word2vec import *
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC, LinearSVC


def loading_data():
    file_name = 'analysis_data/10train.txt'
    segment = 'analysis_data/segment.txt'
    words = []
    if not os.path.exists(segment):
        words = loading(file_name)
        saving(segment, words)
    else:
        with open(segment, 'r') as fin:
            for line in fin:
                words.append(line.split())
    corpus = []
    labels = []
    marks = {}
    m = 0
    for r in words:
        corpus.append(' '.join(r))
        cl = r[0]
        if not marks.has_key(cl):
            tag = True
            for s in marks:
                if cl.startswith(s):
                    tag = False
            if tag == True:
                m += 1
            marks[cl] = m
        labels.append(marks[cl])

    return corpus, labels


def transform(dataset):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(dataset))
    word=vectorizer.get_feature_names() # feature
    weight=tfidf.toarray() # matrix
    return weight
    #vectorizer = TfidfVectorizer(max_features=n_features,use_idf=True)
    #X = vectorizer.fit_transform(dataset)
    #return X,vectorizer


def tfidf_extraction():
    dataset, y = loading_data()
    X = transform(dataset)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=20, n_iter=7, random_state=42)
    X_transformed = svd.fit_transform(X)
    #return X, y
    return X_transformed, y  # matrix X with SDV decomposition

