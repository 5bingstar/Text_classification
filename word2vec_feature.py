#!/usr/bin/python
#-*-coding:utf-8 -*-

import numpy as np
from word2vec import *
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC, LinearSVC


def word2vec_extraction():
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
    save_model = 'word2vec_model/corpus.model'
    if not os.path.exists(save_model):
        generate_model(save_model)
    model = word2vec.Word2Vec.load(save_model)
    marks = {}
    count = {}
    m = 0
    vecs = []              #data features
    labels = []            #data labels
    for text in words:
        cl = text[0]
        if not marks.has_key(cl):
            tag = True
            for s in marks:
                if cl.startswith(s):
                    tag = False
            if tag == True:
                m += 1
            marks[cl] = m
        sums = 0
        vec = np.zeros(200)
        for x in text[1:-1]:
            w = x.decode('utf-8')
            if w in model:
                vec += model[w]
                sums += 1
        vec /= sums
        vecs.append(vec)
        labels.append(marks[cl])
    return vecs, labels
