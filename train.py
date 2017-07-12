#!/usr/bin/python
#-*-coding:utf-8 -*-

import numpy as np
from word2vec_feature import *
from bag_of_word import *
from sklearn.model_selection import train_test_split 

  
def svm_classify(x, y, x_test, y_test):    
    from sklearn.svm import SVC, LinearSVC
    clf = LinearSVC()  #decision_function_shape='ovr') 
    clf.fit(x, y)
    l_test = clf.predict(x_test)
    count = 0
    content = []
    for i in xrange(len(l_test)):
        content.append(str(y_test[i]) + "   " + str(l_test[i]))
        if y_test[i] == l_test[i]:
            count += 1
   # with open("svm_result", "w") as fout:
    #    fout.write("\n".join(content))
    #print "Precision = ",
    return 1.0 * count / len(y_test)


def LR_classify(x, y, x_test, y_test):
    from sklearn.linear_model import LogisticRegression
    logclf = LogisticRegression()
    logclf.fit(x, y)
    l_test = logclf.predict(x_test)
    count = 0
    content = []
    for i in xrange(len(l_test)):
        content.append(str(y_test[i]) + "   " + str(l_test[i]))
        if y_test[i] == l_test[i]:
            count += 1
    #with open("lr_result", "w") as fout:
     #   fout.write("\n".join(content))
    return 1.0 * count / len(y_test)



if __name__ == "__main__":
    x_w2v, y_w2v = word2vec_extraction()
    x_tfidf, y_tfidf = tfidf_extraction()
    x, x_test, y, y_test = train_test_split(x_w2v, y_w2v, test_size=0.1)
    p1_w2v = svm_classify(x, y, x_test, y_test)
    p2_w2v = LR_classify(x, y, x_test, y_test)
    x, x_test, y, y_test = train_test_split(x_tfidf, y_tfidf, test_size=0.1)
    p1_tfidf = svm_classify(x, y, x_test, y_test)
    p2_tfidf = LR_classify(x, y, x_test, y_test)

    print "Using tfidf vector from VSM: "
    print "1. by SVM classification: precision = ",
    print p1_tfidf
    print "2. by LR classification: precision = ",
    print p2_tfidf

    print "Using the word vector from word2vec throught CNN: "
    print "1. by SVM classification: precision = ",
    print p1_w2v
    print "2. by LR classification: precision = ",
    print p2_w2v
