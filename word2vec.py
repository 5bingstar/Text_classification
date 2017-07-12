#!/usr/bin/python
#-*-coding:utf-8 -*-

import jieba
import sys
import os
import re
from gensim.models import word2vec
reload(sys)
sys.setdefaultencoding('utf-8')

def loading(file_name):
    stopwords = {}
    fstop = open('stop_word.txt', 'r')
    for eachWord in fstop:
        stopwords[eachWord.strip().decode('utf-8', 'ignore')] = eachWord.strip().decode('utf-8', 'ignore')
    fstop.close()
    words = []
    with open(file_name, 'r') as fin:
        for line in fin:
            line1 = line.strip().decode('utf-8', 'ignore')
            line1 = re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),line1)
            wordList = list(jieba.cut(line1))   
            outStr = [] 
            for word in wordList:
                if word not in stopwords:  
                    outStr.append(word)  
            if len(outStr) > 5:
                words.append(outStr)
    return words
            
def saving(name, words):
    with open(name, 'wb') as fout:
        for i in xrange(len(words)):
            fout.write(' '.join(words[i]).encode('utf-8'))
            fout.write("\n")

def generate_model(save_model):
    corpus = 'segment.txt'
    data1 = 'database/wiki_chs_1'
    data2 = 'database/wiki_chs_2'
    if not os.path.exists(corpus):
        print "process in segment"
        print "loading data1"
        words = loading(data1)
        print "loading data2"
        words += loading(data2)
        saving(corpus, words)
    sentences = word2vec.Text8Corpus(corpus)
    #save_model_bin = 'corpus.model.bin'
    if not os.path.exists(save_model):
        print "process in model"
        model = word2vec.Word2Vec(sentences, size=200)
        print "finished model"
        model.save(save_model)
        print "saced model"
        #model.wv.save_word2vec_format(save_model_bin, binary=True) 
    
    ''' show example of word vector
    model = word2vec.Word2Vec.load(save_model)
    t = u"目标"
    xx = model.most_similar(t)
    print t.encode('utf-8'),
    print "  的词向量为："
    print model[t]
    print u"与其比较相似的词有".encode('utf-8')
    for x in xx:
        print "%s: %f"  %(x[0].encode('utf-8'), x[1])
    #model_vec = word2vec.KeyedVectors.load_word2vec_format(save_model_bin, binary=True)
'''
generate_model('corpus.model')
