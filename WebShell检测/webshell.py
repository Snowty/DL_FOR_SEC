#!/usr/bin/env python
# -*- coding:utf-8 -*- 
import codecs
import os
import pickle
import re
import subprocess
import sys

import numpy as np
import tensorflow as tf
import tflearn
from sklearn import cross_validation, metrics, preprocessing, svm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from tensorflow.contrib import learn
from tflearn.data_utils import pad_sequences, to_categorical
from tflearn.layers.conv import conv_1d, conv_2d, global_max_pool, max_pool_2d
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization

import xgboost as xgb

max_features=10000
max_document_length=100
min_opcode_count=2
white_count=0
black_count=0

webshell_dir="./webshell/webshell/PHP/"
whitefile_dir="./webshell/normal/php/"
php_bin="/usr/bin/php"

pkl_file="webshell-opcode-cnn.pkl"


# -------------------------------------数据预处理---------------------------------
# 加载文件到一个字符串变量中，过滤回车换行
def load_file(file_path):
    t=""
    with codecs.open(file_path,encoding = "ISO-8859-1") as f:
        for line in f:
            line=line.strip('\n')
            line = line.strip('\r')
            t+=line
    return t

# 排除非php文件，如图片、js等。
def load_files_re(dir):
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        #print d;
        for filename in filelist:
            #print os.path.join(path, filename)
            if filename.endswith('.php') or filename.endswith('.txt'):
                fulepath = os.path.join(path, filename)
                # print("Load %s" % fulepath)
                t = load_file(fulepath)
                # 再将文件存放到列表中
                files_list.append(t)
    return files_list

def get_feature_by_bag_tfidf():
    global white_count
    global black_count
    global max_features
    #print("max_features=%d" % max_features)
    x=[]
    y=[]
    
    # 加载搜集到的webshell样本，并统计样本个数，将webshell样本标记为1
    webshell_files_list = load_files_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    # 加载搜集到的开源软件样本，并统计样本个数，将开源软件样本标记为0
    wp_files_list =load_files_re(whitefile_dir)
    y2=[0]*len(wp_files_list)
    white_count=len(wp_files_list)

    # 将webshell样本与开源软件样本进行合并
    x=webshell_files_list+wp_files_list
    y=y1+y2

    # 2-Gram提取词袋模型
    # ngram_range = (2,2)即为2-Gram；token_pattern代表分词的规则，此处代表按单词切割
    CV = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)
    x=CV.fit_transform(x).toarray()
    # 使用TF-IDF进行处理
    transformer = TfidfTransformer(smooth_idf=False)
    x_tfidf = transformer.fit_transform(x)
    x = x_tfidf.toarray()

    return x,y

# 将php文件转换为opcode，需要先下载安装php的VLD扩展，参考https://www.cnblogs.com/webph/p/6979840.html
def load_file_opcode(file_path):
    global php_bin
    t=""
    cmd=php_bin+" -dvld.active=1 -dvld.execute=0 "+file_path
    #print "exec "+cmd
    status,output=subprocess.getstatusoutput(cmd)

    # 对output进行处理，opcode均是大写字母和下划线组成的单词
    t=output
    tokens=re.findall(r'\s(\b[A-Z_]+\b)\s',output)
    t=" ".join(tokens)
    # print("opcode count %d" % len(t))
    return t

# 遍历读取指定目录下全部php文件，保存其对应的opcode字符串
def load_files_opcode_re(dir):
    global min_opcode_count
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        #print d;
        for filename in filelist:
            #print os.path.join(path, filename)
            if filename.endswith('.php') :
                fulepath = os.path.join(path, filename)
                # print("Load %s opcode" % fulepath)
                try:
                    t = load_file_opcode(fulepath)
                    if len(t) > min_opcode_count:
                        files_list.append(t)
                except Exception:
                    # print("Load %s opcode failed" % fulepath)
                    pass
                #print "Add opcode %s" % t
    return files_list

# 读取保存Webshell样本以及正常php文件的目录，加载对应的opcode字符串，标记webshell为1，正常文件为0
def get_feature_by_opcode():
    global white_count
    global black_count
    global max_features
    global webshell_dir
    global whitefile_dir
    print("max_features=%d webshell_dir=%s whitefile_dir=%s" % (max_features,webshell_dir,whitefile_dir))
    x=[]
    y=[]

    webshell_files_list = load_files_opcode_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_opcode_re(whitefile_dir)
    y2=[0]*len(wp_files_list)
    white_count=len(wp_files_list)

    x=webshell_files_list+wp_files_list
    #print x
    y=y1+y2

    # 2-Gram
    CV = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)
    x=CV.fit_transform(x).toarray()

    #TF-IDF
    transformer = TfidfTransformer(smooth_idf=False)
    x_tfidf = transformer.fit_transform(x)
    x = x_tfidf.toarray()

    return x,y

# -------------------------------------数据预处理结束---------------------------------

# 模型评估
def do_metrics(y_test,y_pred):
    print((classification_report(y_test, y_pred)))
    print("confusion_matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

# xgboost
def do_xgboost(x,y):
    print("start xgboost......")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print((classification_report(y_test, y_pred)))
    print(metrics.confusion_matrix(y_test, y_pred))

# 朴素贝叶斯
def do_nb(x,y):
    print("start Naive Bayes......")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    do_metrics(y_test,y_pred)

# 随机森林
def do_rf(x,y):
    print("start RandomForest......")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    do_metrics(y_test,y_pred)

# SVM
def do_svm(x,y):
    print("start SVM......")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    do_metrics(y_test,y_pred)

# MLP
def do_mlp(x,y):
    print("start MLP......")
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)

    #print clf
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(y_train)
    print(y_pred)
    print(y_test)
    do_metrics(y_test,y_pred)

# CNN
def do_cnn(x,y):
    print("start CNN......")
    global max_document_length
    print("CNN and tf")
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.4, random_state=0)
    y_test=testY
    # 训练、测试数据进行填充和转换，不到最大长度的数据填充0
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    # 二分类问题，把标记数据二值化
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_document_length], name='input')
    # 三个数量为128，长度分别为3，4，5的一维卷积函数处理数据
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # 实例化CNN对象并进行训练数据，训练5轮
    model = tflearn.DNN(network, tensorboard_verbose=0)
    if not os.path.exists(pkl_file):
        model.fit(trainX, trainY,
                  n_epoch=5, shuffle=True, validation_set=0.1,
                  show_metric=True, batch_size=100,run_id="webshell")
        model.save(pkl_file)
    else:
        model.load(pkl_file)

    y_predict_list=model.predict(testX)

    y_predict=[]
    for i in y_predict_list:
        print(i[0])
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)
    print('y_predict_list:')
    print(y_predict_list)
    print('y_predict:')
    print(y_predict)
    #print  y_test

    do_metrics(y_test, y_predict)

# RNN
def do_rnn(x,y):
    print("start RNN......")
    global max_document_length
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.4, random_state=0)
    y_test=testY

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_document_length])
    net = tflearn.embedding(net, input_dim=10240000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=0.1, show_metric=True,
              batch_size=10,run_id="webshell",n_epoch=5)

    y_predict_list=model.predict(testX)
    y_predict=[]
    for i in y_predict_list:
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    do_metrics(y_test, y_predict)


if __name__ == '__main__':
    # -------------------------------------xgboost--------------------------------------
    # print("xgboost and bag and 2-gram")
    # max_features=5000
    # print("max_features=%d" % max_features)
    # x, y = get_feature_by_bag_tfidf()
    # print("load %d white %d black" % (white_count, black_count))
    # do_xgboost(x, y)
    # print("----------------------------------------")

    # print("xgboost and opcode and 4-gram")
    # max_features=10000
    # max_document_length=4000
    # print("max_features=%d max_document_length=%d" % (max_features,max_document_length))
    # x, y = get_feature_by_opcode()
    # print("load %d white %d black" % (white_count, black_count))
    # do_xgboost(x, y)
    # print("----------------------------------------")

    # print("xgboost and wordbag and 2-gram")
    # max_features=10000
    # max_document_length=4000
    # print("max_features=%d max_document_length=%d" % (max_features,max_document_length))
    # x, y = get_feature_by_bag_tfidf()
    # print("load %d white %d black" % (white_count, black_count))
    # do_xgboost(x, y)
    # print("----------------------------------------")
    # -------------------------------------xgboost--------------------------------------


    x, y=get_feature_by_bag_tfidf()
    print("load %d white %d black" % ( white_count,black_count ))

    # 朴素贝叶斯
    do_nb(x,y)

    # # 随机森林
    # do_rf(x,y)

    # # SVM
    # do_svm(x,y)

    # # mlp
    # do_mlp(x,y)

    # # CNN
    # do_cnn(x,y)

    # # RNN
    # do_rnn(x,y)