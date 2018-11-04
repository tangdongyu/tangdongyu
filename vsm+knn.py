import os
import nltk
import math
import pandas as pd
import numpy as np
import string
import re
import shutil
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

data = 'E:\data\20news-18828'
data_train = 'E:\data\20news-18828\20news-18828-train'
data_test = 'E:\data\20news-18828\20news-18828-test'
_dictionary = 'E:\data/dictionary.csv'

def vsm(input):
    # 读取数据
    raw_data = []
    sort = []
    num = 0
    for file1 in os.listdir(input):
        path1 = os.path.join(input, file1)
        num += 1
        for file2 in os.listdir(path1):
            path2 = os.path.join(path1, file2)
            sort.append(num)
            with open(path2, encoding='latin-1') as file:
                document = file.read()
                raw_data.append(document)
    # 处理数据
    new_data = []
    for doc in raw_data:
        delpunctuation = re.compile('[%s]' % re.escape(string.punctuation))
        doc = delpunctuation.sub("", doc)
        lowers = str(doc).lower()
        tokens = nltk.word_tokenize(lowers)
        stemmer = PorterStemmer()
        stoplist = stopwords.words('english')
        words = []
        for word in tokens:
            if word not in stoplist:
                words.append(stemmer.stem(word))
        new_data.append(words)
    # 创建词典
    dictionary = []
    if not os.path.exists(_dictionary):
        count = []
        for doc in new_data:
            count += doc
        count = Counter(count)
        for word in count:
            if count[word] >= 9 and count[word] <= 10000:
                if word not in dictionary:
                    dictionary.append(str(word))
        pd.DataFrame(dictionary).to_csv(_dictionary, sep=",", header=None, index=None)
    else:
        dictionary = np.array(pd.read_csv(_dictionary, sep=" ", header=None)).reshape(1, -1)[0]
    # 生成VSM
    TF_IDF_vectors = []
    vsm_vectors = []
    for doc in new_data:
        TF_IDF_vector = []
        vsm_vector = []
        for word in dictionary:
            tf = str(doc).count(str(word))
            if tf > 0:
                tf = 1 + math.log(tf)
            else:
                tf = 0
            df = len(list(filter(lambda p: word in p, new_data)))
            idf = math.log(len(new_data) / df)
            tf_idf = tf * idf
            TF_IDF_vector.append(tf_idf)
            if word in doc:
                vsm_vector.append('1')
            else:
                vsm_vector.append('0')
        TF_IDF_vectors.append(TF_IDF_vector)
        vsm_vectors.append(vsm_vector)
    return vsm_vectors, sort

def knn(train_X, train_Y, test_X, test_Y):
    similarity = cosine_similarity(test_X, train_X)
    K = 30
    prediction = []
    for item in similarity:
        dic = dict(zip(item, train_Y))
        dic = sorted(dic.items(), key=lambda v: v[0], reverse=True)
        classes = np.zeros((1, len(train_Y)))[0]
        for i in dic[:K]:
            classes[i[1]] += (1 / (1 - i[0]) ** 2)
        dic = dict(zip(classes, range(len(train_Y))))
        dic = sorted(dic.items(), key=lambda v: v[0], reverse=True)
        prediction.append(dic[0][1])
    print("K=", K, "\t Accuracy:\t", accuracy_score(test_Y, prediction))

if __name__ == '__main__':
#     for folder in os.listdir(data):
#         path = os.path.join(data, folder)
#         os.makedirs(data_train + '/' + folder)
#         os.makedirs(data_test + '/' + folder)
#         i = 0
#         for filename in os.listdir(path):
#             if i < len(os.listdir(path)) * 0.8:
#                 shutil.copyfile(os.path.join(path, filename), os.path.join(data_train + '/' + folder, filename))
#             else:
#                 shutil.copyfile(os.path.join(path, filename), os.path.join(data_test + '/' + folder, filename))
#             i += 1
#     print('ready')
    train_data, train_label = vsm(data_train)
    print("train_set_end")
    test_data, test_label = vsm(data_test)
    print("test_set_end")
    knn(train_data, train_label, test_data, test_label)