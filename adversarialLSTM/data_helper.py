# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     data
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/5/25
   Description :  数据预处理的类，生成训练集和测试集
==================================================
"""
__author__ = 'sjyttkl'

import json
import gensim
import pandas as pd
import numpy as np
import config
from  collections import Counter
from tensorflow.contrib import learn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

class DataSet:
    def __init__(self,config):
        self._config = config
        self._dataSource = self._config.dataSource
        self._stopWordSource = self._config.stopWordSource

        self._sequenceLength = self._config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = self._config.model.embeddingSize
        self._batchSize = self._config.batchSize
        self._rate = self._config.rate

        self._stopWordDict = {}
        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}


    def _readData(self,filePath):
        """
        从csv文件中读取数据集
        """
        df = pd.read_csv(filePath,encoding="utf-8")
        labels = df["sentiment"].tolist()
        review = df["review"].tolist() #电影评论的内容
        reviews = [line.strip().split() for line in review] #按照字的粒度进行分割
        return reviews,labels

    def _reviewProcess(selfs,review,sequenceLength,wordToIndex):
        """
        :param review: 传入 一句话，以列表形式分割。
        :param sequenceLength: 固定长度
        :param wordToIndex: 字到 id的映射。字典类型
        :return:返回该句话的。到字典的映射

        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        """
        reviewVec = np.zeros((sequenceLength)) #返回一维 sequenceLength长度的列表
        sequenceLen = sequenceLength
        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)
        for i in range(0,sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["UNK"] #未知单词
        return reviewVec

    def _genTrainEvalData(self,x,y,rate):
        """
        生成训练集和验证集
        :param x: 传入 一个列表集合形式 文本，
        """
        reviews = []
        labels = []
        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(0,len(x)):
            reviewVec = self._reviewProcess(x[i],self._sequenceLength,self._wordToIndex) #这里获得的到 是  每句话中没个词，对应的 id
            reviews.append(reviewVec)
            labels.append([y[i]])

        reviews = np.asarray(reviews,dtype="int64")
        labels = np.asarray(labels,dtype="int32")

        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(x))) #打散
        trainIndex = int(len(x) * rate)

        trainReviews = reviews[shuffle_indices][:trainIndex]
        trainLabels = labels[shuffle_indices][:trainIndex]
        evalReviews = reviews[shuffle_indices][trainIndex:]
        evalLabels = labels[shuffle_indices][trainIndex:]

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self,reivews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """
        allWords = [word for reivew in reivews for word in reivew]
        #去掉停用词
        subWords = [word for word in allWords if word not in self._stopWordDict]

        wordCount = Counter(subWords) #统计词频
        sortWordCount = sorted(wordCount.items(),key=lambda x:x[1],reverse=True)

        #去除低频词
        words = [item[0] for item in sortWordCount if item[1] > 5]

        vocab ,wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding

        #返回word:id
        self._wordToIndex = dict(zip(vocab,list(range(len(vocab)))))
        #返回 id:word
        self._indexToWord = dict(zip(list(range(len(vocab))),vocab))
        # vocab_processor = learn.preprocessing.VocabularyProcessor(self._sequenceLength)
        # vocab_processor.fit(vocab)  # 这里是在制作整个词典，
        # print("返回词典", vocab_processor.vocabulary_._mapping) #word:id
        # print("返回词典",vocab_processor.vocabulary_._reverse_mapping) #id:word
        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据

        # 得到逆词频 ---新添加的
        self._getWordIndexFreq(vocab, reivews)

        with open("../data/wordJson/wordToIndex2.json","w",encoding="utf-8") as f:
            json.dump(self._wordToIndex,f)
        with open("../data/wordJson/indexToWord2.josn","w",encoding="utf-8") as f:
            json.dump(self._indexToWord,f)

    def _getWordEmbedding(self,words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("../word2vec/word2Vec.bin",binary=True)
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("pad") #补齐
        vocab.append("UNK") #未登录词
        wordEmbedding.append(np.zeros(self._embeddingSize)) #初始化 这是为了对 pad,下面的是unk进行的操作
        wordEmbedding.append(np.random.randn(self._embeddingSize))#初始化

        for word in words:
            try:
                vector = wordVec.wv[word] #使用训练好的词向量，进行操作
                vocab.append(word)
                wordEmbedding.append(vector) #把存在的词向量加进去
            except:
                print(word +"不在词向量中")
        return vocab,np.array(wordEmbedding)

    def _getWordIndexFreq(self, vocab, reviews):
        """
        统计词汇空间中各个词出现在多少个文本中
        """
        reviewDicts = [dict(zip(review, range(len(review)))) for review in reviews]#这里是产生一个字典：一个 word: index(这个index是最后一次出现的位置)
        indexFreqs = [0] * len(vocab)  #词表长度
        for word in vocab:
            count = 0
            for review in reviewDicts:
                if word in review:
                    count += 1
            indexFreqs[self._wordToIndex[word]] = count #这里返回的就是  一个字典：id：文本数

        self.indexFreqs = indexFreqs
    def _readStopWord(self,stopWordPath):
        """
        读取停用词
        """
        with open(stopWordPath,"r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快,这里的字典类型是：word:id
            self._stopWordDict = dict(zip(stopWordList,list(range(len(stopWordList)))))

    def dataGen(self):
        """
        初始化训练集和验证集
        """
        #初始化停用词
        self._readStopWord(self._stopWordSource)
        #初始化数据集
        reviews,labels = self._readData(self._dataSource)
        #初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)
        #初始化训练集合测试集
        trainReviews,trainLabels,evalReviews,evalLabels = self._genTrainEvalData(reviews,labels,self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

    # 输出batch数据集
    def nextBatch(self,x,y,batchSize):
        """
        生成batch数据集，用生成器的方式输出
        """
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]
        numBatches = len(x) // batchSize
        for i in range(numBatches):
            start = i * batchSize
            end = start + batchSize
            batchX = np.array(x[start:end],dtype="int64")
            batchY = np.array(y[start:end], dtype="float32")
            yield batchX ,batchY

    # 定义性能指标函数
def mean(item):
    return sum(item) / len(item)

def genMetrics(trueY, predY, binaryPredY):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)

    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)

if __name__ == "__main__":
    data = DataSet(config.Config())
    data.dataGen()

    print("train data shape: {}".format(data.trainReviews.shape))
    print("train label shape: {}".format(data.trainLabels.shape))
    print("eval data shape: {}".format(data.evalReviews.shape))

