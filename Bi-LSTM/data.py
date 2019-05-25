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

import pandas as pd
import numpy as np
class DataSet:
    def __init__(self):
        pass

    def _readData(self,filePath):
        """
        从csv文件中读取数据集
        """
        df = pd.read_csv(filePath,delimiter="\t",encoding="utf-8")
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

    def _genTrainEvalData(self,x_input,y_output,rate):
        """
                生成训练集和验证集
        """