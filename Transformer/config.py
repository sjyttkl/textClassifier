# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     config
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/5/25
   Description :  
==================================================
"""
__author__ = 'sjyttkl'

import pandas as pd
import tensorflow as tf

class TrainConfig:
    epoches =10
    evaluateEvery = 10  #100
    checkpointEvery = 10 #100

class ModelConfig(object):
    embeddingSize = 200

    filters = 128  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
    numHeads = 8  # Attention 的头数
    numBlocks = 1  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.9  # multi head attention 中的dropout

    dropoutKeepProb = 0.5  # 全连接层的dropout
    l2RegLambda = 0.0

class Config:
    sequenceLength = 200
    batchSize = 128
    dataSource = "../data/preProcess/labeledTrain.csv"
    stopWordSource = "../data/english"
    numClasses = 2
    rate = 0.8 #训练集的比例
    training  = TrainConfig()
    model = ModelConfig()
if __name__ == "__main__":
    # 实例化配置参数对象
    config = Config()


