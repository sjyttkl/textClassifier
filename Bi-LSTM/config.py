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
    evaluateEvary = 100
    checkpointEvery = 100
    learningRate = 0.001 #1e-3

class ModelConfig:
    embeddingSize = 200
    hiddenSizes = [256,256]
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config:
    SequenceLength = 200
    batchSize = 138
    dataSource = "../data/preProcess/labeledTrain.csv"
    stopWordSource = "../data/english"
    numClasses = 2
    rate = 0.8 #训练集的比例
    training  = TrainConfig()
    model = ModelConfig()
if __name__ == "__main__":
    # 实例化配置参数对象
    config = Config()


