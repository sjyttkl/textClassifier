# -*- coding: utf-8 -*-

"""

构建模型，模型的架构如下：
1，利用Bi-LSTM获得上下文的信息
2，将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput;wordEmbedding;bwOutput]
3，将2所得的词表示映射到低维
4，hidden_size上每个位置的值都取时间步上最大的值，类似于max-pool
5，softmax分类
==================================================
   File Name：     model
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/7/10
   Description : 来自论文 Recurrent Convolutional Neural Networks for Text Classification
==================================================
"""
__author__ = 'songdongdong'

import tensorflow as tf
class RCNN:
    """
        RCNN 用于文本分类
    """
    def __int__(self,config,wordEmbdding):
        self.input_x = tf.placeholder(tf.int32,[None,config.sequenceLength],name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.config = config
        # 定义l2损失
        l2_loss = tf.constant(0.0)
        #词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbdding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.input_x)
            # 复制一份embedding input
            self.embeddedWords_ = self.embeddedWords

        # 定义两层双向LSTM的模型结构

        #         with tf.name_scope("Bi-LSTM"):
        #             fwHiddenLayers = []
        #             bwHiddenLayers = []
        #             for idx, hiddenSize in enumerate(config.model.hiddenSizes):

        #                 with tf.name_scope("Bi-LSTM-" + str(idx)):
        #                     # 定义前向LSTM结构
        #                     lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
        #                                                                  output_keep_prob=self.dropoutKeepProb)
        #                     # 定义反向LSTM结构
        #                     lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
        #                                                                  output_keep_prob=self.dropoutKeepProb)

        #                 fwHiddenLayers.append(lstmFwCell)
        #                 bwHiddenLayers.append(lstmBwCell)

        #             # 实现多层的LSTM结构， state_is_tuple=True，则状态会以元祖的形式组合(h, c)，否则列向拼接
        #             fwMultiLstm = tf.nn.rnn_cell.MultiRNNCell(cells=fwHiddenLayers, state_is_tuple=True)
        #             bwMultiLstm = tf.nn.rnn_cell.MultiRNNCell(cells=bwHiddenLayers, state_is_tuple=True)

        #             # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
        #             # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
        #             # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
        #             outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(fwMultiLstm, bwMultiLstm, self.embeddedWords, dtype=tf.float32)
        #             fwOutput, bwOutput = outputs

        with tf.name_scope("Bi-LSTM"):
            for idx ,hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
