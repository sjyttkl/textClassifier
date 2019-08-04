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
class   RCNN:
    """
        RCNN 用于文本分类
    """
    def __int__(self,config,wordEmbedding):
        self.input_x = tf.placeholder(tf.int32,[None,config.sequenceLength],name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.config = config
        # 定义l2损失
        l2_loss = tf.constant(0.0)
        #词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
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
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize,state_is_tuple=True),
                                                  output_keep_prob=self.dropoutKeepProb)
                    #定义后向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize,state_is_tuple=True),
                                                               output_keep_prob=self.dropoutKeepProb)
                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # embeddedWords为输入的tensor，[batch_szie, max_time,depth]。batch_size为模型当中batch的大小，应用在文本中时，max_time可以为句子的长度（一般以最长的句子为准，短句需要做padding），depth为输入句子词向量的维度。
                    # time_major 决定了输入输出tensor的格式：如果为true, 向量的形状必须为 `[max_time, batch_size, depth]`.如果为false, tensor的形状必须为`[batch_size, max_time, depth]`.
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h(hidden state), c(memory cell))
                    outputs_,self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell,lstmBwCell,self.embeddedWords_,dtype=tf.float32,
                                                                                  scope="bi-lstm"+str(idx))
                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]  其实，中间的time-step 可以当做为 embeddingSize
                    self.embeddedWords_ = tf.concat(outputs_,2) #因为是情感分类，所以需要对输出的结果进行拼接。 #(?,200,512)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        fwOutput, bwOutput = tf.split(self.embeddedWords_, 2, -1)

        with tf.name_scope("context"):
            shape= [tf.shape(fwOutput)[0],1,tf.shape(fwOutput)[2]]
            self.contextLeft = tf.concat([tf.zeros(shape),fwOutput[:,:-1]],axis=1,name="contextLeft")
            self.contextRight = tf.concat([bwOutput[:,1:],tf.zeros(shape)],axis=1,name="contextRight")\

        # 将前向，后向的输出和最早的词向量拼接在一起得到最终的词表征
        with tf.name_scope("wordRepresentation"):
            self.wordRepre = tf.concat([self.contextLeft,self.embeddedWords,self.contextRight],axis=2)
            wordSize = config.model.hiddenSizes[-1] *2 + self.config.model.embeddingSize

        with tf.name_scope("textRepresentation"):
            outputSize = self.config.model.outputSize
            textW = tf.Variable(tf.random_uniform([wordSize,outputSize],-1.0,1.0),name='W2')
            textB = tf.Variable(tf.constant(0.1,shape=[outputSize],name='b2'))

            # tf.einsum可以指定维度的消除运算
            self.textRepre = tf.tanh(tf.einsum('aij,jk->aik', self.wordRepre, textW) + textB)

        # 做max-pool的操作，将时间步的维度消失
        output = tf.reduce_max(self.textRepre,axis=1)

        #全连接输出
        with tf.name_scope("output"):
            outputW = tf.get_variable("outputW",shape=[outputSize,1],initializer=tf.contrib.layers.xavier_initializer())
            outputB = tf.Variable(tf.constant(0.1,shape=[1]),name="outputB")

            l2_loss += tf.nn.l2_loss(outputW)
            l2_loss += tf.nn.l2_loss(outputB)
            self.predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")  # (?,1)
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32,name="binaryPreds")  # (?,1) #返回 bool类型并转换成 数值类型.False:0,True:1

            # 计算二元交叉熵损失
            with tf.name_scope("loss"):
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2_loss