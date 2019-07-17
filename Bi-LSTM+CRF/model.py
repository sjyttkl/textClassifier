# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     data
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/7/17
   Description :  Bi-Lstm 序列标注
==================================================
"""
__author__ = 'sjyttkl'

import tensorflow as tf
import numpy as np


class BiLstm:
    """
    Bi-lstm 情感分析（文本分类）
    """
    def __init__(self,config,wordEmbedding):
        self.input_x = tf.placeholder(tf.int32,[None,config.sequenceLength],name="input_x")
        self.input_y = tf.placeholder(tf.float32,[None,1],name="input_y")
        self.dropoutKeepProb =tf.placeholder(tf.float32,name="dropoutKeepProb")
        self.config = config
        # 定义l2损失
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.cast(wordEmbedding,dtype=tf.float32,name="word2vec"),name="W")#这里的wordEmbedding是已经训练过的。这里就不需要训练
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W,self.input_x)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx,hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize,state_is_tuple=True),output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize,state_is_tuple=True),output_keep_prob=self.dropoutKeepProb)
                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    #embeddedWords为输入的tensor，[batch_szie, max_time,depth]。batch_size为模型当中batch的大小，应用在文本中时，max_time可以为句子的长度（一般以最长的句子为准，短句需要做padding），depth为输入句子词向量的维度。
                    # time_major 决定了输入输出tensor的格式：如果为true, 向量的形状必须为 `[max_time, batch_size, depth]`.如果为false, tensor的形状必须为`[batch_size, max_time, depth]`.
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h(hidden state), c(memory cell))
                    outputs,self.current_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstmFwCell,cell_bw=lstmBwCell,inputs=self.embeddedWords,dtype=tf.float32,
                                                                                 scope="bi-lstm_" + str(idx),time_major=False)#(?,200,256)
                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]  其实，中间的time-step 可以当做为 embeddingSize
                    self.embeddedWords = tf.concat(outputs, 2)  #因为是情感分类，所以需要对输出的结果进行拼接。 #(?,200,512)
                # 取出最后时间步的输出作为全连接的输入。对于情感类的分类问题，需要一句话全局特征，所以只需要最后一步的效果即可。
        finalOutput = self.embeddedWords[:, -1, :]  # (?,512)
        outputSize = config.model.hiddenSizes[-1] * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2   [256*256][-1] *2 = 512
        output = tf.reshape(finalOutput, [-1, outputSize])  # reshape成全连接层的输入维度#(?,512)

        #全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable("outputW",shape=[self.config.batchSize,outputSize,self.config.tag_size],initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.Variable(tf.constant(0.1,shape=[self.config.batchSize,self.config.sequenceLength,self.config.tag_size],name="bais"))
            l2_loss += tf.nn.l2_loss(outputW)
            l2_loss += tf.nn.l2_loss(bias)
            bilstm_out = tf.tanh(tf.matmul(self.embeddedWords, outputW) + bias)
            # self.predictions = tf.nn.xw_plus_b(output,outputW,bias,name="predictions")#(?,1)
            # self.binaryPreds = tf.cast(tf.greater_equal(self.predictions,0.5),tf.float32,name="binaryPreds")#(?,1) #返回 bool类型并转换成 数值类型.False:0,True:1
        # with tf.name_scope("CRF"):
        #     pass
        with tf.name_scope("loss"):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(bilstm_out, self.input_y,
                                                                                       tf.tile(np.array([self.config.sequenceLength]),
                                                                                               np.array([self.config.batchSize])))
            self.loss = tf.reduce_mean(-log_likelihood)
        with tf.name_scope("accuracy"):

            # Compute the viterbi sequence and score (used for prediction and test time).
            self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(bilstm_out, self.transition_params,
                                                                             tf.tile(np.array([self.config.sequenceLength]),
                                                                                     np.array([self.config.batch_size])))

        #     # Training ops.
        #     # optimizer = tf.train.AdamOptimizer(self.lr)
        #     # self.train_op = optimizer.minimize(loss)
        # # 计算二元交叉熵损失
        # with tf.name_scope("loss"):
        #     losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions,labels=self.input_y)
        #     self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2_loss
        #
        # # Accuracy
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.binaryPreds, self.input_y) #(?,1)  返回是bool类型
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



