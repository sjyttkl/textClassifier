# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     data
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/5/26
   Description :  adversarialLstm  情感分析 https://www.cnblogs.com/jiangxinyang/p/10208363.html   论文：https://arxiv.org/pdf/1605.07725.pdf
==================================================
"""
__author__ = 'sjyttkl'

import tensorflow as tf


# 构建模型
class AdversarialLSTM(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding, indexFreqs):
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.config = config

        # 根据词的频率计算权重
        indexFreqs[0], indexFreqs[1] = 20000, 10000
        weights = tf.cast(tf.reshape(indexFreqs / tf.reduce_sum(indexFreqs), [1, len(indexFreqs)]), dtype=tf.float32)#这里是把 对indexFreqs进行了归一化，并且转成（1，28604）

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用词频计算新的词嵌入矩阵
            normWordEmbedding = self._normalize(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), weights)

            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(normWordEmbedding, self.inputX)

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            with tf.variable_scope("Bi-LSTM", reuse=None):
                self.predictions = self._Bi_LSTMAttention(self.embeddedWords)
                self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32, name="binaryPreds")#直接输出二分类的结果了
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY) #交叉熵
                loss = tf.reduce_mean(losses) #进行平均交叉熵

        with tf.name_scope("perturLoss"):
            with tf.variable_scope("Bi-LSTM", reuse=True):
                perturWordEmbedding = self._addPerturbation(self.embeddedWords, loss)
                perturPredictions = self._Bi_LSTMAttention(perturWordEmbedding)
                perturLosses = tf.nn.sigmoid_cross_entropy_with_logits(logits=perturPredictions, labels=self.inputY)
                perturLoss = tf.reduce_mean(perturLosses)

        self.loss = loss + perturLoss

    def _Bi_LSTMAttention(self,embeddedWords):
        """
        Bi-LSTM + Attention 的模型结构
        """

        config = self.config

        # 定义双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            # 定义前向LSTM结构
            lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                output_keep_prob=self.dropoutKeepProb)
            # 定义反向LSTM结构
            lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=config.model.hiddenSizes, state_is_tuple=True),
                output_keep_prob=self.dropoutKeepProb)

            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # embeddedWords为输入的tensor，[batch_szie, max_time,depth]。batch_size为模型当中batch的大小，应用在文本中时，max_time可以为句子的长度（一般以最长的句子为准，短句需要做padding），depth为输入句子词向量的维度。
            # time_major 决定了输入输出tensor的格式：如果为true, 向量的形状必须为 `[max_time, batch_size, depth]`.如果为false, tensor的形状必须为`[batch_size, max_time, depth]`.
            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
            # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h(hidden state), c(memory cell))
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstmFwCell, cell_bw=lstmBwCell,
                                                                          inputs=embeddedWords, dtype=tf.float32,
                                                                          scope="bi-lstm",
                                                                          time_major=False)  # (?,200,256)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self._attention(H)
            outputSize = config.model.hiddenSizes

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, 1],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
            predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")

        return predictions

    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.config.model.hiddenSizes

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.config.sequenceLength])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.sequenceLength, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output

    def _normalize(self, wordEmbedding, weights):
        """
        对word embedding 结合权重做标准化处理
        """

        mean = tf.matmul(weights, wordEmbedding)  # 结果维度：（1,28604）*（28604,200） = （1，200）#均值
        print(mean)
        powWordEmbedding = tf.pow(wordEmbedding - mean, 2.) #（28604,200）

        var = tf.matmul(weights, powWordEmbedding)  #（1,28604）*（28604,200）= （1，200)  #方差
        print(var)
        stddev = tf.sqrt(1e-6 + var)#标准差,这里的1e-6 是为了防止为零

        return (wordEmbedding - mean) / stddev #这里返回的就是 标准化后的 embedding（28604,200）

    #一般对抗损失，（还存在随机对抗损失、虚拟对抗损失)，对抗扰动（Adversarial perturbation）
    def _addPerturbation(self, embedded, loss):
        """
        添加扰动到word embedding
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N) #这里的tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N会大大节省内存
        grad = tf.stop_gradient(grad)#(?,200,200)
        #是一个tensor或tensor的列表，所有关于xs作为常量（constant），这些tensor不会被反向传播，仿佛它们已经被使用stop_gradients 显式地断开。除此之外，这允许计算偏导数，而不是全导数。
        perturb = self._scaleL2(grad, self.config.model.epsilon)
        return embedded + perturb

    def _scaleL2(self, x, norm_length):
        # shape(x) = (batch, num_timesteps, d)
        # Divide x by max(abs(x)) for a numerically stable L2 norm.
        # 2norm(x) = a * 2norm(x/a)
        # Scale over the full sequence, dims (1, 2)
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12# x:(?,200,200),  结果为：(?,1,1)
        l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6) #sqrt 平方根，这里收到的 l2_正则
        x_unit = x / l2_norm
        return norm_length * x_unit