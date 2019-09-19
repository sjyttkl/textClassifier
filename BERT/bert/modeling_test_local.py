# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     modeling_test_local
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/9/1
   Description :  推荐解析代码：https://blog.csdn.net/Kaiyuan_sjtu/article/details/90265473 。


在BERT模型构建这一块的主要流程：
对输入序列进行Embedding（三个），接下去就是‘Attention is all you need’的内容了
简单一点就是将embedding输入transformer得到输出结果
详细一点就是embedding --> N *【multi-head attention --> Add(Residual) &Norm--> Feed-Forward --> Add(Residual) &Norm】
哈，是不是很简单~
源码中还有一些其他的辅助函数，不是很难理解，这里就不再啰嗦。
==================================================
"""
__author__ = 'songdongdong'
import modeling
import tensorflow as tf

# 假设输入已经经过分词变成word_ids. shape=[2, 3]
input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
# segment_emebdding. 表示第一个样本前两个词属于句子1，后一个词属于句子2.
# 第二个样本的第一个词属于句子1， 第二个词属于句子2，第三个元素0表示padding
token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

# 创建BertConfig实例
# 创建一个BertConfig，词典大小是32000，Transformer的隐单元个数是512
# 8个Transformer block，每个block有6个Attention Head，全连接层的隐单元是1024
config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

# 创建BertModel实例
model = modeling.BertModel(config=config, is_training=True,
     input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

# label_embeddings用于把512的隐单元变换成logits
label_embeddings = tf.get_variable(...)
#得到最后一层的第一个Token也就是[CLS]向量表示，可以看成是一个句子的embedding
pooled_output = model.get_pooled_output()
# 计算logits
logits = tf.matmul(pooled_output, label_embeddings)