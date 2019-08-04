# -*- coding: utf-8 -*-

"""
==================================================
    File Name：     data
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/5/26
   Description :  rcnn train.py 训练数据
==================================================
"""
__author__ = 'sjyttkl'

import tensorflow as tf
import data_helper
import config
import model
import os
import time
import datetime
class Train:
    def __init__(self):
        self._config = config.Config()
        self._data = data_helper.DataSet(self._config)
        self._data.dataGen()
        # 生成训练集和验证集
        self.trainReviews = self._data.trainReviews
        self.trainLabels = self._data.trainLabels
        self.evalReviews = self._data.evalReviews
        self.evalLabels = self._data.evalLabels

        self.wordEmbedding = self._data.wordEmbedding
    # 定义计算图
    def train(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,#自动选择运行设备
                log_device_placement=False)#记录设备指派情况，这个session配置，按照前面的gpu，cpu自动选择
            session_conf.gpu_options.allow_growth = True
            session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
            sess = tf.Session(config=session_conf)  # 建立一个配置如上的会话

            #定义会话
            with sess.as_default():
                lstm = model.RCNN(config = self._config,wordEmbedding=self.wordEmbedding)
                global_step = tf.Variable(0,name="gloal_step",trainable=False)
                # 定义优化函数，传入学习速率参数
                optimizer = tf.train.AdamOptimizer(self._config.training.learningRate)
                #计算梯度，得到梯度和变量
                grads_vars = optimizer.compute_gradients(lstm.loss)
                # 将梯度应用到变量下，生成训练器
                train_op = optimizer.apply_gradients(grads_and_vars=grads_vars,global_step=global_step)
                # 用summary绘制tensorBoard
                grad_summaries = []
                for g,v in grads_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name),g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))  # 绝对路径制作
                print("Writing to {}\n".format(out_dir))

                loss_summary = tf.summary.scalar("loss",lstm.loss)
                acc_summary = tf.summary.scalar("accuracy", lstm.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged]) #summaryOp = tf.summary.merge_all()
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it

                # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                # if not os.path.exists(checkpoint_dir):
                #     os.makedirs(checkpoint_dir)
                # 初始化所有变量
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                # saver = tf.train.Saver(tf.global_variables(), max_to_keep=100, pad_step_number=True)
                # 保存模型的一种方式，保存为pb文件
                saver_dir = os.path.join(out_dir, "checkpoints")
                builder = tf.saved_model.builder.SavedModelBuilder(saver_dir)
                sess.run(tf.global_variables_initializer())
                def train_step(batchX,batchY):
                    feed_dict={
                        lstm.input_x:batchX,
                        lstm.input_y:batchY,
                        lstm.dropoutKeepProb : self._config.model.dropoutKeepProb
                    }
                    _, summary, step, loss,accuracy, predictions, binaryPreds = sess.run(
                        [train_op, train_summary_op, global_step, lstm.loss,lstm.accuracy, lstm.predictions, lstm.binaryPreds],
                        feed_dict)
                    timeStr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())#datetime.datetime.now().isoformat()
                    acc, auc, precision, recall = data_helper.genMetrics(batchY, predictions, binaryPreds)
                    print("train ..{}, step: {}, loss: {},accuracy: {},  acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step,
                                                                                                       loss,accuracy, acc, auc,
                                                                                                       precision,
                                                                                                       recall))
                    train_summary_writer.add_summary(summary, step)
                def devStep(batchX, batchY):
                    """
                    验证函数
                    """
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropoutKeepProb: 1.0
                    }
                    summary, step, loss, accuracy, predictions, binaryPreds = sess.run(
                        [ dev_summary_op, global_step, lstm.loss, lstm.accuracy, lstm.predictions,
                         lstm.binaryPreds],
                        feed_dict)
                    timeStr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())#datetime.datetime.now().isoformat()
                    acc, auc, precision, recall = data_helper.genMetrics(batchY, predictions, binaryPreds)
                    print("dev ..{}, step: {}, loss: {},accuracy: {},  acc: {}, auc: {}, precision: {}, recall: {}".format(
                        timeStr, step,
                        loss, accuracy, acc, auc,
                        precision,
                        recall))
                    dev_summary_writer.add_summary(summary, step)
                    return loss, acc, auc, precision, recall
                for i in range(self._config.training.epoches):
                    # 训练模型
                    print("start training model.......")
                    for batchTrain in self._data.nextBatch(self._data.trainReviews, self._data.trainLabels, self._config.batchSize):
                        train_step(batchTrain[0], batchTrain[1])
                        currentStep = tf.train.global_step(sess, global_step)
                        if currentStep % self._config.training.evaluateEvery == 0:
                            print("\nEvaluation:")
                            losses = []
                            accs = []
                            aucs = []
                            precisions = []
                            recalls = []
                            for batchEval in self._data.nextBatch(self._data.evalReviews, self._data.evalLabels, self._config.batchSize):
                                loss, acc, auc, precision, recall = devStep(batchEval[0], batchEval[1])
                                losses.append(loss)
                                accs.append(acc)
                                aucs.append(auc)
                                precisions.append(precision)
                                recalls.append(recall)
                            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())#datetime.datetime.now().isoformat()
                            print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(time_str,
                                                                                                               currentStep,
                                                                                                               data_helper.mean(
                                                                                                                   losses),
                                                                                                               data_helper.mean(
                                                                                                                   accs),
                                                                                                               data_helper.mean(
                                                                                                                   aucs),
                                                                                                               data_helper.mean(
                                                                                                                   precisions),
                                                                                                               data_helper.mean(
                                                                                                                   recalls)))
                            if currentStep % self._config.training.checkpointEvery == 0:
                                # 保存模型的另一种方法，保存checkpoint文件
                                checkpoint_prefix = os.path.join(saver_dir, "model")
                                path = saver.save(sess, checkpoint_prefix, global_step=currentStep)
                                print("Saved model checkpoint to {}\n".format(path))

                            #目前不需要这种保存形式，pb
                            # inputs = {"inputX": tf.saved_model.utils.build_tensor_info(lstm.input_x),
                            #           "keepProb": tf.saved_model.utils.build_tensor_info(lstm.dropoutKeepProb)}
                            #
                            # outputs = {"binaryPreds": tf.saved_model.utils.build_tensor_info(lstm.binaryPreds)}
                            #
                            # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
                            #                                                                               outputs=outputs,
                            #                                                                               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                            # legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
                            # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                            #                                      signature_def_map={"predict": prediction_signature},
                            #                                      legacy_init_op=legacy_init_op)
                            #
                            # builder.save()



#
if __name__== "__main__":
    train = Train()
    train.train()
    #tf.app.run()