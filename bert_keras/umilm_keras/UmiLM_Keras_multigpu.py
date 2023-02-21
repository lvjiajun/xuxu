# -*- coding: utf-8 -*-
# @Time : 2023/2/11 14:53
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : UmiLM_multigpu.py
# @Project : xuxu
from __future__ import print_function

import os
import sys

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import glob
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
import tensorflow as tf  # 导入tf，备用
import json
import datetime
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# 数据集类型
dirs_name: str = 'pan12-text-alignment-training-corpus-2012-03-16'
type_name: str = '05_translation'
logfile: list = list()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 基本参数
maxlen = 256
batch_size = 64
steps_per_epoch = 1000
epochs = 10000
try:
    config_file = open('confif.json','r',encoding='utf-8')
    config_hp = json.load(config_file)
    config_file.close()
    dirs_name = str(config_hp['dirs_name'])
    type_name = str(config_hp['type_name'])
    maxlen = int(config_hp['maxlen'])
    batch_size = int(config_hp['batch_size'])
    steps_per_epoch = int(config_hp['steps_per_epoch'])
    epochs = int(config_hp['epochs'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_hp['gpu'])
except Exception as e:
    print(f'not find config !\n{e.args}\n')
json_name: str = f'../../Generate_data/data/{dirs_name}_{type_name}_match_'
project_name = f'project_{dirs_name}_{type_name}'
# bert配置
config_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/vocab.txt'

# 用于将控制台所有输出保存至文件，需放在代码最上面
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', 'utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(stream=sys.stdout)

def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    return json.load(open(filename, 'r', encoding='utf-8'))


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    （每次只需要返回一条样本）
    """

    def __iter__(self, random=False):
        for is_end, pair in self.sample(random):
            src_str, tgt_str = pair['src'], pair['tgt']
            token_ids, segment_ids = tokenizer.encode(
                src_str, tgt_str, maxlen=maxlen
            )
            # 返回一条样本
            yield token_ids, segment_ids


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


strategy = tf.distribute.MirroredStrategy()  # 建立单机多卡策略

with strategy.scope():  # 调用该策略

    bert = build_transformer_model(
        config_path,
        checkpoint_path=None,  # 此时可以不加载预训练权重
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
        return_keras_model=False,  # 返回bert4keras类，而不是keras模型
    )

    model = bert.model  # 这个才是keras模型
    output = CrossEntropy(2)(model.inputs + model.outputs)

    model = Model(model.inputs, output)
    model.compile(optimizer=Adam(1e-5))
    model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)  # 必须最后才加载预训练权重


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.
        self.lowest = 1e10
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        val_loss = logs['val_loss']
        logs_tmep = logs['loss']
        file_name = f'_{project_name}_time_{time}_loss_{logs_tmep}_epoch_{str(self.epoch)}_valloss_{val_loss}_'

        condition_epoch = (self.epoch) % 2 == 1

        if condition_epoch:
            metrics = self.evaluate(valid_data)  # 评测模型
            file_obj = open(f'./vallog/{file_name}val.json', 'w', encoding='utf-8')
            json.dump(metrics, file_obj, indent=2)
            file_obj.close()
            condition_blue = metrics['bleu'] > self.best_bleu
            del metrics['pred']
            print('valid_data:', metrics)
            if condition_blue:
                self.best_bleu = metrics['bleu']

        condition_loss = logs['loss'] < self.lowest
        if condition_loss:
            self.lowest = logs['loss']
            model.save_weights(f'./save_model/train_model{file_name}model.weights')  # 保存模型
            print(f'save model path :{file_name}')

        global  logfile
        logfile.append({'epoch': self.epoch,
                             'loss': logs_tmep,
                             'val_loss': val_loss,
                             'time': time,
                             'file_name': f'train_model{file_name}model.weights'})


        _file = open(f'{project_name}_train_log.json','w',encoding='utf-8')
        json.dump(logfile, _file,indent=2)
        _file.close()

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        save_data = list()
        for sent in tqdm(data):
            total += 1
            tgt = ''.join(sent['tgt']).lower()
            pred_tgt = ''.join(autotitle.generate(sent['src'], topk=topk)).lower()
            if pred_tgt.strip():
                scores = self.rouge.get_scores(hyps=pred_tgt, refs=tgt)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu_temp = sentence_bleu(
                    references=[tgt.split(' ')],
                    hypothesis=pred_tgt.split(' '),
                    smoothing_function=self.smooth
                )
                bleu += bleu_temp
                sent['pred'] = pred_tgt
                sent['rouge'] = scores[0]
                sent['bleu'] = bleu_temp
                save_data.append(sent)

        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total

        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
            'pred': save_data
        }


if __name__ == '__main__':

    # 加载数据集
    train_data = load_data(json_name + 'train.json')
    valid_data = load_data(json_name + 'vaild.json')

    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size)

    evaluator = Evaluator()
    tf_callbacks = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    dataset = train_generator.to_dataset(
        types=('float32', 'float32'),
        shapes=([None], [None]),  # 配合后面的padded_batch=True，实现自动padding
        names=('Input-Token', 'Input-Segment'),
        padded_batch=True
    )  # 数据要转为tf.data.Dataset格式，names跟输入层的名字对应

    try:
        log_file = open(f'{project_name}_train_log.json','r',encoding='utf-8')
        logfile = json.load(log_file)
        log_file.close()
    except Exception as e:
        print(f'null log file! {e.args}')
    try:
        if len(logfile) > 0:
            log_data = logfile[len(logfile) - 1]
            file_name = log_data['file_name']
            file_name = f'./save_model/{file_name}'
            model.load_weights(file_name)
            print(f'load {file_name}')
    except Exception as e:
        print(f'no find mode file {logfile[len(logfile) - 1]}')

    try:
        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            validation_data=valid_generator.forfit(),
            validation_steps=len(valid_generator),
            epochs=epochs,
            callbacks=[evaluator, tf_callbacks]
        )
    except Exception as e:
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model.save_weights(f'./save_model/train_exception_{time}_model.weights')  # 保存模型
        print(f'excepyion = {e.with_traceback()}')

    finally:
        print(f'end time = {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

