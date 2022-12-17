
#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933
from __future__ import print_function

import glob
import sys
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab


from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
sys.setrecursionlimit(100000) # 最大递归设置为十万



# 用于将控制台所有输出保存至文件，需放在代码最上面
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w','utf-8')


    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(stream=sys.stdout)



# 基本参数
maxlen = 128
batch_size = 4
steps_per_epoch = 1000
epochs = 1000
rouge_1 = 0.

# bert配置
config_path = 'H:/下载/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'H:/下载/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'H:/下载/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/vocab.txt'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

def load_data(filename):
    D = []
    k=0
    for l in filename:
        print(l)
        text1=open(l,'r',encoding='utf-8')
        text=text1.read()
        title, content = text.split('\t')
        D.append((title, content))
        print(D[k])
        print(k)
        k=k+1
        text1.close()
    return D


txts = glob.glob('G:\\unilm text\\train text\\*.txt')#获取该路径下所有txt文件
txts2 = glob.glob('G:\\unilm text\\train text\\*.txt')
train_data = load_data(txts)
valid_data = load_data(txts2)
#test_data = load_test_data(para.data_root+"/test.txt")


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, pair in self.sample(random):
            s_susp, s_src = pair
            token_ids, segment_ids = tokenizer.encode(
                s_susp, s_src, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


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


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()

def just_show():
    s1 = u"No. And so in like manner with other practical skills,--the geometrician's, astronomer's, professional reciter's. None of these he discovers is what Euthydemus aims at. He hopes to become a great politician and statesman. "
    s2 = u"No, not that; I think she was invited, but said to herself that she could not bear to go there and see another young woman touching heads with her husband over an Italian book and making thrilling hand-contacts with him accidentally."
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))
    print()

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


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_rouge = 0


    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['rouge-1'] > self.best_rouge:
            self.best_rouge = rouge_1
            model.save_weights(r'D:\unlim model\best.model')  # 保存模型
        print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1 = 0
        for title, content in tqdm(data):
            total += 1
            title = ' '.join(title)
            pred_title = ' '.join(autotitle.generate(content, topk))
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]['rouge-1']['f']
        rouge_1 /= total
        return {
            'rouge-1': rouge_1,
        }

if __name__ == '__main__':

    evaluator = Evaluate()

    train_generator = data_generator(train_data, batch_size)

    val_generator = data_generator(valid_data, batch_size)
    for i in range(10000):
        model.fit_generator(train_generator.forfit(),
                            steps_per_epoch=len(train_generator),
                            validation_data=val_generator.forfit(),
                            validation_steps=len(val_generator),
                            epochs=epochs,
                            callbacks=[evaluator])
else:
    model.load_weights(r'D:\unlim model')