import json
import abc
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
import difflib

dirs_name = 'pan12-text-alignment-training-corpus-2012-03-16'
dict_list = list()


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


class Handler(metaclass=ABCMeta):
    @abstractmethod
    def handle(self, data):
        pass


# 保存JSON数据
class Save_json(Handler):
    def handle(self, data):
        with open(f'./data/train_vaild.json', 'r', encoding='utf-8') as f:
            json.dump(data, f)
            f.close()


# 进行数据的分割，使用的是sklearn 的train_test_split
class Train_test_vaild_split(Handler):
    def __init__(self):
        self.next = Save_json()

    def handle(self, data):
        train, vaild = train_test_split(data, test_size=0.25, random_state=0)
        self.next.handle({'train': train, 'vaild': vaild})


# 进行数据的匹配
class Matching_split(Handler):
    def __init__(self):
        self.next = Train_test_vaild_split()

    def handle(self, data):
        re_data = list()
        for data_ in data:
            text1 = text_segmentate(data_['text1'], maxlen=128, seps='.!?')
            text2 = text_segmentate(data_['text2'], maxlen=128, seps='.!?')
            data_['text'] = self.diff(text1, text2)
            re_data.append(dict(data_))
        self.next.handle(re_data)

    def jaccard(self, a_text, b_text):
        pass

    def rouge(self, a_text, b_text: list):
        for text_ in a_text:
            while len(b_text) != 0:
                b_text_ = b_text[0]
                700
                del b_text[0]

    def diff(self, a_text, b_text):
        re_data = list()
        for text_ in a_text:
            b_text_ = difflib.get_close_matches(text_, b_text, n=1, cutoff=0.001)
            b_text.remove(b_text_)
            re_data.append({'text1': text_, 'text2': b_text_})
        return re_data

    def tf_idf(self, a_text, b_text):
        pass


# 把换行符去除为空格的接口
class Handling_Newline(Handler):
    def __init__(self):
        self.next = Matching_split()

    def handle(self, data):
        re_data = list()
        for data_ in data:
            data_['text1'] = data_['text1'].replace('\n', ' ')
            data_['text2'] = data_['text2'].replace('\n', ' ')
        re_data.append(dict(data_))
        self.next.handle(re_data)


# 去掉一写特殊的标点符号
class Handling_special_symbols(Handler):
    def __init__(self):
        self.next = Handling_Newline()

    def handle(self, data, pun_data):
        re_data = list()
        for data_ in data:
            text1 = data_['text1']
            text2 = data_['text2']
            for pd in list(pun_data):
                text1.replace(pd, '')
                text2.replace(pd, '')
            data_['text1'] = text1
            data_['text2'] = text2
            re_data.append(dict(data_))
            self.next.handle(re_data)


if __name__ == '__main__':
    with open(f'./data/{dirs_name}.json', 'r', encoding='utf-8') as f:
        dict_list = json.load(f)
        Del = Handling_special_symbols()
        Del.handle(dict_list, [',', '.'])
