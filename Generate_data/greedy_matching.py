# -*- coding: utf-8 -*-
# @Time : 2022/12/16 10:45
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : greedy_matching.py
# @Project : xuxu
# @Describe : All you think is star River,all star river is you
import json
import re
import time
from rouge import Rouge
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
import detailed_comparison_test_corpus

rouge = Rouge()

# from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

_hyper_parameter = dict()


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    :param text:
    :param maxlen:
    :param seps:
    :param strips:
    :return:
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


def _shrink_symbol(text: str):
    text = text.replace('  ', ' ')
    text = text.replace('**', '*')
    if text.find('  ') > -1 or text.find('**') > -1:
        return _shrink_symbol(text)
    else:
        return text.replace(' *', '').replace('* ', '')


def wash_data(text: str, pun_data=None):
    text = text \
        .replace('"', '') \
        .replace('/', '') \
        .replace('\\', '')
    text = _shrink_symbol(text)
    if text[0] == ' ' or text[0] == '.':
        text = text[1:]
    if pun_data is not None:
        for pun in list(pun_data):
            text = text.replace(pun, '')
    return text


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """
    将文本按照标点符号划分为若干个短句
    :param text:
    :param maxlen:
    :param seps:
    :param strips:
    :return:
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


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def rouge_matching(doc_sent_list: list, abstract_sent_list: list, greedy: bool = 1) -> list:
    """
    doc_list 输入的是文章内容
    abstract_sent_list输入的是摘要内容1
    文章内容的长度大于摘要内容
    :param doc_sent_list:
    :param abstract_sent_list:
    :param greedy:
    :return:
    """
    if len(abstract_sent_list) > len(doc_sent_list):
        return None

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s).lower()

    abstracts = [_rouge_clean(' '.join(s)).split() for s in abstract_sent_list]
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = [_get_word_ngrams(1, [sent]) for sent in abstracts]
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = [_get_word_ngrams(2, [sent]) for sent in abstracts]

    impossible_sents = []
    for s in range(len(reference_1grams)):
        max_rouge = -1
        max_index = -1
        combin_list = list(range(len(evaluated_1grams)))
        for i in combin_list:
            rouge_1 = cal_rouge(evaluated_1grams[i], reference_1grams[s])['f']
            rouge_2 = cal_rouge(evaluated_2grams[i], reference_2grams[s])['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > max_rouge:
                max_rouge = rouge_score
                max_index = i
        impossible_sents.append({'index': max_index, 'score': max_rouge})
        if (greedy is True) and (max_index != -1):
            del combin_list[max_index]
    return impossible_sents


def preprocess(src, tgt, oracle_ids):
    preprocess_list = []
    for i in range(len(oracle_ids)):
        index_oracle_ids = oracle_ids[i]
        preprocess_list.append({'src': tgt[i],
                                'tgt': src[index_oracle_ids['index']],
                                'rouge': index_oracle_ids['score']})
    return preprocess_list


def get_smart_common_words(path='./smart_common_words.txt'):
    with open(file=path, mode='r', encoding="utf-8") as file:
        data = file.readlines()
        file.close()
        return set(data)


def _sent_tokenize(text_data: list):
    re_list = []
    for data in tqdm(text_data):
        data['src_s'] = _merge_sent(text=_fix_sent(sent_tokenize(data['src'])),
                                    min_len=_hyper_parameter['min_len'])
        data['tgt_s'] = _merge_sent(text=_fix_sent(sent_tokenize(data['tgt'])),
                                    min_len=_hyper_parameter['min_len'])
        re_list.append(dict(data))
    return re_list


def _word_tokenize(text_data: list):
    re_list = []
    # nlp = StanfordCoreNLP(r'./stanford-corenlp-4.5.1')
    for data in tqdm(text_data):
        src_w = []
        tgt_w = []
        for src in data['src_s']:
            src_w.append(word_tokenize(src))
        for tgt in data['tgt_s']:
            tgt_w.append(word_tokenize(tgt))
        data['src_w'] = src_w
        data['tgt_w'] = tgt_w
        re_list.append(data)
    # nlp.close()
    return re_list


def _fix_data(text: list):
    re_list = []
    for data_list in tqdm(text):
        for data in data_list:
            data['src'] = wash_data(data['src'].replace('\n', ' '))
            data['tgt'] = wash_data(data['tgt'].replace('\n', ' '))
            re_list.append(data)
    return re_list


def _fix_sent(text: list, min_len: int = 2) -> list:
    """
    合并部分碎片句子
    :param text:输入片段的list
    :param min_len:最小句子长度
    :return:
    """
    temp_sent = []
    sp_symbol = []
    re_sent = []
    for sent in text:
        if len(sent) < min_len:
            continue

        if sent[0] == ' ' or sent[0] == '.':
            sent = sent[1:]

        if (sent.find('[') > -1 and sent.find(']') == -1) \
                or (sent.find('{') > - 1 and sent.find('}') == -1) \
                or (sent.find('(') > - 1 and sent.find(')') == -1):

            if sent.find('[') > -1:
                sp_symbol.append('[')
            elif sent.find('{') > -1:
                sp_symbol.append('{')
            else:
                sp_symbol.append('(')

            temp_sent.append(sent)
            continue

        if len(temp_sent) > 0:
            temp_sent.append(sent)
            if (sent.find('[') == -1 and sent.find(']') > -1) \
                    or (sent.find('{') == - 1 and sent.find('}') > -1) \
                    or (sent.find('(') == - 1 and sent.find(')') > -1):
                re_sent.append(''.join(temp_sent))
                temp_sent.clear()
                del sp_symbol[len(sp_symbol) - 1]
            continue

        re_sent.append(sent)
    return re_sent


def _match_data_(data):
    if len(data['src_w']) >= len(data['tgt_w']):
        impossible_sents = rouge_matching(data['src_w'], data['tgt_w'])
        data['match'] = preprocess(data['src_s'], data['tgt_s'], impossible_sents)
    else:
        impossible_sents = rouge_matching(data['tgt_w'], data['src_w'])
        temp = preprocess(data['tgt_s'], data['src_s'], impossible_sents)
        temp_list = []
        for temp_ in temp:
            temp_s = temp_['tgt']
            temp_['tgt'] = temp_['src']
            temp_['src'] = temp_s
            temp_list.append(temp_)
        data['match'] = temp_list
    return data


def _match_data(text: list, worker=4):
    process_pool = ProcessPoolExecutor(max_workers=worker)
    re_list = []
    re_list_f = []
    for data in tqdm(text):
        re_list_f.append(process_pool.submit(_match_data_, data))
    for data in tqdm(re_list_f):
        data_ = data.result()
        del data_['src_w']
        del data_['tgt_w']
        re_list.append(data_)
    return re_list


def _merge_sent(text: list, min_len: int = 16) -> list:
    """
    合并长度不够的句子，采用向下合并的方法
    :param text:
    :param min_len:
    :return:
    """
    re_list = []
    temp_list = []
    for data in text:
        if (len(data) < min_len) and ((data.find('!') > -1) or (data.find('?') > -1)):
            temp_list.append(data)
        if (len(data) < min_len) and (len(data.split(' ')) < 4):
            temp_list.append(data)
        else:
            re_list.append(''.join(temp_list) + data)
            temp_list.clear()
    return re_list


def _merge_sent_by_word(text: list, token_len=128):
    for fragment in text:
        src_temp = ''
        src_temp_len = 0
        src_list = list()
        for src in fragment['src']:
            src_token = word_tokenize(src)
            if src_temp_len < token_len:
                src_temp = src_temp + src
                src_temp_len = src_temp_len + len(src_token)
            else:
                src_list.append(src_temp)
                src_temp = ''
                src_temp_len = 0
        src_list.append(src_temp)


def pick_pearl(text: list,
               min_score: float = 0.4,
               max_socre: float = 1.8,
               part_score: float = 0.36):
    """
    挑选珍珠行动，通过rouge得分
    高低继续筛选掉一些匹配错误的句子
    :param text:
    :param min_score:
    :param max_socre:
    :param part_score:
    :return: 返回pearl句子对
    """
    sent_re = []
    lost_re = []
    for data in tqdm(text):
        sent_list = []
        lost_list = []
        rouge_agv = 0
        for sent_p in data['match']:
            rouge_agv += sent_p['rouge']
            if min_score < sent_p['rouge'] < max_socre:
                sent_list.append(sent_p)
            else:
                lost_list.append(sent_p)
        if len(data['match']) > 0:
            rouge_agv /= len(data['match'])
        else:
            rouge_agv = 0

        data['rouge_agv'] = rouge_agv

        if rouge_agv > part_score:
            del data['match']
            data['pass_data'] = sent_list
            data['flaw_data'] = lost_list
            sent_re.append(data)
        else:
            lost_re.append(data)
    return sent_re, lost_re


def rainbow():
    y = 2.5
    while y >= -1.6:
        x = -3.0
        while x <= 4.0:
            if (x * x + y * y - 1) ** 3 <= 3.6 * x * x * y * y * y or (
                    -2.4 < x < -2.1 and 1.5 > y > -1) or (
                    ((2.5 > x > 2.2) or (3.4 < x < 3.7)) and -1 < y < 1.5) or (
                    -1 < y < -0.6 and 3.7 > x > 2.2):
                print('*', end="")
            else:
                print(' ', end="")
            x += 0.1
        print()
        time.sleep(0.25)
        y -= 0.2


def split_data(text: list):
    re_list = []
    for data in tqdm(text):
        re_list += data['pass_data']
    return train_test_split(re_list, test_size=0.25, random_state=0)


if __name__ == '__main__':
    path = 'data'
    dirs_name: str = 'pan12-text-alignment-training-corpus-2012-03-16'
    test_name: str = 'pan12-detailed-comparison-test-corpus-2012-08-12'
    # 数据集类型
    type_name: str = '05_translation'

    '''rainbow       pop data'''
    # rainbow()
    try:
        _file = open(f'greedy_matching.json', 'r', encoding='utf-8')
        _hyper_parameter = json.load(_file)
        _file.close()
        dirs_name = _hyper_parameter["dirs_name"]
        test_name = _hyper_parameter["test_name"]
        type_name = _hyper_parameter["type"]
    except Exception as e:
        _hyper_parameter = {"min_rouge": 0.4,
                            "max_rouge": 1.8,
                            "part_score": 0.36,
                            "greedy": 1,
                            "min_len": 16,
                            "token_len": 128,
                            "max_workers": 4}

    print(_hyper_parameter)
    print('\n\n\n')
    json_name = f'{dirs_name}_{type_name}'

    _file = open(f'./{path}/{json_name}.json', 'r', encoding='utf-8')
    _data = json.load(_file)
    _file.close()

    _file = open(f'./{path}/{json_name}_sent.json', 'w', encoding='utf-8')
    _data = _fix_data(_data)
    sent_data = _sent_tokenize(_data)
    json.dump(sent_data, _file, indent=2)
    _file.close()

    _file = open(f'./{path}/{json_name}_word.json', 'w', encoding='utf-8')
    word_data = _word_tokenize(sent_data)
    json.dump(word_data, _file, indent=2)
    _file.close()

    word_data = json.load(open(f'./{path}/{json_name}_word.json'))

    _file = open(f'./{path}/{json_name}_match.json', 'w', encoding='utf-8')
    match_data = _match_data(word_data)
    json.dump(match_data, _file, indent=2)
    _file.close()

    match_data = json.load(open(f'./{path}/{json_name}_match.json'))
    sent, lose = pick_pearl(text=match_data,
                            min_score=_hyper_parameter['min_rouge'],
                            max_socre=_hyper_parameter['max_rouge'],
                            part_score=_hyper_parameter['part_score'])
    _file = open(f'./{path}/{json_name}_match_pass.json', 'w', encoding='utf-8')
    json.dump(sent, _file, indent=2)
    _file.close()

    _file = open(f'./{path}/{json_name}_match_loss.json', 'w', encoding='utf-8')
    json.dump(lose, _file, indent=2)
    _file.close()

    sent = json.load(open(f'./{path}/{json_name}_match_pass.json'))
    train, vaild = split_data(sent)

    _file = open(f'./{path}/{json_name}_match_train.json', 'w', encoding='utf-8')
    json.dump(train, _file, indent=2)
    _file.close()

    _file = open(f'./{path}/{json_name}_match_vaild.json', 'w', encoding='utf-8')
    json.dump(vaild, _file, indent=2)
    _file.close()

    print('\n\n\n')
    json_name = f'{test_name}_{type_name}'

    _file = open(f'./{path}/{json_name}.json', 'r', encoding='utf-8')
    _data = json.load(_file)
    _file.close()

    _file = open(f'./{path}/{json_name}_sent.json', 'w', encoding='utf-8')
    _data = detailed_comparison_test_corpus._fix_data(_data)
    sent_data = detailed_comparison_test_corpus._sent_tokenize(_data)
    json.dump(sent_data, _file, indent=2)
    _file.close()
