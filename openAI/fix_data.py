# -*- coding: utf-8 -*-
# @Time : 2023/2/23 21:29
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : fix_data.py
# @Project : xuxu
import json

from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

path: str = './../Generate_data/data'
test_name: str = 'pan12-detailed-comparison-test-corpus-2012-08-12'
dirs_name: str = 'pan12-text-alignment-training-corpus-2012-03-16'
type_name: str = '04_artificial_high'

max_token_config = 1300


def avg_text(text: str):
    max_token_num = word_tokenize(text)
    sent_list = sent_tokenize(text)
    re_text = []
    temp_num = 0
    temp_str = ''
    if len(max_token_num) > max_token_config:
        # 算出需要切分多少个片段
        split_num = int(len(max_token_num) / max_token_config) + 1
        # 算出需要切分片段的长度
        split_len = int(len(max_token_num) / split_num)
        for sent in sent_list:
            word_list = word_tokenize(sent)
            '''计算词的数量'''
            len_wordlist = len(word_list)
            '''是否大于'''
            if temp_num + len_wordlist > split_len:
                '''是否查过最大量'''
                if temp_num + len_wordlist < max_token_config:
                    temp_num += len_wordlist
                    temp_str += sent
                    re_text.append({'text': temp_str, 'len': temp_num})
                    temp_str = ''
                    temp_num = 0
                else:
                    '''假设a+b 太大了只能这样进行拆分,先把先前的字符串推入list，'''
                    re_text.append({'text': temp_str, 'len': temp_num})
                    '''再把temp_str ,temp_num 置 当前值'''
                    temp_str = sent
                    temp_num = len_wordlist
            else:
                '''简单的进行叠加'''
                temp_num += len_wordlist
                temp_str += sent

        if temp_num > 100:
            re_text.append({'text': temp_str, 'len': temp_num})
        else:
            if len(re_text) > 0:
                temp_ = re_text.pop()
                temp_['text'] += temp_str
                temp_['len'] += temp_num
                re_text.append(temp_)
            else:
                re_text.append({'text': temp_str, 'len': temp_num})
        return re_text
    else:
        return [{'text': text, 'len': len(max_token_num)}]


def _shrink_symbol(text: str):
    text = text. \
        replace('  ', ' '). \
        replace('--', '-'). \
        replace('**', '*'). \
        replace('..', '.'). \
        replace('\n\n', '\n')

    if text.find('  ') > -1 or \
            text.find('**') > -1 or \
            text.find('\n\n') > -1 or \
            text.find('..') > -1 or \
            text.find('--') > -1:
        return _shrink_symbol(text)
    else:
        return text \
            .replace(' *', ' ').replace('* ', ' ') \
            .replace('\n ', ' ').replace(' \n', ' ') \
            .replace(' .', ' ').replace('. ', ' ') \
            .replace(' -', ' ').replace('- ', ' ')


def wash_data(text: str, pun_data=None):
    text = _shrink_symbol(text)
    text = text \
        .replace('/', '') \
        .replace('\\', '') \
        .replace('\n', ' ') \
        .replace('_', ' ').replace('-', ' ')
    if text[0] == ' ' or text[0] == '.' or text[0] == '\n' or text[0] == ' ':
        text = text[1:]
    if pun_data is not None:
        for pun in list(pun_data):
            text = text.replace(pun, '')
    text = _shrink_symbol(text)
    return text


if __name__ == '__main__':

    json_name = f'{test_name}_{type_name}'
    file = open(f'./{path}/{json_name}.json', 'rt', encoding='utf-8')
    Data = json.load(file)
    file.close()
    for key in tqdm(list(Data.keys())):
        for idx, data in enumerate(Data[key]):
            src = wash_data(data['src'])
            data['src_list'] = avg_text(src)
            Data[key][idx] = data
    file = open(f'./{json_name}_token.json', 'w', encoding='utf-8')
    json.dump(Data, file, indent=2)
    file.close()

    json_name = f'{dirs_name}_{type_name}'
    file = open(f'./{path}/{json_name}.json', 'rt', encoding='utf-8')
    Data = json.load(file)
    file.close()

    # re_data = {}
    # for sent_sp in tqdm(Data):
    #     name = ''
    #     for idx, sent in enumerate(sent_sp):
    #         name = sent['name']
    #         src = wash_data(sent['src'])
    #         sent['src_list'] = avg_text(src)
    #         sent_sp[idx] = sent
    #     re_data[name] = sent_sp
    #
    # file = open(f'./{json_name}_token.json', 'w', encoding='utf-8')
    # json.dump(re_data, file, indent=2)
    # file.close()
