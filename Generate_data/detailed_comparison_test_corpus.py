# -*- coding: utf-8 -*-
# @Time : 2023/1/25 17:40
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : greedy_matching.py
# @Project : xuxu
# @Describe : All you think is star River,all star river is you
import json
from xml.dom.minidom import parse  # 调用parse模块
import os
from tqdm import tqdm
import greedy_matching

test_name = 'pan12-detailed-comparison-test-corpus-2012-08-12'
type_ = '03_artificial_low'
type_data = 'mt5'


def writexml(self, writer, indent="", addindent="", newl="", encoding=None, myself=0):
    if encoding is None:
        writer.write('<?xml version="1.0" ?>' + newl)
    else:
        if myself == 0:
            writer.write('<?xml version="1.0" encoding="%s"?>%s' % (
                encoding, newl))
        else:
            pass
    for node in self.childNodes:
        node.writexml(writer, indent, addindent, newl)


def filewrite(filepath, dict_data):
    dom = parse(f'./data/{test_name}/{type_}/{filepath}')  # 使用parser读取xml中
    ref = dom.getElementsByTagName('document')
    suspcious_reference = ref[0].getAttribute('reference')
    susp = open(f'./data/{test_name}/susp/{suspcious_reference}', 'r', encoding='utf-8')
    names = dom.getElementsByTagName("feature")  # 获取所有‘name’的节点
    text1 = susp.read()
    susp.close()
    change_len = 0
    for i in range(len(names)):  # 循坏读取列表中的内容跟
        if names[i].getAttribute('name') == 'plagiarism':
            this_offset = int(names[i].getAttribute('this_offset'))  # 打印节点数据susp
            this_length = int(names[i].getAttribute('this_length'))  # 打印节点数据
            source_offset = int(names[i].getAttribute('source_offset'))  # 打印节点数据sour
            source_length = int(names[i].getAttribute('source_length'))  # 打印节点数据
            source_reference = names[i].getAttribute('source_reference')

            md1 = dict_data[i]['this_offset'] + \
                  dict_data[i]['this_length'] + \
                  dict_data[i]['source_offset'] + \
                  dict_data[i]['source_length']

            md2 = this_offset + this_length + source_offset + source_length

            if md1 == md2:
                # 修改txt文件
                src_gen = ''.join(dict_data[i]['src_gen'])
                tgt_gen = ''.join(dict_data[i]['tgt_gen'])
                text1 = text1[:int(this_offset)] + src_gen + text1[int(this_offset)
                                                                   + int(this_length):]

                # 修改xml文件
                names[i].setAttribute("this_length", str(len(src_gen)))
                names[i].setAttribute("this_offset", str(this_offset + change_len))
                change_len = change_len + (len(src_gen) - this_length)
            else:
                print('index err!')
    susp = open(f'./data/modify/{test_name}/susp/{suspcious_reference}', 'w', encoding='utf-8')
    susp.write(text1)
    susp.close()

    # fp = open(f'./data/modify/{test_name}/{type_}/{filepath}', 'w', encoding='utf-8')
    # writexml(dom, fp, indent='', newl='', addindent='', encoding='utf-8', myself=1)
    # fp.close()


def _fix_data(text: dict):
    re_dict = dict()
    for data in tqdm(list(text.keys())):
        temp_list = list()
        for data_ in text[data]:
            data_['src'] = greedy_matching.wash_data(data_['src'].replace('\n', ' '))
            data_['tgt'] = greedy_matching.wash_data(data_['tgt'].replace('\n', ' '))
            temp_list.append(data_)
        re_dict[data] = temp_list
    return re_dict


def _sent_tokenize(text_data: dict):
    re_dict = dict()
    for data in tqdm(list(text_data.keys())):
        temp_list = list()
        for data_ in text_data[data]:
            data_['src_s'] = greedy_matching._merge_sent(
                greedy_matching._fix_sent(greedy_matching.sent_tokenize(data_['src'])))
            data_['tgt_s'] = greedy_matching._merge_sent(
                greedy_matching._fix_sent(greedy_matching.sent_tokenize(data_['tgt'])))
            temp_list.append(data_)
        re_dict[data] = temp_list
    return re_dict


if __name__ == '__main__':
    dir_str = f'./data/{test_name}_{type_}_{type_data}/'
    dir_gen = os.listdir(dir_str)
    sent_data = dict()
    for line in tqdm(dir_gen):
        _file = open(dir_str + line, 'r', encoding='utf-8')
        _data = json.load(_file)
        _file.close()
        sent_data[_data[0]['name']] = _data
    dir_test = os.listdir(f'./data/{test_name}/{type_}')
    for line in tqdm(dir_test[1:]):
        rec = filewrite(line, sent_data[line])

    file_ = open(f'./data/{test_name}_fix.json', 'w', encoding='utf-8')
    json.dump(sent_data, file_, indent=2)
    file_.close()
