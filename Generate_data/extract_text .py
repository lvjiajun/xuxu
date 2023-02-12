# -*- coding: utf-8 -*-
# @Time : 2022/11/29 20:15
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : extract_text .py
# @Project : xuxu
# @Describe : All you think is star River,all star river is you
import json
import time
from xml.dom.minidom import parse  # 调用parse模块
import os
from tqdm import tqdm

# 训练数据集
dir_name: str = 'pan12-text-alignment-training-corpus-2012-03-16'
# 测试数据集
test_name: str = 'pan12-detailed-comparison-test-corpus-2012-08-12'
# 数据集类型
type_name: str = '04_artificial_high'
D_data = list()
L_data = dict()


def sunshine():
    print('\n'.join([''.join([('Love'[(x - y) % len('Love')] if ((x * 0.05) ** 2 + (y * 0.1) ** 2 - 1) ** 3 - (
            x * 0.05) ** 2 * (y * 0.1) ** 3 <= 0 else ' ') for x in range(-30, 30)]) for y in range(15, -15, -1)]))


def filewrite(filepath: str, dirs_name: str) -> list:
    """
    解析XML生成JSON匹配判断的方法
    :param filepath: pan12数据集文件夹中文件名
    :param dirs_name: pan12数据集 train and test
    :return:拆分好的数据段
    """
    dom = parse(f'./data/{dirs_name}/{type_name}/' + filepath)  # 使用parser读取xml中
    ref = dom.getElementsByTagName('document')
    suspcious_reference = ref[0].getAttribute('reference')
    susp = open('./data/' + dirs_name + '/susp/' + suspcious_reference, 'r', encoding='utf-8')
    names = dom.getElementsByTagName("feature")  # 获取所有‘name’的节点
    text1 = susp.read()
    temp_list = list()
    for i in range(len(names)):  # 循坏读取列表中的内容跟
        if names[i].getAttribute('name') == 'plagiarism':
            this_offset = int(names[i].getAttribute('this_offset'))  # 打印节点数据susp
            this_length = int(names[i].getAttribute('this_length'))  # 打印节点数据
            source_offset = int(names[i].getAttribute('source_offset'))  # 打印节点数据sour
            source_length = int(names[i].getAttribute('source_length'))  # 打印节点数据
            source_reference = names[i].getAttribute('source_reference')
            sour = open('./data/' + dirs_name + '/src/' + str(source_reference), 'r', encoding='utf-8')

            text1_temp = text1[int(this_offset): int(this_offset + this_length)]

            text2 = sour.read()
            text2 = text2[int(source_offset): int(source_offset + source_length)]
            temp_list.append({'name': filepath,
                              'num': i,
                              'src': text1_temp,
                              'tgt': text2,
                              'this_offset': this_offset,
                              'this_length': this_length,
                              'source_offset': source_offset,
                              'source_length': source_length})
    susp.close()
    sour.close()
    return temp_list


if __name__ == '__main__':
    sunshine()

    dir_list = os.listdir(f'./data/{dir_name}/{type_name}')
    for line in tqdm(dir_list[1:]):
        if line[-3:] == 'xml':
            rec = filewrite(line, dir_name)
            D_data.append(rec)
    f = open(f'./data/{dir_name}_{type_name}.json', 'w', encoding='utf-8')
    json.dump(D_data, f, indent=2)
    f.close()

    dir_list = os.listdir(f'./data/{test_name}/{type_name}')
    for line in tqdm(dir_list):
        if line[-3:] == 'xml':
            rec = filewrite(line, test_name)
            L_data[line] = rec
    f = open(f'./data/{test_name}_{type_name}.json', 'w', encoding='utf-8')
    json.dump(L_data, f, indent=2)
    f.close()
