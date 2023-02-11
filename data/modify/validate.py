# -*- coding: utf-8 -*-
# @Time : 2023/2/6 15:26
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : validate.py
# @Project : xuxu
import json
from xml.dom.minidom import parse  # 调用parse模块
import os
from tqdm import tqdm
# 测试数据集
test_name: str = 'pan12-detailed-comparison-test-corpus-2012-08-12'
# 数据集类型
type_name: str = '03_artificial_low'
D_data = list()
L_data = dict()


def filewrite(filepath: str, dirs_name: str) -> list:
    """
    解析XML生成JSON匹配判断的方法
    :param filepath: pan12数据集文件夹中文件名
    :param dirs_name: pan12数据集 train and test
    :return:拆分好的数据段
    """
    dom = parse('./' + dirs_name + f'/{type_name}/' + filepath)  # 使用parser读取xml中
    ref = dom.getElementsByTagName('document')
    suspcious_reference = ref[0].getAttribute('reference')
    susp = open('./' + dirs_name + '/susp/' + suspcious_reference, 'r', encoding='utf-8')
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
            sour = open('./' + dirs_name + '/src/' + str(source_reference), 'r', encoding='utf-8')

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

    dir_list = os.listdir(f'./{test_name}/03_artificial_low')
    for line in tqdm(dir_list[1:]):
        rec = filewrite(line, test_name)
        D_data.append(rec)
    f = open('./' + test_name + '.json', 'w', encoding='utf-8')
    json.dump(D_data, f, indent=2)
    f.close()