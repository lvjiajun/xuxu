import json
from xml.dom.minidom import parse  # 调用parse模块
import os
from tqdm import tqdm

dir_name = 'pan12-text-alignment-training-corpus-2012-03-16'
test_name = 'pan12-detailed-comparison-test-corpus-2012-08-12'
D_data = list()


def filewrite(filepath, dirs_name):
    rec = 0
    dom = parse('./data/' + dirs_name + '/03_artificial_low/' + filepath)  # 使用parser读取xml中
    ref = dom.getElementsByTagName('document')
    suspcious_reference = ref[0].getAttribute('reference')
    susp = open('./data/' + dirs_name + '/susp/' + suspcious_reference, 'r', encoding='utf-8')
    names = dom.getElementsByTagName("feature")  # 获取所有‘name’的节点
    text1 = susp.read()
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
            D_data.append({'name': filepath,
                           'num': i,
                           'src': text1_temp,
                           'tgt': text2,
                           'this_offset': this_offset,
                           'this_length': this_length,
                           'source_offset': source_offset,
                           'source_length': source_length})
            rec += 1
    susp.close()
    sour.close()
    return rec


if __name__ == '__main__':
    dir_list = os.listdir(f'./data/{dir_name}/03_artificial_low')
    for line in tqdm(dir_list[1:]):
        rec = filewrite(line, dir_name)
    f = open('./data/' + dir_name + '.json', 'w', encoding='utf-8')
    json.dump(D_data, f, indent=2)
    f.close()

    D_data.clear()
    dir_list = os.listdir(f'./data/{test_name}/03_artificial_low')
    for line in tqdm(dir_list[1:]):
        rec = filewrite(line, test_name)
    f = open('./data/' + test_name + '.json', 'w', encoding='utf-8')
    json.dump(D_data, f, indent=2)
    f.close()
