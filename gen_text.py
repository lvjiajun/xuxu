import json
from xml.dom.minidom import parse  # 调用parse模块

dirs_name = 'pan12-text-alignment-training-corpus-2012-03-16'

D_data = list()


def filewrite(filepath, rec):
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
    tol = 0
    rec = 0
    with open('./data/name.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            rec = filewrite(line, rec)
        f = open('./data/' + dirs_name + '.json', 'w', encoding='utf-8')
        json.dump(D_data, f, indent=2)
        f.close()
