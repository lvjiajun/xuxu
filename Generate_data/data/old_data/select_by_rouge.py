import json
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    f = open('./train_rouge.json')
    data = json.load(f)
    f.close()
    re_data = list()
    for i in tqdm(data):
        f = i['score']['rouge-1']['f']
        b = i['blue']
        if (f > 0.4) and (f < 0.92) and (len(i['src']) > 20) and (len(i['tgt']) > 20) and b > 0.2:
            i['src'] = re.sub(r'[^a-zA-Z0-9 ]', '', i['src'])
            i['tgt'] = re.sub(r'[^a-zA-Z0-9 ]', '', i['tgt'])
            re_data.append(i)
    f_t = open('./train_rouge_clear.json', 'w', encoding='utf-8')
    f_v = open('./valid_rouge_clear.json', 'w', encoding='utf-8')
    train, vaild = train_test_split(re_data, test_size=0.1, random_state=0)
    json.dump(train, f_t, indent=2)
    json.dump(vaild, f_v, indent=2)
    f_t.close()
    f_v.close()
    print(f'len of list {len(re_data)}')
