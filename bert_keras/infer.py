# -*- coding: utf-8 -*-
# @Time : 2023/2/13 20:08
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : infer.py
# @Project : xuxu
import json
import os

from tqdm import tqdm

test_name: str = "pan12-detailed-comparison-test-corpus-2012-08-12"
type_name: str = "05_translation"
model_name: str = "mt5_keras"
model_file: str = "best_model.weights"
if __name__ == '__main__':
    try:
        _file = open(f'./infer_conf.json', 'r', encoding='utf-8')
        _conf = json.load(_file)
        _file.close()
        test_name = _conf["test_name"]
        type_name = _conf["type_name"]
        model_name = _conf["model_name"]
        model_file = _conf["model_file"]
    except Exception as e:
        print(f"config null {e.args}")
        pass

    if model_name == "mt5_keras":
        from mt5_keras.mt5_Keras import model, AutoTitle, tokenizer, max_t_len

        _pred = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=max_t_len - 1)
    elif model_name == 'umilm_keras':
        from umilm_keras.UmiLM_Keras import model, AutoTitle, tokenizer, maxlen

        _pred = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen - 1)
    else:
        print('err model_name please chick you config.json')

    json_name = f'{test_name}_{type_name}'
    file_name = f'../Generate_data/data/{json_name}_sent.json'
    file_save = f'./{model_name}/text/{json_name}_fix.json'

    model_dir: str = f"./{model_name}/save_model/{model_file}"
    """
    加载历史文件
    """
    if os.path.exists(file_save):
        _file = open(file_save, 'r', encoding='utf-8')
        save_dict: dict = json.load(_file)
    else:
        print(f'not file {file_save}')
        save_dict: dict = dict()

    if not os.path.exists(file_name):
        print(f"data {file_name} null !")
    else:
        _file = open(file_name, 'r', encoding='utf-8')
        infer_data: dict = json.load(_file)
        _file.close()

        infer_set = set(save_dict.keys())
        try:
            print(f"load model ! {model_dir}")
            model.load_weights(model_dir)

            for key in tqdm(infer_data.keys()):
                """
                查找set里面有没有key 如果有key此key放弃
                """
                if key in infer_set:
                    continue
                re_cluster = list()
                for cluster in infer_data[key]:
                    src_infer_list = list()
                    tgt_infer_list = list()
                    for src_text in cluster['src_s']:
                        src_infer_list.append(_pred.generate(src_text, topk=1))

                    for tgt_text in cluster['tgt_s']:
                        tgt_infer_list.append(_pred.generate(tgt_text, topk=1))

                    # 添加数据
                    cluster['src_gen'] = src_infer_list
                    cluster['tgt_gen'] = tgt_infer_list

                    re_cluster.append(cluster)
                save_dict[key] = re_cluster
                re_cluster.clear()

        except IOError as e:
            print(f'err {e.args}')
        finally:
            _file = open(file_save, 'w', encoding='utf-8')
            json.dump(save_dict, _file, indent=2)
            _file.close()
            print("save infer data !")
