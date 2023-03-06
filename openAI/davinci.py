# -*- coding: utf-8 -*-
# @Time : 2023/2/23 20:09
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : davinc.py
# @Project : xuxu
import json
import openai
from tqdm import tqdm
if __name__ == '__main__':
    api_index = 0
    api_key = []


    def davinci_dialogue(text, max_token):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=text,
            temperature=0.7,
            max_tokens=int(max_token),
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=1
        )
        return response["choices"][0]["text"].strip()


    def chatgpt_dialogue(text):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}]
        )
        return completion['choices'][0]['message']['content']


    save_data = []
    data = {}
    try:
        with open('config.json', 'r', encoding='utf-8') as file:
            config = json.load(file)
            api_key = config['api_key']
            file_name = config['file_name']
            file_key = config['file_key']
            prompt_start = config['prompt_start']
            prompt_end = config['prompt_start']
            file.close()
            if len(api_key) == 0:
                print('config Missing parameter!')
            else:
                openai.api_key = api_key[api_index]
                with open(file_name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for idx_num, key in enumerate(tqdm(data.keys())):
                    sent_list = data[key]
                    for idx, fix_sent in enumerate(sent_list):
                        fix_data = []
                        if "src_gen" not in fix_sent:
                            for idx_, s_sent in enumerate(fix_sent['src_list']):
                                # pred = davinci_dialogue(prompt_start +
                                #                         s_sent['text'] +
                                #                         prompt_end,
                                #                         s_sent['len'] + (s_sent['len'] / 4))
                                pred = chatgpt_dialogue(prompt_start + s_sent['text'])
                                fix_data.append(pred)

                            fix_sent['src_gen_list'] = fix_data
                            fix_sent['src_gen'] = ''.join(fix_data)

                            sent_list[idx] = fix_sent
                    data[key] = sent_list
                    if idx_num % 5 == 0:
                        with open(file_name, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                            f.close()
                api_index += 1
                api_index = api_index % len(api_key)
    except Exception as e:
        print(e.args)
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            f.close()
    finally:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            f.close()
