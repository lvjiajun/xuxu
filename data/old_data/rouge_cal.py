import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from rouge import Rouge


def rouge_cal(data: list):
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    re_data = list()
    for i in tqdm(data):
        src = i['title'].replace('\n', ' ').replace('\\', '').replace('"', '')
        tgt = i['content'].replace('\n', ' ').replace('\\', '').replace('"', '')
        if len(src) < 10 or len(tgt) < 10:
            continue
        score = rouge.get_scores(src, tgt)
        var = sentence_bleu(
            references=[src.split(' ')],
            hypothesis=tgt.split(' '),
            smoothing_function=smooth
        )
        re_data.append({'src': src, 'tgt': tgt, 'score': score[0], 'blue': var})
    return re_data


if __name__ == '__main__':
    f = open('./train.json')
    s = json.load(f)
    h = rouge_cal(s)
    f = open('./train_rouge.json', 'w', encoding='utf-8')
    json.dump(h, f, indent=2)
    pass
