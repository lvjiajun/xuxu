# -*- coding: utf-8 -*-
# @Time : 2023/3/15 21:12
# @Author : cloudjumper
# @Email : lvjiajun@outlook.com
# @File : rewrite_test.py
# @Project : xuxu
'''
本项目是一个改写评估的项目
'''
import os
import json
from nltk.tokenize import word_tokenize


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def get_rouge(a, b):
    evaluated_1grams = _get_word_ngrams(1, a)
    reference_1grams = _get_word_ngrams(1, b)
    evaluated_2grams = _get_word_ngrams(2, a)
    reference_2grams = _get_word_ngrams(2, b)

    rouge_1 = cal_rouge(evaluated_1grams, reference_1grams)['f']
    rouge_2 = cal_rouge(evaluated_2grams, reference_2grams)['f']
    return rouge_1 + rouge_2


def jaccard(a, b):
    return len(set(a).intersection(set(b))) / len(set(a).union(set(b)))


def is_all_chinese(strs):
    num = 0
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            num += 1
            if num > 12:
                return True
    return False


if __name__ == '__main__':
    for name in os.listdir('./'):
        if name.find('.json') != -1:
            with open(name, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                file.close()
                for _idx, sent in enumerate(json_data):
                    if is_all_chinese(sent['src']):
                        src = list(sent['src'])
                        tgt = list(sent['tgt'])
                    else:
                        src = word_tokenize(sent['src'])
                        tgt = word_tokenize(sent['tgt'])
                    sent['jaccard'] = jaccard(set(src), set(tgt))
                    #sent['rouge_f'] = get_rouge(src, tgt)
                    json_data[_idx] = sent
                with open(name, 'w', encoding='utf-8') as file_w:
                    json.dump(json_data, file_w, ensure_ascii=False, indent=2)
                    file_w.close()
