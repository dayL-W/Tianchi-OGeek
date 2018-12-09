#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os

import jieba
from gensim.models import Word2Vec

BASE_PATH = os.path.join(os.path.dirname(__file__), "..\data")
#RawData = os.path.join(BASE_PATH, "RawData")


def get_sentence(fname="train"):
    fname = "oppo_round1_{fname}_20180929.txt".format(fname=fname)
    file_path = os.path.join(BASE_PATH, fname)
    if not os.path.exists(file_path):
        raise FileNotFoundError("{} Not Found!".format(file_path))

    with open(file_path, "r", encoding="utf-8") as f:
        line = f.readline()

        while line:
            line_arr = line.split("\t")

            query_prediction = line_arr[1]
            sentences = json.loads(query_prediction)
            for sentence in sentences:
                yield sentence

            title = line_arr[2]
            yield title

            line = f.readline()


class MySentence(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        print(self.fname)
        for sentence in get_sentence(self.fname):
            seg_list = jieba.cut(sentence)
            seg_list = list(seg_list)
            yield seg_list


def build_model(fname):
    sentences = MySentence(fname)
    model_name = "{}.bin".format(fname)
    print(model_name)
    my_model = Word2Vec(sentences, size=500, window=10, sg=2, min_count=2)

    my_model.wv.save_word2vec_format(model_name, binary=True)


if __name__ == "__main__":
    build_model(fname="train")
