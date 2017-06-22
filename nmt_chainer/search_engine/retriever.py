#!/usr/bin/env python

from __future__ import division
from abc import ABCMeta, abstractmethod
from itertools import chain


"""retriever.py: The main class for retrieving translation pairs"""
__author__ = "Ryota Nakao"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "nakario@gmail.com"
__status__ = "Development"


class BaseEngine:
    __metaclass__ = ABCMeta

    @abstractmethod
    def search(self, query):
        pass


class Retriever:
    def __init__(self, engine, similarity, training=False):
        self.__engine = engine
        self.__similarity = similarity
        self.__training = training

    def retrieve(self, src):
        subset = self.__engine.search(src)
        if self.__training:
            subset = filter(lambda x: x[0] != src, subset)
        subset = self.__rerank(subset, src)
        R = []
        coverage = 0
        src_symbols = src.split(" ")
        for pair in subset:
            sentences = [pair_[0] for pair_ in R] + [pair[0]]
            symbols = flatten([s.split(" ") for s in sentences])
            c_tmp = sum([s in symbols for s in src_symbols]) / len(src_symbols)
            if c_tmp > coverage:
                coverage = c_tmp
                R.append(pair)
        return R

    def __rerank(self, pairs, src):
        return sorted(pairs, reverse=True, key=lambda pair: self.__similarity(pair[0], src))


def flatten(x):
    return list(chain.from_iterable(x))
