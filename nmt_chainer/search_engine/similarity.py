#!/usr/bin/env python

from __future__ import division
from nltk import edit_distance


"""similarity.py: similarity score functions"""
__author__ = "Ryota Nakao"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "nakario@gmail.com"
__status__ = "Development"


def fuzzy_char_level_similarity(x, y):
    return 1 - edit_distance(x, y) / max(len(x), len(y))


def fuzzy_word_level_similarity(x, y, sep=None):
    xs = x.split(sep)
    ys = y.split(sep)
    return 1 - edit_distance(xs, ys) / max(len(xs), len(ys))
