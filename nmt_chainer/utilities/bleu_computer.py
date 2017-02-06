#!/usr/bin/env python
"""bleu_computer.py: compute BLEU score"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import argparse, os

from collections import defaultdict
import math, codecs
from itertools import izip
       
class BleuComputer(object):
    def __init__(self):
        self.ngrams_corrects = {1: 0, 2: 0, 3: 0, 4: 0}
        self.ngrams_total = {1: 0, 2: 0, 3: 0, 4: 0}
        self.total_length = 0
        self.ref_length = 0
        
    def copy(self):
        res = BleuComputer()
        res.ngrams_corrects = self.ngrams_corrects.copy()
        res.ngrams_total = self.ngrams_total.copy()
        res.total_length = self.total_length
        res.ref_length = self.ref_length
        return res
        
    def __repr__(self):
        res = []
        res.append("bleu:%f%%   "%(self.bleu() * 100))
        for n in xrange(1, 5):
            if self.ngrams_total[n] == 0:
                assert self.ngrams_corrects[n] == 0
                ratio_n = 1
            else:
                ratio_n = float(self.ngrams_corrects[n]) / self.ngrams_total[n]
            res.append("%i/%i[%f%%]"%(self.ngrams_corrects[n], self.ngrams_total[n], 100.0 *ratio_n))
        res.append("size of cand/ref: %i/%i[%f]"%(self.total_length, self.ref_length, float(self.total_length) / self.ref_length))
        return " ".join(res)
    
    __str__ = __repr__
    
    def bleu(self):
        if min(self.ngrams_corrects.values()) <= 0:
            return 0
        assert min(self.ngrams_total.values()) >= 0
        assert min(self.ngrams_total.values()) >= min(self.ngrams_corrects.values())
        
        log_brevity_penalty = min(0, 1.0 - float(self.ref_length) / self.total_length)
        log_average_precision = 0.25 *(
                    sum(math.log(v) for v in self.ngrams_corrects.values()) -
                    sum(math.log(v) for v in self.ngrams_total.values())
                    )
        res = math.exp(log_brevity_penalty + log_average_precision)
        return res
    
    def bleu_plus_alpha(self, alpha = 1.0):
        log_brevity_penalty = min(0, 1.0 - float(self.ref_length) / self.total_length)
        log_average_precision = 0.25 *(
                    sum(math.log(v + alpha) for v in self.ngrams_corrects.values()) -
                    sum(math.log(v + alpha) for v in self.ngrams_total.values())
                    )
        res = math.exp(log_brevity_penalty + log_average_precision)
        return res
    
    def update(self, reference, translation):
        self.ref_length += len(reference)
        self.total_length += len(translation)
        for n in xrange(1, 5):
            reference_ngrams = defaultdict(int)
            translation_ngrams = defaultdict(int)
            for start in xrange(0, len(reference) - n + 1):
                ngram = tuple(reference[start : start + n])
                reference_ngrams[ngram] += 1
            for start in xrange(0, len(translation) - n + 1):
                ngram = tuple(translation[start : start + n])
#                 print ngram
                translation_ngrams[ngram] += 1
            for ngram, translation_freq in translation_ngrams.iteritems():
                reference_freq = reference_ngrams[ngram]
                self.ngrams_total[n] += translation_freq
                if ngram in reference_ngrams:
                    if reference_freq >= translation_freq:
                        self.ngrams_corrects[n] += translation_freq
                    else:
                        self.ngrams_corrects[n] += reference_freq
                        
    def update_plus(self, diff):
        ngrams_corrects, ngrams_total, t_len, r_len = diff
        for n in xrange(1, 5):
            self.ngrams_corrects[n] += ngrams_corrects[n]
            self.ngrams_total[n] += ngrams_total[n]
        self.ref_length += r_len
        self.total_length += t_len
        
    def update_minus(self, diff):
        ngrams_corrects, ngrams_total, t_len, r_len = diff
        for n in xrange(1, 5):
            self.ngrams_corrects[n] -= ngrams_corrects[n]
            self.ngrams_total[n] -= ngrams_total[n]
            assert self.ngrams_corrects[n] >= 0
            assert self.ngrams_total[n] >= 0
        self.ref_length -= r_len
        self.total_length -= t_len
        
        assert self.total_length >= 0
        assert self.ref_length >= 0
        
    @staticmethod
    def compute_ngram_info(sentence):
        infos = defaultdict(int)
        for n in xrange(1, 5):
            for start in xrange(0, len(sentence) - n + 1):
                ngram = tuple(sentence[start : start + n])
                infos[ngram] += 1
        return infos, len(sentence)
        
    @staticmethod
    def compute_update_diff_from__infos(reference_info, translation_info):
        ngrams_corrects = {1: 0, 2: 0, 3: 0, 4: 0}
        ngrams_total = {1: 0, 2: 0, 3: 0, 4: 0}

        reference_ngrams, ref_len = reference_info
        translation_ngrams, t_len = translation_info

        for ngram, translation_freq in translation_ngrams.iteritems():
            n = len(ngram)
            reference_freq = reference_ngrams[ngram]
            ngrams_total[n] += translation_freq
            if ngram in reference_ngrams:
                if reference_freq >= translation_freq:
                    ngrams_corrects[n] += translation_freq
                else:
                    ngrams_corrects[n] += reference_freq
        return ngrams_corrects, ngrams_total, t_len, ref_len
        
    @staticmethod
    def compute_update_diff(reference, translation):
        ngrams_corrects = {1: 0, 2: 0, 3: 0, 4: 0}
        ngrams_total = {1: 0, 2: 0, 3: 0, 4: 0}

        for n in xrange(1, 5):
            reference_ngrams = defaultdict(int)
            translation_ngrams = defaultdict(int)
            for start in xrange(0, len(reference) - n + 1):
                ngram = tuple(reference[start : start + n])
                reference_ngrams[ngram] += 1
            for start in xrange(0, len(translation) - n + 1):
                ngram = tuple(translation[start : start + n])
#                 print ngram
                translation_ngrams[ngram] += 1
            for ngram, translation_freq in translation_ngrams.iteritems():
                reference_freq = reference_ngrams[ngram]
                ngrams_total[n] += translation_freq
                if ngram in reference_ngrams:
                    if reference_freq >= translation_freq:
                        ngrams_corrects[n] += translation_freq
                    else:
                        ngrams_corrects[n] += reference_freq
        return ngrams_corrects, ngrams_total, len(translation), len(reference)
    
def get_bc_from_files(ref_fn, trans_fn):
    ref_file = codecs.open(ref_fn, "r", encoding = "utf8")
    trans_file = codecs.open(trans_fn, "r", encoding = "utf8")
    
    bc = BleuComputer()
    for line_ref, line_trans in izip(ref_file, trans_file):
        r = line_ref.strip().split(" ")
        t = line_trans.strip().split(" ")
        bc.update(r, t)
    return bc    
    
    
def command_line():
    parser = argparse.ArgumentParser(description = "Compute BLEU score")
    parser.add_argument("ref")
    parser.add_argument("translations")
    args = parser.parse_args()
    
    bc = get_bc_from_files(args.ref, args.translations)
    
    print bc
        
if __name__ == "__main__":
    command_line()
    
