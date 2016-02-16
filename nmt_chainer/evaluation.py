#!/usr/bin/env python
"""eval.py: Use a RNNSearch Model"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import json
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils

import models
from make_data import Indexer, build_dataset_one_side
from utils import make_batch_src, make_batch_src_tgt, minibatch_provider, compute_bleu_with_unk_as_wrong, de_batch


import collections
import logging
import codecs
import exceptions
import itertools, operator
import os.path
import gzip
# import h5py

logging.basicConfig()
log = logging.getLogger("rnns:evaluation")
log.setLevel(logging.INFO)

def translate_to_file(encdec, eos_idx, test_src_data, mb_size, tgt_voc, 
                   translations_fn, test_references = None, control_src_fn = None, src_voc = None, gpu = None):
    
    log.info("computing translations")
    translations = greedy_batch_translate(encdec, eos_idx, test_src_data, batch_size = mb_size, gpu = gpu)
    
    log.info("writing translation of set to %s"% translations_fn)
    out = codecs.open(translations_fn, "w", encoding = "utf8")
    for t in translations:
        out.write(convert_idx_to_string(t[:-1], tgt_voc) + "\n")
    
    if control_src_fn is not None:
        assert src_voc is not None
        control_out = codecs.open(control_src_fn, "w", encoding = "utf8")
        log.info("writing src of test set to %s"% control_src_fn)
        for s in test_src_data:
            control_out.write(convert_idx_to_string(s, src_voc) + "\n")
        
    if test_references is not None:
        unk_id = len(tgt_voc) - 1
        new_unk_id_ref = unk_id + 7777
        new_unk_id_cand = unk_id + 9999
        bc = compute_bleu_with_unk_as_wrong(test_references, [t[:-1] for t in translations], unk_id, new_unk_id_ref, new_unk_id_cand)
        log.info("bleu: %r"%bc)
        return bc
    else:
        return None


def compute_loss_all(encdec, test_data, eos_idx, mb_size, gpu = None):
    mb_provider_test = minibatch_provider(test_data, eos_idx, mb_size, nb_mb_for_sorting = -1, loop = False,
                                          gpu = gpu)
    test_loss = 0
    test_nb_predictions = 0
    for src_batch, tgt_batch, src_mask in mb_provider_test:
        loss, attn = encdec(src_batch, tgt_batch, src_mask, raw_loss_info = True)
        test_loss += loss[0].data
        test_nb_predictions += loss[1]
    test_loss /= test_nb_predictions
    return test_loss

def greedy_batch_translate(encdec, eos_idx, src_data, batch_size = 80, gpu = None, get_attention = False):
    nb_ex = len(src_data)
    nb_batch = nb_ex / batch_size + (1 if nb_ex % batch_size != 0 else 0)
    res = []
    attn_all = []
    for i in range(nb_batch):
        current_batch_raw_data = src_data[i * batch_size : (i + 1) * batch_size]
        src_batch, src_mask = make_batch_src(current_batch_raw_data, gpu = gpu, volatile = "on")
        sample_greedy, score, attn_list = encdec(src_batch, 50, src_mask, use_best_for_sample = True, 
                                                 keep_attn_values = get_attention)
        deb = de_batch(sample_greedy, mask = None, eos_idx = eos_idx, is_variable = False)
        res += deb
        if get_attention:
            deb_attn = de_batch(attn_list, mask = None, eos_idx = None, is_variable = True, raw = True)
            attn_all += deb_attn 
    if get_attention:
        return res, attn_all
    else:
        return res
     
   
def batch_align(encdec, eos_idx, src_tgt_data, batch_size = 80, gpu = None):
    nb_ex = len(src_tgt_data)
    nb_batch = nb_ex / batch_size + (1 if nb_ex % batch_size != 0 else 0)
    sum_loss = 0
    attn_all = []
    for i in range(nb_batch):
        current_batch_raw_data = src_tgt_data[i * batch_size : (i + 1) * batch_size]
#         print current_batch_raw_data
        src_batch, tgt_batch, src_mask, arg_sort = make_batch_src_tgt(
                    current_batch_raw_data, eos_idx = eos_idx, gpu = gpu, volatile = "on", need_arg_sort = True)
        loss, attn_list = encdec(src_batch, tgt_batch, src_mask, keep_attn_values = True)
        deb_attn = de_batch(attn_list, mask = None, eos_idx = None, is_variable = True, raw = True)
        
        assert len(arg_sort) == len(deb_attn)
        de_sorted_attn = [None] * len(deb_attn)
        for xpos in xrange(len(arg_sort)):
            original_pos = arg_sort[xpos]
            de_sorted_attn[original_pos] = deb_attn[xpos]
        
        attn_all += de_sorted_attn 
        sum_loss += float(loss.data)
    return sum_loss, attn_all
            
def convert_idx_to_string(seq, voc, eos_idx = None):
    trans = []
    for idx_tgt in seq:
        if eos_idx is not None and idx_tgt == eos_idx:
            trans.append("#EOS#")
        else:
            if idx_tgt >= len(voc):
                log.warn("found unknown idx in tgt : %i / %i"% (idx_tgt, len(voc)))
            else:
                trans.append(voc[idx_tgt])
    return " ".join(trans)

def sample_once(encdec, src_batch, tgt_batch, src_mask, src_voc, tgt_voc, eos_idx, max_nb = None):
    print "sample"
    sample_greedy, score, attn_list = encdec(src_batch, 50, src_mask, use_best_for_sample = True, need_score = True)
#                 sample, score = encdec(src_batch, 50, src_mask, use_best_for_sample = False)
    assert len(src_batch[0].data) == len(tgt_batch[0].data)
    assert len(sample_greedy[0]) == len(src_batch[0].data)
    
    debatched_src = de_batch(src_batch, mask = src_mask, eos_idx = None, is_variable= True)
    debatched_tgt = de_batch(tgt_batch, eos_idx = eos_idx, is_variable= True)
    debatched_sample = de_batch(sample_greedy, eos_idx = eos_idx)
    
    for sent_num in xrange(len(debatched_src)):
        if max_nb is not None and sent_num > max_nb:
            break
        src_idx_seq = debatched_src[sent_num]
        tgt_idx_seq = debatched_tgt[sent_num]
        sample_idx_seq = debatched_sample[sent_num]
        print "sent num", sent_num
        print "src idx:", src_idx_seq
        print "src:", convert_idx_to_string(src_idx_seq, src_voc)
        print "tgt idx:", tgt_idx_seq
        print "tgt:", convert_idx_to_string(tgt_idx_seq, tgt_voc, eos_idx = eos_idx)
        print "sample idx:", sample_idx_seq
        print "sample:", convert_idx_to_string(sample_idx_seq, tgt_voc, eos_idx = eos_idx)