#!/usr/bin/env python
"""utils.py: Various utilitity functions for RNNSearch"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import os
import logging
import numpy as np
from chainer import Variable, cuda

logging.basicConfig()
log = logging.getLogger("rnns:utils")
log.setLevel(logging.INFO)

def ensure_path(path):
    try: 
        os.makedirs(path)
        log.info("Created directory %s" % path)
    except OSError:
        if not os.path.isdir(path):
            raise
        
def make_batch_src(src_data, eos_idx, padding_idx = 0, gpu = None, volatile = "off"):
    max_src_size = max(len(x) for x  in src_data)
    mb_size = len(src_data)
    src_batch = [np.empty((mb_size,), dtype = np.int32) for _ in xrange(max_src_size + 1)]
    src_mask = [np.empty((mb_size,), dtype = np.bool) for _ in xrange(max_src_size + 1)]
    
    for num_ex in xrange(mb_size):
        this_src_len = len(src_data[num_ex])
        for i in xrange(max_src_size + 1):
            if i < this_src_len:
                src_batch[i][num_ex] = src_data[num_ex][i]
                src_mask[i][num_ex] = True
            else:
                src_batch[i][num_ex] = padding_idx
                src_mask[i][num_ex] = False

    if gpu is not None:
        return ([Variable(cuda.to_gpu(x, gpu), volatile = volatile) for x in src_batch],
                [Variable(cuda.to_gpu(x, gpu), volatile = volatile) for x in src_mask])
    else:
        return [Variable(x, volatile = volatile) for x in src_batch], [Variable(x, volatile = volatile) for x in src_mask]                
                
def make_batch_src_tgt(training_data, eos_idx = 1, padding_idx = 0, gpu = None, volatile = "off"):
    training_data = sorted(training_data, key = lambda x:len(x[1]), reverse = True)
#     max_src_size = max(len(x) for x, y  in training_data)
    max_tgt_size = max(len(y) for x, y  in training_data)
    mb_size = len(training_data)
    
    src_batch, src_mask = make_batch_src(
                [x for x,y in training_data], eos_idx = eos_idx, padding_idx = padding_idx, gpu = gpu)
    
    lengths_list = []
    lowest_non_finished = mb_size -1
    for pos in xrange(max_tgt_size + 1):
        while pos > len(training_data[lowest_non_finished][1]):
            lowest_non_finished -= 1
            assert lowest_non_finished >= 0
        mb_length_at_this_pos = lowest_non_finished + 1
        assert len(lengths_list) == 0 or mb_length_at_this_pos <= lengths_list[-1]
        lengths_list.append(mb_length_at_this_pos)
        
    tgt_batch = []
    for i in xrange(max_tgt_size + 1):
        current_mb_size = lengths_list[i]
        assert current_mb_size > 0
        tgt_batch.append(np.empty((current_mb_size,), dtype = np.int32))
        for num_ex in xrange(current_mb_size):
            assert len(training_data[num_ex][1]) >= i
            if len(training_data[num_ex][1]) == i:
                tgt_batch[-1][num_ex] = eos_idx
            else:
                tgt_batch[-1][num_ex] = training_data[num_ex][1][i]
        
    if gpu is not None:
        tgt_batch_v = [Variable(cuda.to_gpu(x, gpu), volatile = volatile) for x in tgt_batch]
    else:
        tgt_batch_v = [Variable(x, volatile = volatile) for x in tgt_batch]
    
    return src_batch, tgt_batch_v, src_mask

def minibatch_looper(data, mb_size, loop = True, avoid_copy = False):
    current_start = 0
    data_exhausted = False
    while not data_exhausted:
        if avoid_copy and len(data) >= current_start + mb_size:
            training_data_sampled = data[current_start: current_start + mb_size]
            current_start += mb_size
            if current_start >= len(data):
                if loop:
                    current_start = 0
                else:
                    data_exhausted = True
                    break
        else:
            training_data_sampled = []
            while len(training_data_sampled) < mb_size:
                remaining = mb_size - len(training_data_sampled)
                training_data_sampled += data[current_start:current_start + remaining]
                current_start += remaining
                if current_start >= len(data):
                    if loop:
                        current_start = 0
                    else:
                        data_exhausted = True
                        break
        
        yield training_data_sampled
        
def batch_sort_and_split(batch, size_parts, sort_key = lambda x:len(x[1]), inplace = False):
    if not inplace:
        batch = list(batch)
    batch.sort(key = sort_key)
    nb_mb_for_sorting = len(batch) / size_parts + (1 if len(batch) % size_parts != 0 else 0)
    for num_batch in xrange(nb_mb_for_sorting):
        mb_raw = batch[num_batch * size_parts : (num_batch + 1) * size_parts]
        yield mb_raw
        
def minibatch_provider(data, eos_idx, mb_size, nb_mb_for_sorting = 1, loop = True, inplace_sorting = False, gpu = None):
    if nb_mb_for_sorting == -1:
        assert loop == False
        for mb_raw in batch_sort_and_split(data, mb_size, inplace = inplace_sorting):
            src_batch, tgt_batch, src_mask = make_batch_src_tgt(mb_raw, eos_idx = eos_idx, gpu = gpu)
            yield src_batch, tgt_batch, src_mask
    else:
        assert nb_mb_for_sorting > 0
        required_data = nb_mb_for_sorting * mb_size
        for large_batch in minibatch_looper(data, required_data, loop = loop, avoid_copy = False):
            # ok to sort in place since minibatch_looper will return copies
            for mb_raw in batch_sort_and_split(large_batch, mb_size, inplace = True):
                src_batch, tgt_batch, src_mask = make_batch_src_tgt(mb_raw, eos_idx = eos_idx, gpu = gpu)
                yield src_batch, tgt_batch, src_mask
             
def compute_bleu_with_unk_as_wrong(references, candidates, unk_id, new_unk_id_ref, new_unk_id_cand):
    import bleu_computer
    assert new_unk_id_ref != new_unk_id_cand
    bc = bleu_computer.BleuComputer()
    for ref, cand in zip(references, candidates):
        ref_mod = tuple((x if x != unk_id else new_unk_id_ref) for x in ref)
        cand_mod = tuple((int(x) if int(x) != unk_id else new_unk_id_cand) for x in cand)
        bc.update(ref_mod, cand_mod)
    return bc

def de_batch(batch, mask = None, eos_idx = None, is_variable = False, raw = False):
    res = []  
    mb_size = len(batch[0].data) if is_variable else len(batch[0])
    for sent_num in xrange(mb_size):
        res.append([])
        for src_pos in range(len(batch)):
            if mask is None or mask[src_pos].data[sent_num]:
                idx = batch[src_pos].data[sent_num] if is_variable else batch[src_pos][sent_num]
                if not raw:
                    idx = int(cuda.to_cpu(idx))
                res[sent_num].append(None)
                res[sent_num][src_pos]  = idx
                if eos_idx is not None and idx == eos_idx:
                    break
    return res
