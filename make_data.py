#!/usr/bin/env python
"""make_data.py: prepare data for training"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import collections
import numpy as np
import logging
import codecs
import json
import exceptions
import itertools, operator
import os.path
import gzip
# import h5py

logging.basicConfig()
log = logging.getLogger("rnns:make_data")
log.setLevel(logging.INFO)

def ensure_path(path):
    import os
    try: 
        os.makedirs(path)
        log.info("Created directory %s" % path)
    except OSError:
        if not os.path.isdir(path):
            raise

class Indexer(object):
    def __init__(self):
        self.dic = {}
        self.lst = []
        
    def convert(self, seq):
        assert len(self.dic) == len(self.lst)
        unk_idx = len(self.lst)
#         res = np.empty( (len(seq),), dtype = np.int32)
        res = [None] * len(seq)
        for pos, w in enumerate(seq):
            if w in self.dic:
                idx = self.dic.get(w, unk_idx)
                res[pos] = idx
        return res
    
    def convert_with_unk_count(self, seq):
        assert len(self.dic) == len(self.lst)
        unk_idx = len(self.lst)
#         res = np.empty( (len(seq),), dtype = np.int32)
        res = [None] * len(seq)
        unk_count = 0
        for pos, w in enumerate(seq):
            if w in self.dic:
                idx = self.dic[w]
            else:
                idx = unk_idx
                unk_count += 1
            res[pos] = idx
        return res, unk_count
    
    def __len__(self):
        assert len(self.dic) == len(self.lst)
        return len(self.lst)

def build_index(fn, voc_limit = None, max_nb_ex = None):
    f = codecs.open(fn, encoding= "utf8")
    counts = collections.defaultdict(int)
    for num_ex, line in enumerate(f):
        if max_nb_ex is not None and num_ex >= max_nb_ex:
            break
        line =line.strip().split(" ")
        for w in line:
            counts[w] += 1
    
      
    sorted_counts = sorted(counts.items(), key = operator.itemgetter(1), reverse = True)
    
    res = Indexer()
    
    for idx,(w, cnt) in enumerate(sorted_counts[:voc_limit]):
        res.dic[w] = idx
        res.lst.append(w)
        assert len(res.lst) == len(res.dic)
    
    return res
    
    
def build_dataset(src_fn, tgt_fn, src_voc_limit = None, tgt_voc_limit = None, max_nb_ex = None, dic_src = None, dic_tgt = None):
    if dic_src is None:
        log.info("building src_dic")
        dic_src = build_index(src_fn, src_voc_limit, max_nb_ex)
        
    if dic_tgt is None:
        log.info("building tgt_dic")
        dic_tgt = build_index(tgt_fn, tgt_voc_limit, max_nb_ex)
    
    
    log.info("start indexing")
    
    src = codecs.open(src_fn, encoding= "utf8")
    tgt = codecs.open(tgt_fn, encoding= "utf8")
    
    res = []
    
    num_ex = 0
    total_token_src = 0
    total_token_tgt = 0
    total_count_unk_src = 0
    total_count_unk_tgt = 0
    while 1:
        if max_nb_ex is not None and num_ex >= max_nb_ex:
            break
        
        line_src = src.readline()
        line_tgt = tgt.readline()
        
        if len(line_src) == 0:
            assert len(line_tgt) == 0
            break
        
        line_src = line_src.strip().split(" ")
        line_tgt = line_tgt.strip().split(" ")
        
        seq_src, unk_cnt_src= dic_src.convert_with_unk_count(line_src)
        seq_tgt, unk_cnt_tgt = dic_tgt.convert_with_unk_count(line_tgt)

        total_count_unk_src += unk_cnt_src
        total_count_unk_tgt += unk_cnt_tgt
        
        total_token_src += len(seq_src)
        total_token_tgt += len(seq_tgt)

        res.append((seq_src, seq_tgt))
        num_ex += 1
        
    return res, dic_src, dic_tgt, total_count_unk_src, total_count_unk_tgt, total_token_src, total_token_tgt, num_ex

def cmdline():
    import sys
    import argparse
    parser = argparse.ArgumentParser(description= "Prepare training data.", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src_fn", help = "source language text file for training data")
    parser.add_argument("tgt_fn", help = "target language text file for training data")
    parser.add_argument("save_prefix", help = "created files will be saved with this prefix")
    parser.add_argument("--src_voc_size", type = int, default = 32000, 
                        help = "limit source vocabulary size to the n most frequent words")
    parser.add_argument("--tgt_voc_size", type = int, default = 32000,
                        help = "limit target vocabulary size to the n most frequent words")
#     parser.add_argument("--add_to_valid_set_every", type = int)
#     parser.add_argument("--shuffle", default = False, action = "store_true")
#     parser.add_argument("--enable_fast_shuffle", default = False, action = "store_true")
    parser.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
    
    parser.add_argument("--test_src", help = "specify a source test set")
    parser.add_argument("--test_tgt", help = "specify a target test set")
    
    parser.add_argument("--dev_src", help = "specify a source dev set")
    parser.add_argument("--dev_tgt", help = "specify a target dev set")
    args = parser.parse_args()
    
    if not ((args.test_src is None) == (args.test_tgt is None)):
        print >>sys.stderr, "Command Line Error: either specify both --test_src and --test_tgt or neither"
        sys.exit(1)
    
    if not ((args.dev_src is None) == (args.dev_tgt is None)):
        print >>sys.stderr, "Command Line Error: either specify both --test_src and --test_tgt or neither"
        sys.exit(1)
    
    save_prefix_dir, save_prefix_fn = os.path.split(args.save_prefix)
    ensure_path(save_prefix_dir)
    
    config_fn = args.save_prefix + ".data.config"
    voc_fn = args.save_prefix + ".voc"
    data_fn = args.save_prefix + ".data.json.gz"
#     valid_data_fn = args.save_prefix + "." + args.model + ".valid.data.npz"
    
    already_existing_files = []
    for filename in [config_fn, voc_fn, data_fn]:#, valid_data_fn]:
        if os.path.exists(filename):
            already_existing_files.append(filename)
    if len(already_existing_files) > 0:
        print "Warning: existing files are going to be replaced: ",  already_existing_files
        raw_input("Press Enter to Continue")
        
        
    def load_data(src_fn, tgt_fn, max_nb_ex = None, dic_src = None, dic_tgt = None):
        
        training_data, dic_src, dic_tgt, total_count_unk_src, total_count_unk_tgt, total_token_src, total_token_tgt, nb_ex = build_dataset(
                                            src_fn, tgt_fn, src_voc_limit = args.src_voc_size, 
                                            tgt_voc_limit = args.tgt_voc_size, max_nb_ex = max_nb_ex, 
                                            dic_src = dic_src, dic_tgt = dic_tgt)
        
        log.info("%i sentences loaded"%nb_ex)
        
        log.info("size dic src: %i"%len(dic_src))
        log.info("size dic tgt: %i"%len(dic_tgt))
        
        log.info("#tokens src: %i   of which %i (%f%%) are unknown"%(total_token_src, total_count_unk_src, 
                                                                     float(total_count_unk_src * 100) / total_token_src))
        
        log.info("#tokens tgt: %i   of which %i (%f%%) are unknown"%(total_token_tgt, total_count_unk_tgt, 
                                                                 float(total_count_unk_tgt * 100) / total_token_tgt))
        
        return training_data, dic_src, dic_tgt
    
    log.info("loading training data from %s and %s"%(args.src_fn, args.tgt_fn))
    training_data, dic_src, dic_tgt = load_data(args.src_fn, args.tgt_fn, max_nb_ex = args.max_nb_ex)
    
    test_data = None
    if args.test_src is not None:
        log.info("loading test data from %s and %s"%(args.test_src, args.test_tgt))
        test_data, test_dic_src, test_dic_tgt = load_data(args.test_src, args.test_tgt, dic_src = dic_src, dic_tgt = dic_tgt)
        
        assert test_dic_src is dic_src
        assert test_dic_tgt is dic_tgt

    dev_data = None
    if args.dev_src is not None:
        log.info("loading dev data from %s and %s"%(args.dev_src, args.dev_tgt))
        dev_data, dev_dic_src, dev_dic_tgt = load_data(args.dev_src, args.dev_tgt, dic_src = dic_src, dic_tgt = dic_tgt)
        
        assert dev_dic_src is dic_src
        assert dev_dic_tgt is dic_tgt

#     if args.shuffle:
#         log.info("shuffling data")
#         if args.enable_fast_shuffle:
#             shuffle_in_unison_faster(data_input, data_target)
#         else:
#             data_input, data_target = shuffle_in_unison(data_input, data_target)
    log.info("saving config to %s"%config_fn)
    json.dump(args.__dict__, open(config_fn, "w"), indent=2, separators=(',', ': '))

    log.info("saving voc to %s"%voc_fn)
    json.dump([dic_src.lst, dic_tgt.lst], open(voc_fn, "w"))
    
    log.info("saving train_data to %s"%data_fn)
    data_all = {"train": training_data}
    if test_data is not None:
        data_all["test"] = test_data
    if dev_data is not None:
        data_all["dev"] = dev_data
    json.dump(data_all, gzip.open(data_fn, "wb"), indent=2, separators=(',', ': '))
#     fh5 = h5py.File(args.save_data_to_hdf5, 'w')
#     train_grp = fh5.create_group("train")
#     train_grp.attrs["size"] = len(training_data)
#     for i in range(len(training_data)):
#         train_grp.create_dataset("s%i"%i, data = training_data[i][0], compression="gzip")
#         train_grp.create_dataset("t%i"%i, data = training_data[i][1], compression="gzip")
    
#     if args.add_to_valid_set_every:
#         log.info("saving valid_data to %s"%valid_data_fn)
#         np.savez_compressed(open(valid_data_fn, "wb"), data_input = data_input_valid, data_target = data_target_valid)
        
if __name__ == '__main__':
    cmdline()