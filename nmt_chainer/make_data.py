#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_data.py: prepare data for training"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import collections
import logging
import codecs
import json
import operator
import os.path
import gzip

from utils import ensure_path
# import h5py

logging.basicConfig()
log = logging.getLogger("rnns:make_data")
log.setLevel(logging.INFO)

# class Indexer(object):
#     def __init__(self, unk_tag = "#UNK#"):
#         self.dic = {}
#         self.lst = []
#         self.finalized = False
#
#     def add_word(self, w, should_be_new = False):
#         assert not self.finalized
#         assert w is not None
#         if w not in self.dic:
#             new_idx = len(self.lst)
#             self.dic[w] = new_idx
#             self.lst.append(w)
#             assert len(self.lst) == len(self.dic)
#         else:
#             assert not should_be_new
#
#     def assign_index_to_voc(self, voc_iter, all_should_be_new = False):
#         assert not self.finalized
#         for w in voc_iter:
#             self.add_word(w, should_be_new = all_should_be_new)
#
#     def finalize(self):
#         assert not self.finalized
#         self.dic[None] = len(self.lst)
#         self.lst.append(None)
#         self.finalized = True
#
#     def get_unk_idx(self):
#         assert self.finalized
#         return self.dic[None]
#
#     def convert(self, seq):
#         assert self.finalized
#         assert len(self.dic) == len(self.lst)
#         unk_idx = self.get_unk_idx()
# #         res = np.empty( (len(seq),), dtype = np.int32)
#         res = [None] * len(seq)
#         for pos, w in enumerate(seq):
#             assert w is not None
#             idx = self.dic.get(w, unk_idx)
#             res[pos] = idx
#         return res
#
#     def deconvert(self, seq, unk_tag = "#UNK#", no_oov = True, eos_idx = None):
#         assert self.finalized
#         assert eos_idx is None or eos_idx >= len(self.lst)
#         res = []
#         for num, idx in enumerate(seq):
#             if idx >= len(self.lst):
#                 if eos_idx is not None and eos_idx == idx:
#                     w = "#EOS#"
#                 elif no_oov:
#                     raise KeyError()
#                 else:
#                     log.warn("unknown idx: %i / %i"%(idx, len(self.lst)))
#                     continue
#             else:
#                 w = self.lst[idx]
#             if w is None:
#                 if callable(unk_tag):
#                     w = unk_tag(num)
#                 else:
#                     w = unk_tag
#             res.append(w)
#         return res
#
#     def __len__(self):
#         assert self.finalized
#         assert len(self.dic) == len(self.lst)
#         return len(self.lst)
#
#     def to_serializable(self):
#         return {"type": "simple_indexer", "rev": 1,  "voc_lst" : self.lst}
#
#     @staticmethod
#     def make_from_serializable(datas):
#         assert datas["type"] == "simple_indexer"
#         assert datas["rev"] == 1
#         voc_lst = datas["voc_lst"]
#         res = Indexer()
#         res.lst = list(voc_lst)
#         for idx, w in enumerate(voc_lst):
#             res.dic[w] = idx
#         res.finalized = True
#         return res


class Indexer(object):

    def __init__(self, unk_tag="#UNK#"):
        self.dic = {}
        self.lst = []
        self.unk_label_dictionary = None
        self.finalized = False

    def add_word(self, w, should_be_new=False, should_not_be_int=True):
        assert not self.finalized
        assert not (should_not_be_int and isinstance(w, int))
        if w not in self.dic:
            new_idx = len(self.lst)
            self.dic[w] = new_idx
            self.lst.append(w)
            assert len(self.lst) == len(self.dic)
        else:
            assert not should_be_new

    def assign_index_to_voc(self, voc_iter, all_should_be_new=False):
        assert not self.finalized
        for w in voc_iter:
            self.add_word(w, should_be_new=all_should_be_new)

    def finalize(self):
        assert not self.finalized
        assert len(self.dic) == len(self.lst)
        self.add_word(0, should_be_new=False, should_not_be_int=False)
        self.finalized = True

    def get_one_unk_idx(self, w):
        assert self.finalized
        assert w not in self.dic
        if self.unk_label_dictionary is not None:
            return self.dic[self.unk_label_dictionary.get(w, 1)]
        else:
            return self.dic[0]

    def is_unk_idx(self, idx):
        assert self.finalized
        assert idx < len(self.lst)
        return isinstance(self.lst[idx], int)

    def add_unk_label_dictionary(self, unk_dic):
        assert not self.finalized
        self.unk_label_dictionary = unk_dic

    def convert(self, seq):
        assert self.finalized
        assert len(self.dic) == len(self.lst)
        res = [None] * len(seq)
        for pos, w in enumerate(seq):
            assert not isinstance(w, int)
            if w in self.dic:
                idx = self.dic[w]
            else:
                idx = self.get_one_unk_idx(w)
            res[pos] = idx
        return res

    def convert_and_update_unk_tags(self, seq, give_unk_label):
        assert not self.finalized
        assert len(self.dic) == len(self.lst)
        res = [None] * len(seq)
        for pos, w in enumerate(seq):
            assert not isinstance(w, int)
            if w in self.dic:
                idx = self.dic[w]
            else:
                aligned_pos = give_unk_label(pos, w)
                if aligned_pos not in self.dic:
                    self.add_word(aligned_pos, should_be_new=True,
                                  should_not_be_int=False)
                idx = self.dic[aligned_pos]
            res[pos] = idx
        return res

    def deconvert(self, seq, unk_tag="#UNK#", no_oov=True, eos_idx=None):
        assert self.finalized
        assert eos_idx is None or eos_idx >= len(self.lst)
        res = []
        for num, idx in enumerate(seq):
            if idx >= len(self.lst):
                if eos_idx is not None and eos_idx == idx:
                    w = "#EOS#"
                elif no_oov:
                    raise KeyError()
                else:
                    log.warn("unknown idx: %i / %i" % (idx, len(self.lst)))
                    continue
            else:
                w = self.lst[idx]

            if isinstance(w, int):
                if callable(unk_tag):
                    w = unk_tag(num, w)
                else:
                    w = unk_tag

            res.append(w)
        return res

    def __len__(self):
        assert self.finalized
        assert len(self.dic) == len(self.lst)
        return len(self.lst)

    def to_serializable(self):
        return {"type": "simple_indexer", "rev": 1,  "voc_lst": self.lst, "unk_label_dic": self.unk_label_dictionary}

    @staticmethod
    def make_from_serializable(datas):
        if isinstance(datas, list):
            # legacy mode
            log.info("loading legacy voc")
            voc_lst = datas
            assert 0 not in voc_lst
            res = Indexer()
            res.lst = list(voc_lst)  # add UNK
            for idx, w in enumerate(res.lst):
                assert isinstance(w, basestring)
                res.dic[w] = idx
            assert len(res.dic) == len(res.lst)
            res.finalize()
            return res
        else:
            assert isinstance(datas, dict)
            assert datas["type"] == "simple_indexer"
            assert datas["rev"] == 1
            voc_lst = datas["voc_lst"]
            res = Indexer()
            res.lst = list(voc_lst)
            res.unk_label_dictionary = datas["unk_label_dic"]
            for idx, w in enumerate(voc_lst):
                res.dic[w] = idx
            res.finalized = True
            return res

MakeDataInfosOneSide = collections.namedtuple(
    "MakeDataInfosOneSide", ["total_count_unk", "total_token", "nb_ex"])

MakeDataInfos = collections.namedtuple("MakeDataInfos", ["total_count_unk_src", "total_count_unk_tgt", "total_token_src",
                                                         "total_token_tgt", "nb_ex"])


def segment(line, type="word"):
    if type == "word":
        return line.split(" ")
    elif type == "word2char":
        return tuple("".join(line.split(" ")))
    elif type == "char":
        return tuple(line)
    else:
        raise NotImplemented


def build_index_from_string(str, voc_limit=None, max_nb_ex=None, segmentation_type="word"):
    counts = collections.defaultdict(int)
    line = segment(str.strip(), type=segmentation_type)  # .split(" ")

    for w in line:
        counts[w] += 1

    sorted_counts = sorted(
        counts.items(), key=operator.itemgetter(1), reverse=True)

    res = Indexer()

    for w, _ in sorted_counts[:voc_limit]:
        res.add_word(w, should_be_new=True)
    res.finalize()

    return res

def build_index(fn, voc_limit=None, max_nb_ex=None, segmentation_type="word"):
    f = codecs.open(fn, encoding="utf8")
    counts = collections.defaultdict(int)
    for num_ex, line in enumerate(f):
        if max_nb_ex is not None and num_ex >= max_nb_ex:
            break
        line = segment(line.strip(), type=segmentation_type)  # .split(" ")
        for w in line:
            counts[w] += 1

    sorted_counts = sorted(
        counts.items(), key=operator.itemgetter(1), reverse=True)

    res = Indexer()

    for w, _ in sorted_counts[:voc_limit]:
        res.add_word(w, should_be_new=True)
    res.finalize()

    return res


def build_dataset_one_side_from_string(src_str, src_voc_limit=None, max_nb_ex=None, dic_src=None,
                           segmentation_type = "word"):
    if dic_src is None:
        log.info("building src_dic")
        dic_src = build_index_from_string(src_str, src_voc_limit, max_nb_ex,
                              segmentation_type = segmentation_type)

    log.info("start indexing")

    res = []

    num_ex = 0
    total_token_src = 0
    total_count_unk_src = 0

    line_src = src_str

    if len(line_src) > 0:
        line_src = line_src.strip().split(" ")

        seq_src = dic_src.convert(line_src)
        unk_cnt_src = sum(dic_src.is_unk_idx(w) for w in seq_src)

        total_count_unk_src += unk_cnt_src

        total_token_src += len(seq_src)

        res.append(seq_src)
        num_ex += 1

    return res, dic_src, MakeDataInfosOneSide(total_count_unk_src,
                                              total_token_src,
                                              num_ex
                                              )

def build_dataset_one_side(src_fn, src_voc_limit=None, max_nb_ex=None, dic_src=None,
                           segmentation_type = "word"):
    if dic_src is None:
        log.info("building src_dic")
        dic_src = build_index(src_fn, src_voc_limit, max_nb_ex,
                              segmentation_type = segmentation_type)

    log.info("start indexing")

    src = codecs.open(src_fn, encoding="utf8")

    res = []

    num_ex = 0
    total_token_src = 0
    total_count_unk_src = 0
    while 1:
        if max_nb_ex is not None and num_ex >= max_nb_ex:
            break

        line_src = src.readline()

        if len(line_src) == 0:
            break

        line_src = line_src.strip().split(" ")

        seq_src = dic_src.convert(line_src)
        unk_cnt_src = sum(dic_src.is_unk_idx(w) for w in seq_src)

        total_count_unk_src += unk_cnt_src

        total_token_src += len(seq_src)

        res.append(seq_src)
        num_ex += 1

    return res, dic_src, MakeDataInfosOneSide(total_count_unk_src,
                                              total_token_src,
                                              num_ex
                                              )

def build_dataset_for_nbest_list_scoring(dic_src, nbest_list):
    res = []
    num_ex = 0
    total_token_src = 0
    total_count_unk_src = 0
    for sublist in nbest_list:
        res.append([])
        for sentence in sublist:
            seq_src = dic_src.convert(sentence)
            unk_cnt_src = sum(dic_src.is_unk_idx(w) for w in seq_src)
    
            total_count_unk_src += unk_cnt_src
    
            total_token_src += len(seq_src)
    
            res[-1].append(seq_src)
            num_ex += 1  
    return res, MakeDataInfosOneSide(total_count_unk_src,
                                              total_token_src,
                                              num_ex
                                              )
# 
# def build_dataset_one_side_from_list_of_list(src_fn, src_voc_limit=None, max_nb_ex=None, dic_src=None,
#                            segmentation_type = "word"):
#     if dic_src is None:
#         log.info("building src_dic")
#         dic_src = build_index(src_fn, src_voc_limit, max_nb_ex,
#                               segmentation_type = segmentation_type)
# 
#     log.info("start indexing")
# 
#     src = codecs.open(src_fn, encoding="utf8")
# 
#     res = []
# 
#     num_ex = 0
#     total_token_src = 0
#     total_count_unk_src = 0
#     while 1:
#         if max_nb_ex is not None and num_ex >= max_nb_ex:
#             break
# 
#         line_src = src.readline()
# 
#         if len(line_src) == 0:
#             break
# 
#         line_src = line_src.strip().split(" ")
# 
#         seq_src = dic_src.convert(line_src)
#         unk_cnt_src = sum(dic_src.is_unk_idx(w) for w in seq_src)
# 
#         total_count_unk_src += unk_cnt_src
# 
#         total_token_src += len(seq_src)
# 
#         res.append(seq_src)
#         num_ex += 1
# 
#     return res, dic_src, MakeDataInfosOneSide(total_count_unk_src,
#                                               total_token_src,
#                                               num_ex
#                                               )

def build_dataset(src_fn, tgt_fn,
                  src_voc_limit=None, tgt_voc_limit=None, max_nb_ex=None, dic_src=None, dic_tgt=None,
                  tgt_segmentation_type="word", src_segmentation_type="word"):
    if dic_src is None:
        log.info("building src_dic")
        dic_src = build_index(src_fn, src_voc_limit, max_nb_ex,
                              segmentation_type=src_segmentation_type)

    if dic_tgt is None:
        log.info("building tgt_dic")
        dic_tgt = build_index(tgt_fn, tgt_voc_limit, max_nb_ex,
                              segmentation_type=tgt_segmentation_type)

    log.info("start indexing")

    src = codecs.open(src_fn, encoding="utf8")
    tgt = codecs.open(tgt_fn, encoding="utf8")

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

#         line_src = line_src.strip().split(" ")
        
        line_src = segment(
            line_src.strip(), type=src_segmentation_type)
        
        line_tgt = segment(
            line_tgt.strip(), type=tgt_segmentation_type)  # .split(" ")

        seq_src = dic_src.convert(line_src)
        unk_cnt_src = sum(dic_src.is_unk_idx(w) for w in seq_src)

        seq_tgt = dic_tgt.convert(line_tgt)
        unk_cnt_tgt = sum(dic_tgt.is_unk_idx(w) for w in seq_tgt)

        total_count_unk_src += unk_cnt_src
        total_count_unk_tgt += unk_cnt_tgt

        total_token_src += len(seq_src)
        total_token_tgt += len(seq_tgt)

        res.append((seq_src, seq_tgt))
        num_ex += 1

    return res, dic_src, dic_tgt, MakeDataInfos(total_count_unk_src,
                                                total_count_unk_tgt,
                                                total_token_src,
                                                total_token_tgt,
                                                num_ex
                                                )



def build_dataset_with_align_info(src_fn, tgt_fn, align_fn, 
                                  src_voc_limit = None, tgt_voc_limit = None, max_nb_ex = None, 
                                  dic_src = None, dic_tgt = None,
                                  invert_alignment_links = False, add_to_valid_every = 1000,
                                  mode = "unk_align"):
    from aligned_parse_reader import load_aligned_corpus
    corpus = load_aligned_corpus(
        src_fn, tgt_fn, align_fn, skip_empty_align=True, invert_alignment_links=invert_alignment_links)

    # first find the most frequent words
    log.info("computing restricted vocabulary")
    counts_src = collections.defaultdict(int)
    counts_tgt = collections.defaultdict(int)
    for num_ex, (sentence_src, sentence_tgt, alignment) in enumerate(corpus):
        if num_ex % add_to_valid_every == 0:
            continue

        if max_nb_ex is not None and num_ex >= max_nb_ex:
            break
        for pos, w in enumerate(sentence_src):
            counts_src[w] += 1
        for w in sentence_tgt:
            counts_tgt[w] += 1
    sorted_counts_src = sorted(
        counts_src.items(), key=operator.itemgetter(1), reverse=True)
    sorted_counts_tgt = sorted(
        counts_tgt.items(), key=operator.itemgetter(1), reverse=True)

    dic_src = Indexer()
    dic_src.assign_index_to_voc((w for w, _ in sorted_counts_src[
                                :src_voc_limit]), all_should_be_new=True)

    dic_tgt = Indexer()

    dic_tgt.assign_index_to_voc((w for w, _ in sorted_counts_tgt[:tgt_voc_limit]), all_should_be_new = True)
    
    if mode == "all_align":
        dic_src.finalize()
        dic_tgt.finalize()
    
    log.info("computed restricted vocabulary")

#     src_unk_tag = "#S_UNK_%i#!"
#     tgt_unk_tag = "#T_UNK_%i#!"

    corpus = load_aligned_corpus(
        src_fn, tgt_fn, align_fn, skip_empty_align=True, invert_alignment_links=invert_alignment_links)

    total_token_src = 0
    total_token_tgt = 0
    total_count_unk_src = [0]
    total_count_unk_tgt = [0]
    res = []
    valid = []
    fertility_counts = collections.defaultdict(
        lambda: collections.defaultdict(int))
    for num_ex, (sentence_src, sentence_tgt, alignment) in enumerate(corpus):
        if max_nb_ex is not None and num_ex >= max_nb_ex:
            break
        local_fertilities = {}
        local_alignments = {}
        for left, right in alignment:
            for src_pos in left:
                local_fertilities[src_pos] = len(right)
            for tgt_pos in right:
                assert tgt_pos not in local_alignments
                local_alignments[tgt_pos] = left[0]
    
        if mode == "unk_align":
            def give_fertility(pos, w):
                fertility = local_fertilities.get(pos, 0)
                fertility_counts[w][fertility] += 1
                total_count_unk_src[0] += 1
                return fertility
            
            def give_alignment(pos, w):
                total_count_unk_tgt[0] += 1
                aligned_pos = local_alignments.get(pos, -1)
                return aligned_pos
            
            seq_idx_src = dic_src.convert_and_update_unk_tags(sentence_src, give_unk_label = give_fertility)
            total_token_src += len(seq_idx_src)
            
            seq_idx_tgt = dic_tgt.convert_and_update_unk_tags(sentence_tgt, give_unk_label = give_alignment)
            total_token_tgt += len(seq_idx_tgt)
            
        elif mode == "all_align":
            def give_fertility(pos, w):
                fertility = local_fertilities.get(pos, 0)
                fertility_counts[w][fertility] += 1
                return fertility
            
            def give_alignment(pos):
                aligned_pos = local_alignments.get(pos, -1)
                return aligned_pos
            
            seq_src = dic_src.convert(sentence_src)
            unk_cnt_src = sum(dic_src.is_unk_idx(w) for w in seq_src)
            
            seq_tgt = dic_tgt.convert(sentence_tgt)
            unk_cnt_tgt = sum(dic_tgt.is_unk_idx(w) for w in seq_tgt)
    
            total_count_unk_src[0] += unk_cnt_src
            total_count_unk_tgt[0] += unk_cnt_tgt
            
            total_token_src += len(seq_src)
            total_token_tgt += len(seq_tgt)
            
            seq_idx_src = []
            for pos in range(len(seq_src)):
                w = seq_src[pos]
                seq_idx_src.append( (w, give_fertility(pos, w)) )
                
            seq_idx_tgt = []
            for pos in range(len(seq_tgt)):
                w = seq_tgt[pos]
                seq_idx_tgt.append( (w, give_alignment(pos)) )
            
        else:
            assert False
        
        if num_ex % add_to_valid_every == 0:
            valid.append((seq_idx_src, seq_idx_tgt))
        else:
            res.append((seq_idx_src, seq_idx_tgt))
    
    if mode == "unk_align":
        src_unk_voc_to_fertility = {}
        for w in fertility_counts:
            most_common_fertility = sorted(fertility_counts[w].items(), key = operator.itemgetter(1), reverse = True)[0][0]
            src_unk_voc_to_fertility[w] = most_common_fertility
            
        dic_src.add_unk_label_dictionary(src_unk_voc_to_fertility)
        
        dic_tgt.finalize()
        dic_src.finalize()
    
    return res, valid, dic_src, dic_tgt, MakeDataInfos(total_count_unk_src[0], 
                                                total_count_unk_tgt[0], 
                                                total_token_src, 
                                                total_token_tgt, 
                                                num_ex
                                                )
    return res, valid, dic_src, dic_tgt, MakeDataInfos(total_count_unk_src[0],
                                                       total_count_unk_tgt[0],
                                                       total_token_src,
                                                       total_token_tgt,
                                                       num_ex
                                                       )


def define_parser(parser):
    parser.add_argument(
        "src_fn", help="source language text file for training data")
    parser.add_argument(
        "tgt_fn", help="target language text file for training data")
    parser.add_argument(
        "save_prefix", help="created files will be saved with this prefix")
    parser.add_argument("--align_fn", help="align file for training data")
    parser.add_argument("--src_voc_size", type=int, default=32000,
                        help="limit source vocabulary size to the n most frequent words")
    parser.add_argument("--tgt_voc_size", type=int, default=32000,
                        help="limit target vocabulary size to the n most frequent words")
#     parser.add_argument("--add_to_valid_set_every", type = int)
#     parser.add_argument("--shuffle", default = False, action = "store_true")
#     parser.add_argument("--enable_fast_shuffle", default = False, action = "store_true")

    parser.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
    
    parser.add_argument("--test_src", help = "specify a source test set")
    parser.add_argument("--test_tgt", help = "specify a target test set")
    
    parser.add_argument("--dev_src", help = "specify a source dev set")
    parser.add_argument("--dev_tgt", help = "specify a target dev set")
    parser.add_argument("--mode_align", choices = ["unk_align", "all_align"])
    parser.add_argument("--use_voc", help = "specify an exisiting vocabulary file")
    
    parser.add_argument("--tgt_segmentation_type", choices = ["word", "word2char", "char"], default = "word")
    parser.add_argument("--src_segmentation_type", choices = ["word", "word2char", "char"], default = "word")
    
    
    
def cmdline(arguments=None):
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Prepare training data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    define_parser(parser)
    
    args = parser.parse_args(args = arguments)
    
    do_make_data(args)
    
    
def do_make_data(args):
    
    
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
    for filename in [config_fn, voc_fn, data_fn]:  # , valid_data_fn]:
        if os.path.exists(filename):
            already_existing_files.append(filename)
    if len(already_existing_files) > 0:
        print "Warning: existing files are going to be replaced: ",  already_existing_files
        raw_input("Press Enter to Continue")

    def load_data(src_fn, tgt_fn, max_nb_ex=None, dic_src=None, dic_tgt=None, align_fn=None):

        if align_fn is not None:
            log.info("making training data with alignment")
            training_data, valid_data, dic_src, dic_tgt, make_data_infos = build_dataset_with_align_info(
                                            src_fn, tgt_fn, align_fn, src_voc_limit = args.src_voc_size, 
                                            tgt_voc_limit = args.tgt_voc_size, max_nb_ex = max_nb_ex, 
                                            dic_src = dic_src, dic_tgt = dic_tgt, mode = args.mode_align)
        else:
            training_data, dic_src, dic_tgt, make_data_infos = build_dataset(
                src_fn, tgt_fn, src_voc_limit=args.src_voc_size,
                tgt_voc_limit=args.tgt_voc_size, max_nb_ex=max_nb_ex,
                dic_src=dic_src, dic_tgt=dic_tgt,
                tgt_segmentation_type=args.tgt_segmentation_type,
                src_segmentation_type=args.src_segmentation_type)
            valid_data = None

        log.info("%i sentences loaded" % make_data_infos.nb_ex)
        if valid_data is not None:
            log.info("valid set size:%i sentences" % len(valid_data))
        log.info("size dic src: %i" % len(dic_src))
        log.info("size dic tgt: %i" % len(dic_tgt))

        log.info("#tokens src: %i   of which %i (%f%%) are unknown" % (make_data_infos.total_token_src,
                                                                       make_data_infos.total_count_unk_src,
                                                                       float(make_data_infos.total_count_unk_src * 100) /
                                                                       make_data_infos.total_token_src))

        log.info("#tokens tgt: %i   of which %i (%f%%) are unknown" % (make_data_infos.total_token_tgt,
                                                                       make_data_infos.total_count_unk_tgt,
                                                                       float(make_data_infos.total_count_unk_tgt * 100) /
                                                                       make_data_infos.total_token_tgt))

        return training_data, valid_data, dic_src, dic_tgt

    dic_src = None
    dic_tgt = None
    if args.use_voc is not None:
        log.info("loading voc from %s" % args.use_voc)
        src_voc, tgt_voc = json.load(open(args.use_voc))
        dic_src = Indexer.make_from_serializable(src_voc)
        dic_tgt = Indexer.make_from_serializable(tgt_voc)

    log.info("loading training data from %s and %s" %
             (args.src_fn, args.tgt_fn))
    training_data, valid_data, dic_src, dic_tgt = load_data(args.src_fn, args.tgt_fn, max_nb_ex=args.max_nb_ex,
                                                            dic_src=dic_src, dic_tgt=dic_tgt, align_fn=args.align_fn)

    test_data = None
    if args.test_src is not None:
        log.info("loading test data from %s and %s" %
                 (args.test_src, args.test_tgt))
        test_data, _, test_dic_src, test_dic_tgt = load_data(
            args.test_src, args.test_tgt, dic_src=dic_src, dic_tgt=dic_tgt)

        assert test_dic_src is dic_src
        assert test_dic_tgt is dic_tgt

    dev_data = None
    if args.dev_src is not None:
        log.info("loading dev data from %s and %s" %
                 (args.dev_src, args.dev_tgt))
        dev_data, _, dev_dic_src, dev_dic_tgt = load_data(
            args.dev_src, args.dev_tgt, dic_src=dic_src, dic_tgt=dic_tgt)

        assert dev_dic_src is dic_src
        assert dev_dic_tgt is dic_tgt

#     if args.shuffle:
#         log.info("shuffling data")
#         if args.enable_fast_shuffle:
#             shuffle_in_unison_faster(data_input, data_target)
#         else:
#             data_input, data_target = shuffle_in_unison(data_input, data_target)
    log.info("saving config to %s" % config_fn)
    json.dump(args.__dict__, open(config_fn, "w"),
              indent=2, separators=(',', ': '))

    log.info("saving voc to %s" % voc_fn)
    json.dump([dic_src.to_serializable(), dic_tgt.to_serializable()],
              open(voc_fn, "w"), indent=2, separators=(',', ': '))

    log.info("saving train_data to %s" % data_fn)
    data_all = {"train": training_data}
    if test_data is not None:
        data_all["test"] = test_data
    if dev_data is not None:
        data_all["dev"] = dev_data
    if valid_data is not None:
        data_all["valid"] = valid_data

    json.dump(data_all, gzip.open(data_fn, "wb"),
              indent=2, separators=(',', ': '))
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
