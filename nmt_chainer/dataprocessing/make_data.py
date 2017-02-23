#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_data.py: prepare data for training"""
import nmt_chainer
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

from nmt_chainer.utilities.utils import ensure_path
import nmt_chainer.dataprocessing.processors as processors

logging.basicConfig()
log = logging.getLogger("rnns:make_data")
log.setLevel(logging.INFO)


# 
# MakeDataInfosOneSide = collections.namedtuple(
#     "MakeDataInfosOneSide", ["total_count_unk", "total_token", "nb_ex"])
#   
# MakeDataInfos = collections.namedtuple("MakeDataInfos", ["total_count_unk_src", "total_count_unk_tgt", "total_token_src",
#                                                          "total_token_tgt", "nb_ex"])
#  
#  
# def build_index_from_string(str, voc_limit=None, max_nb_ex=None, segmentation_type="word"):
#     counts = collections.defaultdict(int)
#     line = segment(str.strip(), type=segmentation_type)  # .split(" ")
#  
#     for w in line:
#         counts[w] += 1
#  
#     sorted_counts = sorted(
#         counts.items(), key=operator.itemgetter(1), reverse=True)
#  
#     res = Indexer()
#  
#     for w, _ in sorted_counts[:voc_limit]:
#         res.add_word(w, should_be_new=True)
#     res.finalize()
#  
#     return res
#  
# def build_index(fn, voc_limit=None, max_nb_ex=None, segmentation_type="word"):
#     f = codecs.open(fn, encoding="utf8")
#     counts = collections.defaultdict(int)
#     for num_ex, line in enumerate(f):
#         if max_nb_ex is not None and num_ex >= max_nb_ex:
#             break
#         line = segment(line.strip(), type=segmentation_type)  # .split(" ")
#         for w in line:
#             counts[w] += 1
#  
#     sorted_counts = sorted(
#         counts.items(), key=operator.itemgetter(1), reverse=True)
#  
#     res = Indexer()
#  
#     for w, _ in sorted_counts[:voc_limit]:
#         res.add_word(w, should_be_new=True)
#     res.finalize()
#  
#     return res
#  
#  
#  
# def build_dataset_one_side_from_string(src_str, src_voc_limit=None, max_nb_ex=None, dic_src=None,
#                            segmentation_type = "word"):
#     if dic_src is None:
#         log.info("building src_dic")
#         dic_src = build_index_from_string(src_str, src_voc_limit, max_nb_ex,
#                               segmentation_type = segmentation_type)
#  
#     log.info("start indexing")
#  
#     res = []
#  
#     num_ex = 0
#     total_token_src = 0
#     total_count_unk_src = 0
#  
#     line_src = src_str
#  
#     if len(line_src) > 0:
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
#  

# def build_dataset_one_side(src_fn, src_voc_limit=None, max_nb_ex=None, dic_src=None,
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
  
# def build_dataset_for_nbest_list_scoring(dic_src, nbest_list):
#     res = []
#     num_ex = 0
#     total_token_src = 0
#     total_count_unk_src = 0
#     for sublist in nbest_list:
#         res.append([])
#         for sentence in sublist:
#             seq_src = dic_src.convert(sentence)
#             unk_cnt_src = sum(dic_src.is_unk_idx(w) for w in seq_src)
#      
#             total_count_unk_src += unk_cnt_src
#      
#             total_token_src += len(seq_src)
#      
#             res[-1].append(seq_src)
#             num_ex += 1  
#     return res, MakeDataInfosOneSide(total_count_unk_src,
#                                               total_token_src,
#                                               num_ex
#                                               )


def do_make_data(config):
#     raw_input("Press Enter to Continue 222")
    
    save_prefix_dir, save_prefix_fn = os.path.split(config.save_prefix)
    ensure_path(save_prefix_dir)

    config_fn = config.save_prefix + ".data.config"
    voc_fn = config.save_prefix + ".voc"
    data_fn = config.save_prefix + ".data.json.gz"
#     valid_data_fn = config.save_prefix + "." + config.model + ".valid.data.npz"

    voc_fn_src = config.save_prefix + ".src.voc"
    voc_fn_tgt = config.save_prefix + ".tgt.voc"
    
    files_that_will_be_created = [config_fn, voc_fn, data_fn]
    
    if config.bpe_src is not None:
        bpe_data_file_src = config.save_prefix + ".src.bpe"
        files_that_will_be_created.append(bpe_data_file_src)
    
    if config.bpe_tgt is not None:
        bpe_data_file_tgt = config.save_prefix + ".tgt.bpe"
        files_that_will_be_created.append(bpe_data_file_tgt)
    
    already_existing_files = []
    for filename in files_that_will_be_created:  # , valid_data_fn]:
        if os.path.exists(filename):
            already_existing_files.append(filename)
    if len(already_existing_files) > 0:
        print "Warning: existing files are going to be replaced: ",  already_existing_files
        raw_input("Press Enter to Continue")

    if config.use_voc is not None:
        log.info("loading voc from %s" % config.use_voc)
#         src_voc, tgt_voc = json.load(open(config.use_voc))
#         src_pp = processors.load_pp_from_data(json.load(open(src_voc)))
#         tgt_pp = IndexingPrePostProcessor.make_from_serializable(tgt_voc)
        src_pp, tgt_pp = processors.load_pp_pair_from_file(config.use_voc)
    else:
        if config.bpe_src is not None:
            src_pp = (processors.SimpleSegmenter(config.src_segmentation_type) +
                      processors.BPEProcessing(bpe_data_file = bpe_data_file_src, symbols = config.bpe_src, separator = "._@@@") +
                    processors.IndexingPrePostProcessor(voc_limit = config.src_voc_size)
                    )
        else:
            src_pp = (processors.SimpleSegmenter(config.src_segmentation_type) +
                    processors.IndexingPrePostProcessor(voc_limit = config.src_voc_size)
                    )
        
        if config.latin_src:
            src_pp = processors.LatinScriptProcess() + src_pp
        
        if config.bpe_tgt is not None:
            tgt_pp = (processors.SimpleSegmenter(config.tgt_segmentation_type) +
                      processors.BPEProcessing(bpe_data_file = bpe_data_file_tgt, symbols = config.bpe_tgt, separator = "._@@@") +
                                 processors.IndexingPrePostProcessor(voc_limit = config.tgt_voc_size))
        else:
            tgt_pp = (processors.SimpleSegmenter(config.tgt_segmentation_type) +
                                 processors.IndexingPrePostProcessor(voc_limit = config.tgt_voc_size))
    
        if config.latin_tgt:
            tgt_pp = processors.LatinScriptProcess() + tgt_pp
    
    def load_data(src_fn, tgt_fn, max_nb_ex=None):

        training_data, stats_src, stats_tgt = processors.build_dataset_pp(
            src_fn, tgt_fn, src_pp, tgt_pp,
            max_nb_ex=max_nb_ex)

        log.info("src data stats:\n%s", stats_src.make_report())
        log.info("tgt data stats:\n%s", stats_tgt.make_report())

        return training_data



    log.info("loading training data from %s and %s" %
             (config.src_fn, config.tgt_fn))
    training_data = load_data(config.src_fn, config.tgt_fn, max_nb_ex=config.max_nb_ex)

    test_data = None
    if config.test_src is not None:
        log.info("loading test data from %s and %s" %
                 (config.test_src, config.test_tgt))
        test_data = load_data(
            config.test_src, config.test_tgt)


    dev_data = None
    if config.dev_src is not None:
        log.info("loading dev data from %s and %s" %
                 (config.dev_src, config.dev_tgt))
        dev_data = load_data(
            config.dev_src, config.dev_tgt)


#     if config.shuffle:
#         log.info("shuffling data")
#         if config.enable_fast_shuffle:
#             shuffle_in_unison_faster(data_input, data_target)
#         else:
#             data_input, data_target = shuffle_in_unison(data_input, data_target)
    log.info("saving config to %s" % config_fn)
    config.save_to(config_fn)
#     json.dump(config.__dict__, open(config_fn, "w"),
#               indent=2, separators=(',', ': '))

    log.info("saving voc to %s" % voc_fn)
    processors.save_pp_pair_to_file(src_pp, tgt_pp, voc_fn)
#     json.dump([src_pp.to_serializable(), tgt_pp.to_serializable()],
#               open(voc_fn, "w"), indent=2, separators=(',', ': '))

    log.info("saving train_data to %s" % data_fn)
    data_all = {"train": training_data}
    if test_data is not None:
        data_all["test"] = test_data
    if dev_data is not None:
        data_all["dev"] = dev_data

    json.dump(data_all, gzip.open(data_fn, "wb"),
              indent=2, separators=(',', ': '))

