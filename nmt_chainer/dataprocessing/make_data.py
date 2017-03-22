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


def do_make_data(config):
    #     raw_input("Press Enter to Continue 222")

    save_prefix_dir, save_prefix_fn = os.path.split(config.data.save_prefix)
    ensure_path(save_prefix_dir)

    config_fn = config.data.save_prefix + ".data.config"
    voc_fn = config.data.save_prefix + ".voc"
    data_fn = config.data.save_prefix + ".data.json.gz"
#     valid_data_fn = config.save_prefix + "." + config.model + ".valid.data.npz"

#     voc_fn_src = config.save_prefix + ".src.voc"
#     voc_fn_tgt = config.save_prefix + ".tgt.voc"

    files_that_will_be_created = [config_fn, voc_fn, data_fn]

    if config.processing.bpe_src is not None:
        bpe_data_file_src = config.data.save_prefix + ".src.bpe"
        files_that_will_be_created.append(bpe_data_file_src)

    if config.processing.bpe_tgt is not None:
        bpe_data_file_tgt = config.data.save_prefix + ".tgt.bpe"
        files_that_will_be_created.append(bpe_data_file_tgt)

    if config.processing.joint_bpe is not None:
        bpe_data_file_joint = config.data.save_prefix + ".joint.bpe"
        files_that_will_be_created.append(bpe_data_file_joint)

    already_existing_files = []
    for filename in files_that_will_be_created:  # , valid_data_fn]:
        if os.path.exists(filename):
            already_existing_files.append(filename)
    if len(already_existing_files) > 0:
        print "Warning: existing files are going to be replaced: ", already_existing_files
        raw_input("Press Enter to Continue")

    if config.processing.use_voc is not None:
        log.info("loading voc from %s" % config.processing.use_voc)
#         src_voc, tgt_voc = json.load(open(config.use_voc))
#         src_pp = processors.load_pp_from_data(json.load(open(src_voc)))
#         tgt_pp = IndexingPrePostProcessor.make_from_serializable(tgt_voc)
        bi_idx = processors.load_pp_pair_from_file(config.processing.use_voc)
    else:

        bi_idx = processors.BiIndexingPrePostProcessor(voc_limit1=config.processing.src_voc_size,
                                                       voc_limit2=config.processing.tgt_voc_size)
        pp = processors.BiProcessorChain()

        if config.processing.latin_tgt:
            pp.add_tgt_processor(processors.LatinScriptProcess(config.processing.latin_type))

        if config.processing.latin_src:
            pp.add_src_processor(processors.LatinScriptProcess(config.processing.latin_type))

        pp.add_src_processor(processors.SimpleSegmenter(config.processing.src_segmentation_type))
        if config.processing.bpe_src is not None:
            pp.add_src_processor(
                processors.BPEProcessing(bpe_data_file=bpe_data_file_src, symbols=config.processing.bpe_src, separator="._@@@"))

        pp.add_tgt_processor(processors.SimpleSegmenter(config.processing.tgt_segmentation_type))
        if config.processing.bpe_tgt is not None:
            pp.add_tgt_processor(
                processors.BPEProcessing(bpe_data_file=bpe_data_file_tgt, symbols=config.processing.bpe_tgt, separator="._@@@"))

        if config.processing.joint_bpe is not None:
            pp.add_biprocessor(processors.JointBPEBiProcessor(bpe_data_file=bpe_data_file_joint,
                                                              symbols=config.processing.joint_bpe, separator="._@@@"))

        bi_idx.add_preprocessor(pp)

    def load_data(src_fn, tgt_fn, max_nb_ex=None, infos_dict=None):

        training_data, stats_src, stats_tgt = processors.build_dataset_pp(
            src_fn, tgt_fn, bi_idx,
            max_nb_ex=max_nb_ex)

        log.info("src data stats:\n%s", stats_src.make_report())
        log.info("tgt data stats:\n%s", stats_tgt.make_report())

        if infos_dict is not None:
            infos_dict["src"] = stats_src.report_as_obj()
            infos_dict["tgt"] = stats_tgt.report_as_obj()

        return training_data

    infos = collections.OrderedDict()
    infos["train"] = collections.OrderedDict()

    log.info("loading training data from %s and %s" %
             (config.data.src_fn, config.data.tgt_fn))
    training_data = load_data(config.data.src_fn, config.data.tgt_fn, max_nb_ex=config.data.max_nb_ex, infos_dict=infos["train"])

    dev_data = None
    if config.data.dev_src is not None:
        log.info("loading dev data from %s and %s" %
                 (config.data.dev_src, config.data.dev_tgt))
        infos["dev"] = collections.OrderedDict()
        dev_data = load_data(
            config.data.dev_src, config.data.dev_tgt, infos_dict=infos["dev"])

    test_data = None
    if config.data.test_src is not None:
        log.info("loading test data from %s and %s" %
                 (config.data.test_src, config.data.test_tgt))
        infos["test"] = collections.OrderedDict()
        test_data = load_data(
            config.data.test_src, config.data.test_tgt, infos_dict=infos["test"])

    config.insert_section("infos", infos, even_if_readonly=True, keep_at_bottom="metadata", overwrite=False)

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
    processors.save_pp_pair_to_file(bi_idx, voc_fn)
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
