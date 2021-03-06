#!/usr/bin/env python
"""eval.py: Use a RNNSearch Model"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import io
import json
import logging
import os.path
import re
import sys
import time
import unicodedata
from collections import Counter
from typing import List, Optional

# import h5py
import bokeh.embed
import numpy as np
import six
from chainer import cuda, serializers

import nmt_chainer.dataprocessing.make_data as make_data
import nmt_chainer.dataprocessing.make_data_conf as make_data_conf
import nmt_chainer.training_module.train as train
import nmt_chainer.training_module.train_config as train_config
import nmt_chainer.translation.beam_search as beam_search
from nmt_chainer.dataprocessing.processors import build_dataset_one_side_pp
from nmt_chainer.translation.beam_search import BeamSearchParams
# import visualisation
from nmt_chainer.utilities import bleu_computer
from nmt_chainer.utilities.argument_parsing_tools import OrderedNamespace
from nmt_chainer.utilities.file_infos import create_filename_infos
from nmt_chainer.utilities.utils import ensure_path

from . import dictionnary_handling

# from utils import make_batch_src, make_batch_src_tgt, minibatch_provider, compute_bleu_with_unk_as_wrong, de_batch
from nmt_chainer.translation.evaluation import (  # convert_idx_to_string,; convert_idx_to_string_with_attn
    batch_align, beam_search_translate, greedy_batch_translate)

__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"





logging.basicConfig()
log = logging.getLogger("rnns:eval")
log.setLevel(logging.INFO)


class AttentionVisualizer(object):
    def __init__(self):
        self.plots_list = []

    def add_plot(self, src_w, tgt_w, attn, include_sum=True, visual_attribs=None):
        from nmt_chainer.utilities import visualisation
        alignment = np.zeros((len(src_w) + 1, len(tgt_w)))
        sum_al = [0] * len(tgt_w)
        for i in six.moves.range(len(src_w)):
            for j in six.moves.range(len(tgt_w)):
                alignment[i, j] = attn[j][i]
                sum_al[j] += alignment[i, j]
        for j in six.moves.range(len(tgt_w)):
            alignment[len(src_w), j] = sum_al[j]

        src = src_w
        if include_sum:
            src += ["SUM_ATTN"]
        if visual_attribs is not None:
            p1 = visualisation.make_alignment_figure(
                src, tgt_w, alignment, **visual_attribs)
        else:
            p1 = visualisation.make_alignment_figure(src, tgt_w, alignment)

        self.plots_list.append(p1)

    def make_plot(self, output_file):
        from nmt_chainer.utilities import visualisation
        log.info("writing attention to {0}".format(output_file))
        p_all = visualisation.Column(*self.plots_list)
        if isinstance(output_file, tuple):
            script_output_fn, div_output_fn = output_file
            script, div = bokeh.embed.components(p_all)
            with open(script_output_fn, 'wt') as f:
                f.write(script)
            with open(div_output_fn, 'wt') as f:
                f.write(div)
        else:
            visualisation.output_file(output_file)
            visualisation.show(p_all)


class RichOutputWriter(object):
    def __init__(self, filename):
        log.info("writing JSON translation infos to %s" % filename)
        self.filename = filename
        self.output = open(filename, "w")
        self.no_entry_yet = True
        self.output.write("[\n")

    def add_info(self, src, translated, t, score, attn, unk_mapping=None):
        if not self.no_entry_yet:
            self.output.write(",\n")
        else:
            self.no_entry_yet = False
        self.output.write("{\"tr\": ")
        self.output.write(json.dumps(translated))
        self.output.write(",\n\"attn\": ")
        self.output.write(json.dumps(
            [[float(a) for a in a_list] for a_list in attn]))
        if unk_mapping is not None:
            self.output.write(",\n\"unk_mapping\": ")
            self.output.write(json.dumps(unk_mapping))
        self.output.write("}")

    def finish(self):
        self.output.write("\n]")
        self.output.close()
        log.info("done writing JSON translation infos to %s" % self.filename)


def beam_search_all(gpu, encdec, eos_idx, src_data,
                    beam_search_params:beam_search.BeamSearchParams,
                    nb_steps,
                    nb_steps_ratio, 
                    post_score_length_normalization, post_score_length_normalization_strength,
                    post_score_coverage_penalty, post_score_coverage_penalty_strength,
                    groundhog,
                    tgt_unk_id, tgt_indexer, 
                    prob_space_combination=False, reverse_encdec=None,
                    replace_unk=False,
                    src=None,
                    dic=None,
                    remove_unk=False,
                    normalize_unicode_unk=False,
                    attempt_to_relocate_unk_source=False,
                    nbest=None,
                    constraints_fn_list:Optional[List[beam_search.BeamSearchConstraints]]=None,
                    use_astar=False,
                    astar_params:beam_search.AStarParams=beam_search.AStarParams(),
                    use_chainerx=False,
                    show_progress_bar=True,
                    thread=None):

    log.info("starting beam search translation of %i sentences" % len(src_data))
    if isinstance(encdec, (list, tuple)) and len(encdec) > 1:
        log.info("using ensemble of %i models" % len(encdec))

    with cuda.get_device_from_id(gpu):
        translations_gen = beam_search_translate(
            encdec, eos_idx, src_data, 
            beam_search_params,
            nb_steps=nb_steps,
            gpu=gpu, 
            nb_steps_ratio=nb_steps_ratio,
            need_attention=True,
            post_score_length_normalization=post_score_length_normalization,
            post_score_length_normalization_strength=post_score_length_normalization_strength,
            post_score_coverage_penalty=post_score_coverage_penalty,
            post_score_coverage_penalty_strength=post_score_coverage_penalty_strength,
            groundhog=groundhog, 
            prob_space_combination=prob_space_combination,
            reverse_encdec=reverse_encdec,
            nbest=nbest,
            constraints_fn_list=constraints_fn_list,
            use_astar=use_astar,
            astar_params=astar_params,
            use_chainerx=use_chainerx,
            show_progress_bar=show_progress_bar,
            thread=thread)

        for num_t, translations in enumerate(translations_gen):
            res_trans = []
            for trans in translations:
                (t, score, attn) = trans
#                 t, score = bests[1]
#                 ct = convert_idx_to_string(t, tgt_voc + ["#T_UNK#"])
#                 ct = convert_idx_to_string_with_attn(t, tgt_voc, attn, unk_idx = len(tgt_voc))
                if tgt_unk_id == "align":
                    def unk_replacer(num_pos, unk_id):
                        unk_pattern = "#T_UNK_%i#"
                        a = attn[num_pos]
                        xp = cuda.get_array_module(a)
                        src_pos = int(xp.argmax(a))
                        return unk_pattern % src_pos
                elif tgt_unk_id == "id":
                    def unk_replacer(num_pos, unk_id):
                        unk_pattern = "#T_UNK_%i#"
                        return unk_pattern % unk_id
                else:
                    assert False

                translated = tgt_indexer.deconvert_swallow(t, unk_tag=unk_replacer)

                unk_mapping = []
                ct = " ".join(translated)
                if ct != '':
                    unk_pattern = re.compile(r"#T_UNK_(\d+)#")
                    for idx, word in enumerate(ct.split(' ')):
                        match = unk_pattern.match(word)
                        if (match):
                            unk_mapping.append(match.group(1) + '-' + str(idx))

                    if replace_unk:
                        from nmt_chainer.utilities import replace_tgt_unk
                        translated = replace_tgt_unk.replace_unk_from_string(ct, src, dic, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source).strip().split(" ")

                res_trans.append((src_data[num_t], translated, t, score, attn, unk_mapping))

            yield res_trans


def translate_to_file_with_beam_search(dest_fn, gpu, encdec, eos_idx, src_data, 
                                       beam_search_params:beam_search.BeamSearchParams,
                                       nb_steps,
                                       nb_steps_ratio, 
                                       post_score_length_normalization, post_score_length_normalization_strength,
                                       post_score_coverage_penalty, post_score_coverage_penalty_strength,
                                       groundhog,
                                       tgt_unk_id, tgt_indexer, 
                                       prob_space_combination=False, reverse_encdec=None,
                                       generate_attention_html=None, attn_graph_with_sum=True, attn_graph_attribs=None, src_indexer=None, rich_output_filename=None,
                                       replace_unk=False,
                                       src=None,
                                       dic=None,
                                       remove_unk=False,
                                       normalize_unicode_unk=False,
                                       attempt_to_relocate_unk_source=False,
                                       unprocessed_output_filename=None,
                                       nbest=None,
                                       constraints_fn_list:Optional[List[beam_search.BeamSearchConstraints]]=None,
                                       use_astar=False,
                                       astar_params:beam_search.AStarParams=beam_search.AStarParams(),
                                       use_chainerx=False,
                                       show_progress_bar=True,
                                       thread=None):

    log.info("writing translation to %s " % dest_fn)
    out = io.open(dest_fn, "wt", encoding="utf8")

    translation_iterator = beam_search_all(gpu, encdec, eos_idx, src_data, 
                                           beam_search_params,
                                           nb_steps,
                                           nb_steps_ratio, 
                                           post_score_length_normalization, post_score_length_normalization_strength,
                                           post_score_coverage_penalty, post_score_coverage_penalty_strength,
                                           groundhog,
                                           tgt_unk_id, tgt_indexer, 
                                           prob_space_combination=prob_space_combination, reverse_encdec=reverse_encdec,
                                           replace_unk=replace_unk,
                                           src=src,
                                           dic=dic,
                                           remove_unk=remove_unk,
                                           normalize_unicode_unk=normalize_unicode_unk,
                                           attempt_to_relocate_unk_source=attempt_to_relocate_unk_source,
                                           nbest=nbest,
                                           constraints_fn_list=constraints_fn_list,
                                           use_astar=use_astar,
                                           astar_params=astar_params,
                                           use_chainerx=use_chainerx,
                                           show_progress_bar=show_progress_bar,
                                           thread=thread)

    attn_vis = None
    if generate_attention_html is not None:
        attn_vis = AttentionVisualizer()
        assert src_indexer is not None

    rich_output = None
    if rich_output_filename is not None:
        rich_output = RichOutputWriter(rich_output_filename)

    unprocessed_output = None
    if unprocessed_output_filename is not None:
        unprocessed_output = io.open(unprocessed_output_filename, "wt", encoding="utf8")

    for idx, translations in enumerate(translation_iterator):
        for src, translated, t, score, attn, unk_mapping in translations:
            if rich_output is not None:
                rich_output.add_info(src, translated, t, score, attn, unk_mapping=unk_mapping)
            if attn_vis is not None:
                attn_vis.add_plot(
                    src_indexer.deconvert_swallow(src),
                    translated,
                    attn,
                    attn_graph_with_sum,
                    attn_graph_attribs)
            ct = tgt_indexer.deconvert_post(translated)
            if nbest is not None:
                out.write("{0} ||| {1}\n".format(idx, ct))
            else:
                out.write(ct + "\n")
            if unprocessed_output is not None:
                unprocessed_output.write(" ".join(translated) + "\n")

    if rich_output is not None:
        rich_output.finish()

    if attn_vis is not None:
        attn_vis.make_plot(generate_attention_html)


def create_and_load_encdec_from_files(config_training_fn, trained_model):
    log.info("loading model config from %s" % config_training_fn)

    config_training = train_config.load_config_train(config_training_fn)
    encdec, eos_idx, src_indexer, tgt_indexer = train.create_encdec_and_indexers_from_config_dict(config_training)

    log.info("loading model from %s" % trained_model)
    serializers.load_npz(trained_model, encdec)

    return encdec, eos_idx, src_indexer, tgt_indexer


def check_if_vocabulary_info_compatible(this_eos_idx, this_src_indexer, this_tgt_indexer, eos_idx, src_indexer, tgt_indexer):
    if eos_idx != this_eos_idx:
        raise Exception("incompatible models")
    if len(src_indexer) != len(this_src_indexer):
        raise Exception("incompatible models")
    if len(tgt_indexer) != len(this_tgt_indexer):
        raise Exception("incompatible models")


def get_src_tgt_dev_from_config_eval(config_eval):
    if config_eval.training_config is not None:
        training_config_file = config_eval.training_config
    elif 'load_model_config' in config_eval.process and config_eval.process.load_model_config is not None:
        training_config_file = config_eval.process.load_model_config[0]
        if "," in training_config_file:
            training_config_file=training_config_file.split(",")[0]
    log.info("attempting to retrieve dev/test files from %s", training_config_file)
    training_config = train_config.load_config_train(training_config_file)
    data_prefix = training_config.training_management.data_prefix
    data_config_filename = data_prefix + ".data.config"
    data_config = make_data_conf.load_config(data_config_filename)
    return (data_config["data"]["dev_src"],
            data_config["data"]["dev_tgt"],
            data_config["data"]["test_src"],
            data_config["data"]["test_tgt"])


def create_encdec(config_eval):
    encdec_list = []
    eos_idx, src_indexer, tgt_indexer = None, None, None
    model_infos_list = []

    if config_eval.training_config is not None:
        assert config_eval.trained_model is not None
        encdec, eos_idx, src_indexer, tgt_indexer = create_and_load_encdec_from_files(
            config_eval.training_config, config_eval.trained_model)
        model_infos_list.append(create_filename_infos(config_eval.trained_model))
        encdec_list.append(encdec)

    if 'load_model_config' in config_eval.process and config_eval.process.load_model_config is not None:
        for config_filename_and_others in config_eval.process.load_model_config:
            other_models_for_averaging = None
            if "," in config_filename_and_others:
                config_filename_and_others_splitted = config_filename_and_others.split(",")
                config_filename = config_filename_and_others_splitted[0]
                other_models_for_averaging = config_filename_and_others_splitted[1:]
            else:
                config_filename = config_filename_and_others
            log.info(
                "loading model and parameters from config %s" %
                config_filename)
            config_training = train_config.load_config_train(config_filename)
            (encdec, this_eos_idx, this_src_indexer, this_tgt_indexer), model_infos = train.create_encdec_and_indexers_from_config_dict(config_training,
                                                                                                                                        load_config_model="yes",
                                                                                                                                        return_model_infos=True,
                                                                                                                                        additional_models_parameters_for_averaging=other_models_for_averaging)
            model_infos_list.append(model_infos)
            if eos_idx is None:
                assert len(encdec_list) == 0
                assert src_indexer is None
                assert tgt_indexer is None
                eos_idx, src_indexer, tgt_indexer = this_eos_idx, this_src_indexer, this_tgt_indexer
            else:
                check_if_vocabulary_info_compatible(this_eos_idx, this_src_indexer, this_tgt_indexer, eos_idx, src_indexer, tgt_indexer)

            encdec_list.append(encdec)

    assert len(encdec_list) > 0

    if 'additional_training_config' in config_eval.process and config_eval.process.additional_training_config is not None:
        assert len(config_eval.process.additional_training_config) == len(config_eval.process.additional_trained_model)

        for (config_training_fn, trained_model_fn) in six.moves.zip(config_eval.process.additional_training_config,
                                                          config_eval.process.additional_trained_model):
            this_encdec, this_eos_idx, this_src_indexer, this_tgt_indexer = create_and_load_encdec_from_files(
                config_training_fn, trained_model_fn)

            check_if_vocabulary_info_compatible(this_eos_idx, this_src_indexer, this_tgt_indexer, eos_idx, src_indexer, tgt_indexer)
            model_infos_list.append(create_filename_infos(trained_model_fn))

            encdec_list.append(this_encdec)

    if config_eval.process.use_chainerx:
        if 'gpu' in config_eval.process and config_eval.process.gpu is not None:
            encdec_list = [encdec.to_device("cuda:%i"%config_eval.process.gpu) for encdec in encdec_list]
        else:
            encdec_list = [encdec.to_device("native:0") for encdec in encdec_list]
    else:
        if 'gpu' in config_eval.process and config_eval.process.gpu is not None:
            encdec_list = [encdec.to_gpu(config_eval.process.gpu) for encdec in encdec_list]

        

    if 'reverse_training_config' in config_eval.process and config_eval.process.reverse_training_config is not None:
        reverse_encdec, reverse_eos_idx, reverse_src_indexer, reverse_tgt_indexer = create_and_load_encdec_from_files(
            config_eval.process.reverse_training_config, config_eval.process.reverse_trained_model)

        if eos_idx != reverse_eos_idx:
            raise Exception("incompatible models")

        if len(src_indexer) != len(reverse_src_indexer):
            raise Exception("incompatible models")

        if len(tgt_indexer) != len(reverse_tgt_indexer):
            raise Exception("incompatible models")

        if config_eval.process.gpu is not None:
            reverse_encdec = reverse_encdec.to_gpu(config_eval.process.gpu)
    else:
        reverse_encdec = None

    return encdec_list, eos_idx, src_indexer, tgt_indexer, reverse_encdec, model_infos_list

def placeholder_constraints_builder(src_indexer, tgt_indexer, units_placeholders=False):
    if False: #not units_placeholders:
        placeholder_matcher = re.compile(r"<K-\d+>")
        def make_constraints(src, src_seq)->beam_search.BeamSearchConstraints:
            src_placeholders_list = placeholder_matcher.findall(src)
            src_placeholders_set = set(src_placeholders_list)
            if len(src_placeholders_set) != len(src_placeholders_list):
                raise Exception("Current assumption is that there is no duplicate placeholders")
            def constraint_fn(tgt_seq):
                tgt = tgt_indexer.deconvert(tgt_seq)
                tgt_placeholders_list = placeholder_matcher.findall(tgt)
                tgt_placeholders_set = set(tgt_placeholders_list)
                if len(tgt_placeholders_set) != len(tgt_placeholders_list):
                    return -1 #disallow duplicate placeholders on target side
                if len(tgt_placeholders_set - src_placeholders_set) != 0:
                    return -1 #disallow generating a placeholder not in source
                if tgt_placeholders_set == src_placeholders_set:
                    return 1 #all constraints satisfied
                #else return proportion of required placeholders in target
                assert len(tgt_placeholders_set) < len(src_placeholders_set)
                return len(tgt_placeholders_set) / len(src_placeholders_set)
            
            return beam_search.BeamSearchConstraints(constraint_fn=constraint_fn)
        return make_constraints
    else:
        placeholder_dictionary = {}
        placeholders_list_all = []
        placeholders_list = []
        for i in range(100):
            placeholder = "<K-%02i>"%i
            tgt_idx = tgt_indexer.convert(placeholder)
            src_idx = src_indexer.convert(placeholder)
            if len(tgt_idx) != 1 or len(src_idx)!=1:
                pass
            else:
                tgt_idx = tgt_idx[0]
                src_idx = src_idx[0]
                if not tgt_indexer.is_unk_idx(tgt_idx) and not src_indexer.is_unk_idx(src_idx):
                    placeholder_dictionary[src_idx] = tgt_idx
                    placeholders_list_all.append( (placeholder, src_idx, tgt_idx))
                    placeholders_list.append(tgt_idx)
        log.info("Found %i placeholders: %r  l:%i,%i"%(len(placeholders_list_all), placeholders_list_all, 
                len(tgt_indexer), len(src_indexer)))

        def make_constraints(src, src_seq)->beam_search.BeamSearchConstraints:
            required_tgt_idx = beam_search.TgtIdxConstraint()
            required_tgt_idx.set_placeholders_idx_list(placeholders_list)
            for idx in src_seq:
                if idx in placeholder_dictionary:
                    required_tgt_idx.add(placeholder_dictionary[idx])
            
            return beam_search.BeamSearchConstraints(required_tgt_idx=required_tgt_idx)
        return make_constraints


def do_eval(config_eval):
    src_fn = config_eval.process.src_fn
    tgt_fn = config_eval.output.tgt_fn
    mode = config_eval.method.mode
    gpu = config_eval.process.gpu
    dest_fn = config_eval.process.dest_fn
    mb_size = config_eval.process.mb_size
    nb_steps = config_eval.method.nb_steps
    nb_steps_ratio = config_eval.method.nb_steps_ratio
    max_nb_ex = config_eval.process.max_nb_ex
    nbest_to_rescore = config_eval.output.nbest_to_rescore
    nbest = config_eval.output.nbest

    beam_width = config_eval.method.beam_width
    beam_pruning_margin = config_eval.method.beam_pruning_margin
    beam_score_length_normalization = config_eval.method.beam_score_length_normalization
    beam_score_length_normalization_strength = config_eval.method.beam_score_length_normalization_strength
    beam_score_coverage_penalty = config_eval.beam_score_coverage_penalty
    beam_score_coverage_penalty_strength = config_eval.beam_score_coverage_penalty_strength
    always_consider_eos_and_placeholders = config_eval.method.always_consider_eos_and_placeholders

    if config_eval.process.force_placeholders:
        # making it  default for now
        always_consider_eos_and_placeholders = True

    post_score_length_normalization = config_eval.method.post_score_length_normalization
    post_score_length_normalization_strength = config_eval.method.post_score_length_normalization_strength
    groundhog = config_eval.method.groundhog
    tgt_unk_id = config_eval.output.tgt_unk_id
    force_finish = config_eval.method.force_finish
    prob_space_combination = config_eval.method.prob_space_combination
    generate_attention_html = config_eval.output.generate_attention_html
    rich_output_filename = config_eval.output.rich_output_filename

    ref = config_eval.output.ref
    dic = config_eval.output.dic
    normalize_unicode_unk = config_eval.output.normalize_unicode_unk
    attempt_to_relocate_unk_source = config_eval.output.attempt_to_relocate_unk_source
    remove_unk = config_eval.output.remove_unk

    post_score_coverage_penalty = config_eval.method.post_score_coverage_penalty
    post_score_coverage_penalty_strength = config_eval.method.post_score_coverage_penalty_strength

    time_start = time.perf_counter()

    

    astar_params = beam_search.AStarParams(
        astar_batch_size=config_eval.method.astar_batch_size,
        astar_max_queue_size=config_eval.method.astar_max_queue_size,
        astar_prune_margin=config_eval.method.astar_prune_margin,
        astar_prune_ratio=config_eval.method.astar_prune_ratio,
        length_normalization_exponent=config_eval.method.astar_length_normalization_exponent,
        length_normalization_constant=config_eval.method.astar_length_normalization_constant,
        astar_priority_eval_string=config_eval.method.astar_priority_eval_string,
        max_length_diff = config_eval.method.astar_max_length_diff)

    make_constraints_dict = None

    if config_eval.process.server is None and config_eval.process.multiserver_config is None:
        encdec_list, eos_idx, src_indexer, tgt_indexer, reverse_encdec, model_infos_list = create_encdec(config_eval)

        eval_dir_placeholder = "@eval@/"
        if dest_fn.startswith(eval_dir_placeholder):
            if config_eval.trained_model is not None:
                training_model_filename = config_eval.trained_model
            else:
                if len(config_eval.process.load_model_config) == 0:
                    log.error("Cannot detect value for $eval$ placeholder")
                    sys.exit(1)
                training_model_filename = config_eval.process.load_model_config[0]

            eval_dir = os.path.join(os.path.dirname(training_model_filename), "eval")
            dest_fn = os.path.join(eval_dir, dest_fn[len(eval_dir_placeholder):])
            log.info("$eval$ detected. dest_fn is: %s ", dest_fn)
            ensure_path(eval_dir)

        if src_fn is None:
            (dev_src_from_config, dev_tgt_from_config, test_src_from_config, test_tgt_from_config) = get_src_tgt_dev_from_config_eval(config_eval)
            if test_src_from_config is None:
                log.error("Could not find value for source text, either on command line or in config files")
                sys.exit(1)
            log.info("using files from config as src:%s", test_src_from_config)
            src_fn = test_src_from_config
            if ref is None:
                log.info("using files from config as ref:%s", test_tgt_from_config)
                ref = test_tgt_from_config

        if config_eval.process.force_placeholders:
            if make_constraints_dict is None:
                make_constraints_dict = {}
            make_constraints_dict["ph_constraint"] = placeholder_constraints_builder(src_indexer, tgt_indexer, 
                    units_placeholders=config_eval.process.units_placeholders)

                    

        if config_eval.process.bilingual_dic_for_reranking:
            if make_constraints_dict is None:
                make_constraints_dict = {}

            print("**making ja en dic")
            ja_en_search, en_ja_search = dictionnary_handling.load_search_trie(
                config_eval.process.bilingual_dic_for_reranking, config_eval.process.invert_bilingual_dic_for_reranking)
            
            print("**define constraints")
            make_constraints_dict["dic_constraint"] = dictionnary_handling.make_constraint(ja_en_search, en_ja_search, tgt_indexer)
            


        elif False:

            re_word = re.compile(r"[A-Za-z]+")
            re_digits = re.compile(r"\d+")
            def unsegment(s):
                res = []
                for w in s.split(" "):
                    if w.startswith("▁"):
                        w = " " + w[1:]
                    res.append(w)
                return "".join(res)

            def make_constraints(src, src_seq):
                line_src = unsegment(src)
                line_src = unicodedata.normalize('NFKC', line_src)
                word_list = [word for word in re_word.findall(line_src) if len(word) > 3]
                digit_list = [digit for digit in re_digits.findall(line_src) if len(digit) > 2]
                if len(word_list) == 0 and len(digit_list)==0:
                    def constraint_fn(tgt_seq):
                        return 1
                else:
                    def constraint_fn(tgt_seq):
                        tgt = tgt_indexer.deconvert(tgt_seq)
                        line_tgt = unsegment(tgt)
                        line_tgt = unicodedata.normalize('NFKC', line_tgt)
                        matched_word = 0
                        for word in word_list:
                            if word in line_ref:
                                matched_word += 1

                        matched_digit = 0
                        for digit in digit_list:
                            if digit in line_ref:
                                matched_digit += 1

                        if matched_word == len(word_list) and matched_digit == len(digit_list):
                            return 1
                        else:
                            return (matched_word + matched_digit)/(len(word_list) + len(digit_list))

                    return constraint_fn

        else:
            make_constraints_dict = None

        log.info("opening source file %s" % src_fn)

        preprocessed_input = build_dataset_one_side_pp(src_fn, src_pp=src_indexer,
                                                           max_nb_ex=max_nb_ex,
                                                           make_constraints_dict=make_constraints_dict)

        if make_constraints_dict is not None:
            src_data, stats_src_pp, constraints_list = preprocessed_input
        else:
             src_data, stats_src_pp = preprocessed_input
             constraints_list = None                                             
        log.info("src data stats:\n%s", stats_src_pp.make_report())

        translation_infos = OrderedNamespace()
        translation_infos["src"] = src_fn
        translation_infos["tgt"] = tgt_fn
        translation_infos["ref"] = ref

        for num_model, model_infos in enumerate(model_infos_list):
            translation_infos["model%i" % num_model] = model_infos

    if dest_fn is not None:
        save_eval_config_fn = dest_fn + ".eval.init.config.json"
        log.info("Saving initial eval config to %s" % save_eval_config_fn)
        config_eval.save_to(save_eval_config_fn)


#     log.info("%i sentences loaded" % make_data_infos.nb_ex)
#     log.info("#tokens src: %i   of which %i (%f%%) are unknown"%(make_data_infos.total_token,
#                                                                  make_data_infos.total_count_unk,
#                                                                  float(make_data_infos.total_count_unk * 100) /
#                                                                     make_data_infos.total_token))

    tgt_data = None
    if tgt_fn is not None:
        log.info("opening target file %s" % tgt_fn)
        tgt_data, stats_tgt_pp = build_dataset_one_side_pp(tgt_fn, src_pp=tgt_indexer,
                                                           max_nb_ex=max_nb_ex)
        log.info("tgt data stats:\n%s", stats_tgt_pp.make_report())
#         log.info("%i sentences loaded"%make_data_infos.nb_ex)
#         log.info("#tokens src: %i   of which %i (%f%%) are unknown"%(make_data_infos.total_token,
#                                                                  make_data_infos.total_count_unk,
#                                                                  float(make_data_infos.total_count_unk * 100) /
#                                                                     make_data_infos.total_token))

#     translations = greedy_batch_translate(encdec, eos_idx, src_data, batch_size = mb_size, gpu = args.gpu)



    time_all_loaded = time.perf_counter()

    if mode == "translate":
        log.info("writing translation of to %s" % dest_fn)
        with cuda.get_device_from_id(gpu):
            assert len(encdec_list) == 1
            translations = greedy_batch_translate(
                encdec_list[0], eos_idx, src_data, batch_size=mb_size, gpu=gpu, nb_steps=nb_steps, 
                                use_chainerx=config_eval.process.use_chainerx)
        out = io.open(dest_fn, "wt", encoding="utf8")
        for t in translations:
            if t[-1] == eos_idx:
                t = t[:-1]
            ct = tgt_indexer.deconvert(t, unk_tag="#T_UNK#")
#             ct = convert_idx_to_string(t, tgt_voc + ["#T_UNK#"])
            out.write(ct + "\n")

    elif mode == "beam_search" or mode == "eval_bleu" or mode == "astar_search" or mode == "astar_eval_bleu":
        if config_eval.process.server is not None:
            from nmt_chainer.translation.server import do_start_server
            do_start_server(config_eval)
        elif config_eval.process.multiserver_config is not None:
            from nmt_chainer.translation.multiserver import do_start_server
            do_start_server(config_eval.process.multiserver_config, config_eval.output.log_config)
        else:


            def translate_closure(beam_width, nb_steps_ratio):
                beam_search_params = beam_search.BeamSearchParams(
                                                beam_width=beam_width,
                                                beam_pruning_margin=beam_pruning_margin,
                                                beam_score_coverage_penalty=beam_score_coverage_penalty,
                                                beam_score_coverage_penalty_strength=beam_score_coverage_penalty_strength,
                                                beam_score_length_normalization=beam_score_length_normalization,
                                                beam_score_length_normalization_strength=beam_score_length_normalization_strength,
                                                force_finish=force_finish,
                                                use_unfinished_translation_if_none_found=True,
                                                always_consider_eos_and_placeholders=always_consider_eos_and_placeholders
                )


                translate_to_file_with_beam_search(dest_fn, gpu, encdec_list, eos_idx, src_data,
                                                beam_search_params=beam_search_params,
                                                nb_steps=nb_steps,
                                                nb_steps_ratio=nb_steps_ratio,
                                                post_score_length_normalization=post_score_length_normalization,
                                                post_score_length_normalization_strength=post_score_length_normalization_strength,
                                                post_score_coverage_penalty=post_score_coverage_penalty,
                                                post_score_coverage_penalty_strength=post_score_coverage_penalty_strength,
                                                groundhog=groundhog,
                                                tgt_unk_id=tgt_unk_id,
                                                tgt_indexer=tgt_indexer,
                                                prob_space_combination=prob_space_combination,
                                                reverse_encdec=reverse_encdec,
                                                generate_attention_html=generate_attention_html,
                                                src_indexer=src_indexer,
                                                rich_output_filename=rich_output_filename,
                                                unprocessed_output_filename=dest_fn + ".unprocessed",
                                                nbest=nbest,
                                                constraints_fn_list=constraints_list,
                                                use_astar= (mode == "astar_search" or mode == "astar_eval_bleu"),
                                                astar_params=astar_params,
                                                use_chainerx=config_eval.process.use_chainerx)

                translation_infos["dest"] = dest_fn
                translation_infos["unprocessed"] = dest_fn + ".unprocessed"
                if mode == "eval_bleu" or mode == "astar_eval_bleu":
                    if ref is not None:
                        bc = bleu_computer.get_bc_from_files(ref, dest_fn)
                        print("bleu before unk replace:", bc)
                        translation_infos["bleu"] = bc.bleu()
                        translation_infos["bleu_infos"] = str(bc)
                    else:
                        print("bleu before unk replace: No Ref Provided")

                    from nmt_chainer.utilities import replace_tgt_unk
                    replace_tgt_unk.replace_unk(dest_fn, src_fn, dest_fn + ".unk_replaced", dic, remove_unk,
                                                normalize_unicode_unk,
                                                attempt_to_relocate_unk_source)
                    translation_infos["unk_replaced"] = dest_fn + ".unk_replaced"

                    if ref is not None:
                        bc = bleu_computer.get_bc_from_files(ref, dest_fn + ".unk_replaced")
                        print("bleu after unk replace:", bc)
                        translation_infos["post_unk_bleu"] = bc.bleu()
                        translation_infos["post_unk_bleu_infos"] = str(bc)
                    else:
                        print("bleu before unk replace: No Ref Provided")
                    return -bc.bleu()
                else:
                    return None

            if  config_eval.process.do_hyper_param_search is not None:
                study_filename, study_name, n_trials = do_hyper_param_search
                n_trials = int(n_trials)
                import optuna
                def objective(trial):
                    nb_steps_ratio = trial.suggest_uniform('nb_steps_ratio', 0.9, 3.5)
                    beam_width = trial.suggest_int("beam_width", 2, 50)
                    return translate_closure(beam_width, nb_steps_ratio)
                study = optuna.create_study(study_name=study_name, storage="sqlite:///" + study_filename)
                study.optimize(objective, n_trials=n_trials)
                print(study.best_params)
                print(study.best_value)
                print(study.best_trial)
                
            else: # hyperparams optim
                translate_closure(beam_width, nb_steps_ratio)

    elif mode == "translate_attn":
        log.info("writing translation + attention as html to %s" % dest_fn)
        with cuda.get_device_from_id(gpu):
            assert len(encdec_list) == 1
            translations, attn_all = greedy_batch_translate(
                encdec_list[0], eos_idx, src_data, batch_size=mb_size, gpu=gpu,
                get_attention=True, nb_steps=nb_steps, use_chainerx=config_eval.process.use_chainerx)
#         tgt_voc_with_unk = tgt_voc + ["#T_UNK#"]
#         src_voc_with_unk = src_voc + ["#S_UNK#"]
        assert len(translations) == len(src_data)
        assert len(attn_all) == len(src_data)
        attn_vis = AttentionVisualizer()
        for num_t in six.moves.range(len(src_data)):
            src_idx_list = src_data[num_t]
            tgt_idx_list = translations[num_t][:-1]
            attn = attn_all[num_t]
#             assert len(attn) == len(tgt_idx_list)

            src_w = src_indexer.deconvert_swallow(src_idx_list, unk_tag="#S_UNK#") + ["SUM_ATTN"]
            tgt_w = tgt_indexer.deconvert_swallow(tgt_idx_list, unk_tag="#T_UNK#")
#             src_w = [src_voc_with_unk[idx] for idx in src_idx_list] + ["SUM_ATTN"]
#             tgt_w = [tgt_voc_with_unk[idx] for idx in tgt_idx_list]
#             for j in six.moves.range(len(tgt_idx_list)):
#                 tgt_idx_list.append(tgt_voc_with_unk[t_and_attn[j][0]])
#
    #         print([src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list)

            attn_vis.add_plot(src_w, tgt_w, attn)

        attn_vis.make_plot(dest_fn)

    elif mode == "align":
        import nmt_chainer.utilities.visualisation as visualisation
        assert tgt_data is not None
        assert len(tgt_data) == len(src_data)
        log.info("writing alignment as html to %s" % dest_fn)
        with cuda.get_device_from_id(gpu):
            assert len(encdec_list) == 1
            loss, attn_all = batch_align(
                encdec_list[0], eos_idx, list(six.moves.zip(src_data, tgt_data)), batch_size=mb_size, gpu=gpu, 
                            use_chainerx=config_eval.process.use_chainerx)
#         tgt_voc_with_unk = tgt_voc + ["#T_UNK#"]
#         src_voc_with_unk = src_voc + ["#S_UNK#"]

        assert len(attn_all) == len(src_data)
        plots_list = []
        for num_t in six.moves.range(len(src_data)):
            src_idx_list = src_data[num_t]
            tgt_idx_list = tgt_data[num_t]
            attn = attn_all[num_t]
#             assert len(attn) == len(tgt_idx_list)

            alignment = np.zeros((len(src_idx_list) + 1, len(tgt_idx_list)))
            sum_al = [0] * len(tgt_idx_list)
            for i in six.moves.range(len(src_idx_list)):
                for j in six.moves.range(len(tgt_idx_list)):
                    alignment[i, j] = attn[j][i]
                    sum_al[j] += alignment[i, j]
            for j in six.moves.range(len(tgt_idx_list)):
                alignment[len(src_idx_list), j] = sum_al[j]

            src_w = src_indexer.deconvert_swallow(src_idx_list, unk_tag="#S_UNK#") + ["SUM_ATTN"]
            tgt_w = tgt_indexer.deconvert_swallow(tgt_idx_list, unk_tag="#T_UNK#")
#             src_w = [src_voc_with_unk[idx] for idx in src_idx_list] + ["SUM_ATTN"]
#             tgt_w = [tgt_voc_with_unk[idx] for idx in tgt_idx_list]
#             for j in six.moves.range(len(tgt_idx_list)):
#                 tgt_idx_list.append(tgt_voc_with_unk[t_and_attn[j][0]])
#
    #         print([src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list)
            p1 = visualisation.make_alignment_figure(
                src_w, tgt_w, alignment)
#             p2 = visualisation.make_alignment_figure(
#                             [src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list, alignment)
            plots_list.append(p1)
        p_all = visualisation.Column(*plots_list)
        visualisation.output_file(dest_fn)
        visualisation.show(p_all)
#     for t in translations_with_attn:
#         for x, attn in t:
#             print(x, attn)


#         out.write(convert_idx_to_string([x for x, attn in t], tgt_voc + ["#T_UNK#"]) + "\n")

    elif mode == "score_nbest":
        log.info("opening nbest file %s" % nbest_to_rescore)
        nbest_f = io.open(nbest_to_rescore, 'rt', encoding="utf8")
        nbest_list = [[]]
        for line in nbest_f:
            line = line.strip().split("|||")
            num_src = int(line[0].strip())
            if num_src >= len(nbest_list):
                assert num_src == len(nbest_list)
                if max_nb_ex is not None and num_src >= max_nb_ex:
                    break
                nbest_list.append([])
            else:
                assert num_src == len(nbest_list) - 1
            sentence = line[1].strip()
            nbest_list[-1].append(sentence.split(" "))

        log.info("found nbest lists for %i source sentences" % len(nbest_list))
        nbest_converted, make_data_infos = make_data.build_dataset_for_nbest_list_scoring(tgt_indexer, nbest_list)
        log.info("total %i sentences loaded" % make_data_infos.nb_ex)
        log.info("#tokens src: %i   of which %i (%f%%) are unknown" % (make_data_infos.total_token,
                                                                       make_data_infos.total_count_unk,
                                                                       float(make_data_infos.total_count_unk * 100) /
                                                                       make_data_infos.total_token))
        if len(nbest_list) != len(src_data[:max_nb_ex]):
            log.warn("mismatch in lengths nbest vs src : %i != %i" % (len(nbest_list), len(src_data[:max_nb_ex])))
            assert len(nbest_list) == len(src_data[:max_nb_ex])

        log.info("starting scoring")
        from nmt_chainer.utilities import utils
        res = []
        for num in six.moves.range(len(nbest_converted)):
            if num % 200 == 0:
                print(num, file=sys.stderr)
            elif num % 50 == 0:
                print("*", file=sys.stderr)

            res.append([])
            src, tgt_list = src_data[num], nbest_converted[num]
            src_batch, src_mask = utils.make_batch_src([src], gpu=gpu, volatile="on")

            assert len(encdec_list) == 1
            scorer = encdec_list[0].nbest_scorer(src_batch, src_mask)

            nb_batches = (len(tgt_list) + mb_size - 1) // mb_size
            for num_batch in six.moves.range(nb_batches):
                tgt_batch, arg_sort = utils.make_batch_tgt(tgt_list[num_batch * nb_batches: (num_batch + 1) * nb_batches],
                                                           eos_idx=eos_idx, gpu=gpu, volatile="on", need_arg_sort=True)
                scores, attn = scorer(tgt_batch)
                scores, _ = scores
                scores = scores.data

                assert len(arg_sort) == len(scores)
                de_sorted_scores = [None] * len(scores)
                for xpos in six.moves.range(len(arg_sort)):
                    original_pos = arg_sort[xpos]
                    de_sorted_scores[original_pos] = scores[xpos]
                res[-1] += de_sorted_scores
        print('', file=sys.stderr)
        log.info("writing scores to %s" % dest_fn)
        out = io.open(dest_fn, "wt", encoding="utf8")
        for num in six.moves.range(len(res)):
            for score in res[num]:
                out.write("%i %f\n" % (num, score))

    time_end = time.perf_counter()
    translation_infos["loading_time"] = time_all_loaded - time_start
    translation_infos["translation_time"] = time_end - time_all_loaded
    translation_infos["total_time"] = time_end - time_start
    if dest_fn is not None:
        config_eval_session = config_eval.copy(readonly=False)
        config_eval_session.add_section("translation_infos", keep_at_bottom="metadata")
        config_eval_session["translation_infos"] = translation_infos
        config_eval_session.set_metadata_modified_time()
        save_eval_config_fn = dest_fn + ".eval.config.json"
        log.info("Saving eval config to %s" % save_eval_config_fn)
        config_eval_session.save_to(save_eval_config_fn)
