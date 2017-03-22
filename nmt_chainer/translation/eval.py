#!/usr/bin/env python
"""eval.py: Use a RNNSearch Model"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import json
import numpy as np
from chainer import cuda, serializers
import sys
from nmt_chainer.dataprocessing.processors import build_dataset_one_side_pp
import nmt_chainer.dataprocessing.make_data as make_data
import nmt_chainer.training_module.train as train
import nmt_chainer.training_module.train_config as train_config
import re

# from utils import make_batch_src, make_batch_src_tgt, minibatch_provider, compute_bleu_with_unk_as_wrong, de_batch
from nmt_chainer.translation.evaluation import (greedy_batch_translate,
                                                #                         convert_idx_to_string,
                                                batch_align,
                                                beam_search_translate,
                                                #                         convert_idx_to_string_with_attn
                                                )

# import visualisation
from nmt_chainer.utilities import bleu_computer
import logging
import codecs
# import h5py
import bokeh.embed


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
        for i in xrange(len(src_w)):
            for j in xrange(len(tgt_w)):
                alignment[i, j] = attn[j][i]
                sum_al[j] += alignment[i, j]
        for j in xrange(len(tgt_w)):
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
            with open(script_output_fn, 'w') as f:
                f.write(script.encode('utf-8'))
            with open(div_output_fn, 'w') as f:
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


def beam_search_all(gpu, encdec, eos_idx, src_data, beam_width, beam_pruning_margin, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, nb_steps,
                    nb_steps_ratio, beam_score_length_normalization, beam_score_length_normalization_strength, post_score_length_normalization, post_score_length_normalization_strength,
                    post_score_coverage_penalty, post_score_coverage_penalty_strength,
                    groundhog,
                    tgt_unk_id, tgt_indexer, force_finish=False,
                    prob_space_combination=False, reverse_encdec=None,
                    use_unfinished_translation_if_none_found=False,
                    replace_unk=False,
                    src=None,
                    dic=None,
                    remove_unk=False,
                    normalize_unicode_unk=False,
                    attempt_to_relocate_unk_source=False):

    log.info("starting beam search translation of %i sentences" % len(src_data))
    if isinstance(encdec, (list, tuple)) and len(encdec) > 1:
        log.info("using ensemble of %i models" % len(encdec))

    with cuda.get_device(gpu):
        translations_gen = beam_search_translate(
            encdec, eos_idx, src_data, beam_width=beam_width, nb_steps=nb_steps,
            gpu=gpu, beam_pruning_margin=beam_pruning_margin,
            beam_score_coverage_penalty=beam_score_coverage_penalty,
            beam_score_coverage_penalty_strength=beam_score_coverage_penalty_strength,
            nb_steps_ratio=nb_steps_ratio,
            need_attention=True,
            beam_score_length_normalization=beam_score_length_normalization,
            beam_score_length_normalization_strength=beam_score_length_normalization_strength,
            post_score_length_normalization=post_score_length_normalization,
            post_score_length_normalization_strength=post_score_length_normalization_strength,
            post_score_coverage_penalty=post_score_coverage_penalty,
            post_score_coverage_penalty_strength=post_score_coverage_penalty_strength,
            groundhog=groundhog, force_finish=force_finish,
            prob_space_combination=prob_space_combination,
            reverse_encdec=reverse_encdec,
            use_unfinished_translation_if_none_found=use_unfinished_translation_if_none_found)

        for num_t, (t, score, attn) in enumerate(translations_gen):
            if num_t % 200 == 0:
                print >>sys.stderr, num_t,
            elif num_t % 40 == 0:
                print >>sys.stderr, "*",
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
                unk_pattern = re.compile("#T_UNK_(\d+)#")
                for idx, word in enumerate(ct.split(' ')):
                    match = unk_pattern.match(word)
                    if (match):
                        unk_mapping.append(match.group(1) + '-' + str(idx))

                if replace_unk:
                    from nmt_chainer.utilities import replace_tgt_unk
                    translated = replace_tgt_unk.replace_unk_from_string(ct, src, dic, remove_unk, normalize_unicode_unk, attempt_to_relocate_unk_source).strip().split(" ")

            yield src_data[num_t], translated, t, score, attn, unk_mapping
        print >>sys.stderr


def translate_to_file_with_beam_search(dest_fn, gpu, encdec, eos_idx, src_data, beam_width, beam_pruning_margin, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, nb_steps,
                                       nb_steps_ratio, beam_score_length_normalization, beam_score_length_normalization_strength, post_score_length_normalization, post_score_length_normalization_strength,
                                       post_score_coverage_penalty, post_score_coverage_penalty_strength,
                                       groundhog,
                                       tgt_unk_id, tgt_indexer, force_finish=False,
                                       prob_space_combination=False, reverse_encdec=None,
                                       generate_attention_html=None, attn_graph_with_sum=True, attn_graph_attribs=None, src_indexer=None, rich_output_filename=None,
                                       use_unfinished_translation_if_none_found=False,
                                       replace_unk=False,
                                       src=None,
                                       dic=None,
                                       remove_unk=False,
                                       normalize_unicode_unk=False,
                                       attempt_to_relocate_unk_source=False):

    log.info("writing translation to %s " % dest_fn)
    out = codecs.open(dest_fn, "w", encoding="utf8")

    translation_iterator = beam_search_all(gpu, encdec, eos_idx, src_data, beam_width, beam_pruning_margin, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, nb_steps,
                                           nb_steps_ratio, beam_score_length_normalization, beam_score_length_normalization_strength, post_score_length_normalization, post_score_length_normalization_strength,
                                           post_score_coverage_penalty, post_score_coverage_penalty_strength,
                                           groundhog,
                                           tgt_unk_id, tgt_indexer, force_finish=force_finish,
                                           prob_space_combination=prob_space_combination, reverse_encdec=reverse_encdec,
                                           use_unfinished_translation_if_none_found=use_unfinished_translation_if_none_found,
                                           replace_unk=replace_unk,
                                           src=src,
                                           dic=dic,
                                           remove_unk=remove_unk,
                                           normalize_unicode_unk=normalize_unicode_unk,
                                           attempt_to_relocate_unk_source=attempt_to_relocate_unk_source)

    attn_vis = None
    if generate_attention_html is not None:
        attn_vis = AttentionVisualizer()
        assert src_indexer is not None

    rich_output = None
    if rich_output_filename is not None:
        rich_output = RichOutputWriter(rich_output_filename)

    for src, translated, t, score, attn, unk_mapping in translation_iterator:
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
        out.write(ct + "\n")

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


def create_encdec(config_eval):
    encdec_list = []
    eos_idx, src_indexer, tgt_indexer = None, None, None

    if config_eval.training_config is not None:
        assert config_eval.trained_model is not None
        encdec, eos_idx, src_indexer, tgt_indexer = create_and_load_encdec_from_files(
            config_eval.training_config, config_eval.trained_model)

        encdec_list.append(encdec)

    if 'load_model_config' in config_eval.process and config_eval.process.load_model_config is not None:
        for config_filename in config_eval.process.load_model_config:
            log.info(
                "loading model and parameters from config %s" %
                config_filename)
            config_training = train_config.load_config_train(config_filename)
            encdec, this_eos_idx, this_src_indexer, this_tgt_indexer = train.create_encdec_and_indexers_from_config_dict(config_training, load_config_model="yes")

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

        for (config_training_fn, trained_model_fn) in zip(config_eval.process.additional_training_config,
                                                          config_eval.process.additional_trained_model):
            this_encdec, this_eos_idx, this_src_indexer, this_tgt_indexer = create_and_load_encdec_from_files(
                config_training_fn, trained_model_fn)

            check_if_vocabulary_info_compatible(this_eos_idx, this_src_indexer, this_tgt_indexer, eos_idx, src_indexer, tgt_indexer)

#             if args.gpu is not None:
#                 this_encdec = this_encdec.to_gpu(args.gpu)

            encdec_list.append(this_encdec)

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

    return encdec_list, eos_idx, src_indexer, tgt_indexer, reverse_encdec


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

    beam_width = config_eval.method.beam_width
    beam_pruning_margin = config_eval.method.beam_pruning_margin
    beam_score_length_normalization = config_eval.method.beam_score_length_normalization
    beam_score_length_normalization_strength = config_eval.method.beam_score_length_normalization_strength
    beam_score_coverage_penalty = config_eval.beam_score_coverage_penalty
    beam_score_coverage_penalty_strength = config_eval.beam_score_coverage_penalty_strength
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

    if dest_fn is not None:
        save_eval_config_fn = dest_fn + ".eval.config.json"
        log.info("Saving eval config to %s" % save_eval_config_fn)
        config_eval.save_to(save_eval_config_fn)
#     json.dump(config_eval, open(save_eval_config_fn, "w"), indent=2, separators=(',', ': '))

    encdec_list, eos_idx, src_indexer, tgt_indexer, reverse_encdec = create_encdec(config_eval)

    if src_fn is not None:
        log.info("opening source file %s" % src_fn)
        src_data, stats_src_pp = build_dataset_one_side_pp(src_fn, src_pp=src_indexer,
                                                           max_nb_ex=max_nb_ex)
        log.info("src data stats:\n%s", stats_src_pp.make_report())

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

    if mode == "translate":
        log.info("writing translation of to %s" % dest_fn)
        with cuda.get_device(gpu):
            assert len(encdec_list) == 1
            translations = greedy_batch_translate(
                encdec_list[0], eos_idx, src_data, batch_size=mb_size, gpu=gpu, nb_steps=nb_steps)
        out = codecs.open(dest_fn, "w", encoding="utf8")
        for t in translations:
            if t[-1] == eos_idx:
                t = t[:-1]
            ct = tgt_indexer.deconvert(t, unk_tag="#T_UNK#")
#             ct = convert_idx_to_string(t, tgt_voc + ["#T_UNK#"])
            out.write(ct + "\n")

    elif mode == "beam_search":
        if config_eval.process.server is not None:
            from nmt_chainer.translation.server import do_start_server
            do_start_server(config_eval)
        else:
            translate_to_file_with_beam_search(dest_fn, gpu, encdec_list, eos_idx, src_data,
                                               beam_width=beam_width,
                                               beam_pruning_margin=beam_pruning_margin,
                                               beam_score_coverage_penalty=beam_score_coverage_penalty,
                                               beam_score_coverage_penalty_strength=beam_score_coverage_penalty_strength,
                                               nb_steps=nb_steps,
                                               nb_steps_ratio=nb_steps_ratio,
                                               beam_score_length_normalization=beam_score_length_normalization,
                                               beam_score_length_normalization_strength=beam_score_length_normalization_strength,
                                               post_score_length_normalization=post_score_length_normalization,
                                               post_score_length_normalization_strength=post_score_length_normalization_strength,
                                               post_score_coverage_penalty=post_score_coverage_penalty,
                                               post_score_coverage_penalty_strength=post_score_coverage_penalty_strength,
                                               groundhog=groundhog,
                                               tgt_unk_id=tgt_unk_id,
                                               tgt_indexer=tgt_indexer,
                                               force_finish=force_finish,
                                               prob_space_combination=prob_space_combination,
                                               reverse_encdec=reverse_encdec,
                                               generate_attention_html=generate_attention_html,
                                               src_indexer=src_indexer,
                                               rich_output_filename=rich_output_filename,
                                               use_unfinished_translation_if_none_found=True)

    elif mode == "eval_bleu":
        #         assert args.ref is not None
        translate_to_file_with_beam_search(dest_fn, gpu, encdec_list, eos_idx, src_data,
                                           beam_width=beam_width,
                                           beam_pruning_margin=beam_pruning_margin,
                                           beam_score_coverage_penalty=beam_score_coverage_penalty,
                                           beam_score_coverage_penalty_strength=beam_score_coverage_penalty_strength,
                                           nb_steps=nb_steps,
                                           nb_steps_ratio=nb_steps_ratio,
                                           beam_score_length_normalization=beam_score_length_normalization,
                                           beam_score_length_normalization_strength=beam_score_length_normalization_strength,
                                           post_score_length_normalization=post_score_length_normalization,
                                           post_score_length_normalization_strength=post_score_length_normalization_strength,
                                           post_score_coverage_penalty=post_score_coverage_penalty,
                                           post_score_coverage_penalty_strength=post_score_coverage_penalty_strength,
                                           groundhog=groundhog,
                                           tgt_unk_id=tgt_unk_id,
                                           tgt_indexer=tgt_indexer,
                                           force_finish=force_finish,
                                           prob_space_combination=prob_space_combination,
                                           reverse_encdec=reverse_encdec,
                                           generate_attention_html=generate_attention_html,
                                           src_indexer=src_indexer,
                                           rich_output_filename=rich_output_filename,
                                           use_unfinished_translation_if_none_found=True)

        if ref is not None:
            bc = bleu_computer.get_bc_from_files(ref, dest_fn)
            print "bleu before unk replace:", bc
        else:
            print "bleu before unk replace: No Ref Provided"

        from nmt_chainer.utilities import replace_tgt_unk
        replace_tgt_unk.replace_unk(dest_fn, src_fn, dest_fn + ".unk_replaced", dic, remove_unk,
                                    normalize_unicode_unk,
                                    attempt_to_relocate_unk_source)

        if ref is not None:
            bc = bleu_computer.get_bc_from_files(ref, dest_fn + ".unk_replaced")
            print "bleu after unk replace:", bc
        else:
            print "bleu before unk replace: No Ref Provided"

    elif mode == "translate_attn":
        log.info("writing translation + attention as html to %s" % dest_fn)
        with cuda.get_device(gpu):
            assert len(encdec_list) == 1
            translations, attn_all = greedy_batch_translate(
                encdec_list[0], eos_idx, src_data, batch_size=mb_size, gpu=gpu,
                get_attention=True, nb_steps=nb_steps)
#         tgt_voc_with_unk = tgt_voc + ["#T_UNK#"]
#         src_voc_with_unk = src_voc + ["#S_UNK#"]
        assert len(translations) == len(src_data)
        assert len(attn_all) == len(src_data)
        attn_vis = AttentionVisualizer()
        for num_t in xrange(len(src_data)):
            src_idx_list = src_data[num_t]
            tgt_idx_list = translations[num_t][:-1]
            attn = attn_all[num_t]
#             assert len(attn) == len(tgt_idx_list)

            src_w = src_indexer.deconvert_swallow(src_idx_list, unk_tag="#S_UNK#") + ["SUM_ATTN"]
            tgt_w = tgt_indexer.deconvert_swallow(tgt_idx_list, unk_tag="#T_UNK#")
#             src_w = [src_voc_with_unk[idx] for idx in src_idx_list] + ["SUM_ATTN"]
#             tgt_w = [tgt_voc_with_unk[idx] for idx in tgt_idx_list]
#             for j in xrange(len(tgt_idx_list)):
#                 tgt_idx_list.append(tgt_voc_with_unk[t_and_attn[j][0]])
#
    #         print [src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list

            attn_vis.add_plot(src_w, tgt_w, attn)

        attn_vis.make_plot(dest_fn)

    elif mode == "align":
        import nmt_chainer.utilities.visualisation as visualisation
        assert tgt_data is not None
        assert len(tgt_data) == len(src_data)
        log.info("writing alignment as html to %s" % dest_fn)
        with cuda.get_device(gpu):
            assert len(encdec_list) == 1
            loss, attn_all = batch_align(
                encdec_list[0], eos_idx, zip(src_data, tgt_data), batch_size=mb_size, gpu=gpu)
#         tgt_voc_with_unk = tgt_voc + ["#T_UNK#"]
#         src_voc_with_unk = src_voc + ["#S_UNK#"]

        assert len(attn_all) == len(src_data)
        plots_list = []
        for num_t in xrange(len(src_data)):
            src_idx_list = src_data[num_t]
            tgt_idx_list = tgt_data[num_t]
            attn = attn_all[num_t]
#             assert len(attn) == len(tgt_idx_list)

            alignment = np.zeros((len(src_idx_list) + 1, len(tgt_idx_list)))
            sum_al = [0] * len(tgt_idx_list)
            for i in xrange(len(src_idx_list)):
                for j in xrange(len(tgt_idx_list)):
                    alignment[i, j] = attn[j][i]
                    sum_al[j] += alignment[i, j]
            for j in xrange(len(tgt_idx_list)):
                alignment[len(src_idx_list), j] = sum_al[j]

            src_w = src_indexer.deconvert_swallow(src_idx_list, unk_tag="#S_UNK#") + ["SUM_ATTN"]
            tgt_w = tgt_indexer.deconvert_swallow(tgt_idx_list, unk_tag="#T_UNK#")
#             src_w = [src_voc_with_unk[idx] for idx in src_idx_list] + ["SUM_ATTN"]
#             tgt_w = [tgt_voc_with_unk[idx] for idx in tgt_idx_list]
#             for j in xrange(len(tgt_idx_list)):
#                 tgt_idx_list.append(tgt_voc_with_unk[t_and_attn[j][0]])
#
    #         print [src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list
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
#             print x, attn


#         out.write(convert_idx_to_string([x for x, attn in t], tgt_voc + ["#T_UNK#"]) + "\n")

    elif mode == "score_nbest":
        log.info("opening nbest file %s" % nbest_to_rescore)
        nbest_f = codecs.open(nbest_to_rescore, encoding="utf8")
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
        for num in xrange(len(nbest_converted)):
            if num % 200 == 0:
                print >>sys.stderr, num,
            elif num % 50 == 0:
                print >>sys.stderr, "*",

            res.append([])
            src, tgt_list = src_data[num], nbest_converted[num]
            src_batch, src_mask = utils.make_batch_src([src], gpu=gpu, volatile="on")

            assert len(encdec_list) == 1
            scorer = encdec_list[0].nbest_scorer(src_batch, src_mask)

            nb_batches = (len(tgt_list) + mb_size - 1) / mb_size
            for num_batch in xrange(nb_batches):
                tgt_batch, arg_sort = utils.make_batch_tgt(tgt_list[num_batch * nb_batches: (num_batch + 1) * nb_batches],
                                                           eos_idx=eos_idx, gpu=gpu, volatile="on", need_arg_sort=True)
                scores, attn = scorer(tgt_batch)
                scores, _ = scores
                scores = scores.data

                assert len(arg_sort) == len(scores)
                de_sorted_scores = [None] * len(scores)
                for xpos in xrange(len(arg_sort)):
                    original_pos = arg_sort[xpos]
                    de_sorted_scores[original_pos] = scores[xpos]
                res[-1] += de_sorted_scores
        print >>sys.stderr
        log.info("writing scores to %s" % dest_fn)
        out = codecs.open(dest_fn, "w", encoding="utf8")
        for num in xrange(len(res)):
            for score in res[num]:
                out.write("%i %f\n" % (num, score))
