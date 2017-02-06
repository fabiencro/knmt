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
import nmt_chainer.models.encoder_decoder
from make_data import Indexer, build_dataset_one_side
import make_data
import nmt_chainer.training_module.train as train

# from utils import make_batch_src, make_batch_src_tgt, minibatch_provider, compute_bleu_with_unk_as_wrong, de_batch
from nmt_chainer.evaluation import (greedy_batch_translate, 
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
from nmt_chainer.utilities import argument_parsing_tools

import nmt_chainer.models.rnn_cells as rnn_cells

logging.basicConfig()
log = logging.getLogger("rnns:eval")
log.setLevel(logging.INFO)

class CommandLineValuesException(Exception):
    pass

class AttentionVisualizer(object):
    def __init__(self):
        self.plots_list = []
        
    def add_plot(self, src_w, tgt_w, attn):
        from nmt_chainer.utilities import visualisation
        alignment = np.zeros((len(src_w) + 1, len(tgt_w)))
        sum_al =[0] * len(tgt_w)
        for i in xrange(len(src_w)):
            for j in xrange(len(tgt_w)):
                alignment[i,j] = attn[j][i]
                sum_al[j] += alignment[i,j]
        for j in xrange(len(tgt_w)):        
            alignment[len(src_w), j] =  sum_al[j]
            
        p1 = visualisation.make_alignment_figure(
                            src_w + ["SUM_ATTN"], tgt_w, alignment)
        
        self.plots_list.append(p1)
            
    def make_plot(self, output_file):
        from nmt_chainer.utilities import visualisation
        log.info("writing attention to %s"% output_file)
        p_all = visualisation.Column(*self.plots_list)
        visualisation.output_file(output_file)
        visualisation.show(p_all)

class RichOutputWriter(object):
    def __init__(self, filename):
        log.info("writing JSON translation infos to %s"% filename)
        self.filename = filename
        self.output = open(filename, "w")
        self.no_entry_yet = True
        self.output.write("[\n")
        
    def add_info(self, src, translated, t, score, attn):
        if not self.no_entry_yet:
            self.output.write(",\n")
        else:
            self.no_entry_yet = False
        self.output.write("{\"tr\": ")
        self.output.write(json.dumps(translated))
        self.output.write(",\n\"attn\": ")
        self.output.write(json.dumps([[float(a) for a in a_list] for a_list in attn]))
        self.output.write("}")
            
    def finish(self):
        self.output.write("\n]")
        self.output.close()
        log.info("done writing JSON translation infos to %s"% self.filename)
          


def beam_search_all(gpu, encdec, eos_idx, src_data, beam_width, beam_pruning_margin, nb_steps,
       nb_steps_ratio, post_score_length_normalization, length_normalization_strength, 
       groundhog,
       tgt_unk_id, tgt_indexer, force_finish = False,
       prob_space_combination = False, reverse_encdec = None,
       use_unfinished_translation_if_none_found = False):
    
    log.info("starting beam search translation of %i sentences"% len(src_data))
    if isinstance(encdec, (list, tuple)) and len(encdec) > 1:
        log.info("using ensemble of %i models"%len(encdec))
        
    with cuda.get_device(gpu):
        translations_gen = beam_search_translate(
                    encdec, eos_idx, src_data, beam_width = beam_width, nb_steps = nb_steps, 
                                    gpu = gpu, beam_pruning_margin = beam_pruning_margin, nb_steps_ratio = nb_steps_ratio,
                                    need_attention = True, post_score_length_normalization = post_score_length_normalization,
                                    length_normalization_strength = length_normalization_strength,
                                    groundhog = groundhog, force_finish = force_finish,
                                    prob_space_combination = prob_space_combination,
                                    reverse_encdec = reverse_encdec,
                                    use_unfinished_translation_if_none_found = use_unfinished_translation_if_none_found)
        
        for num_t, (t, score, attn) in enumerate(translations_gen):
            if num_t %200 == 0:
                print >>sys.stderr, num_t,
            elif num_t %40 == 0:
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
                    return unk_pattern%src_pos
            elif tgt_unk_id == "id":
                def unk_replacer(num_pos, unk_id):
                    unk_pattern = "#T_UNK_%i#"
                    return unk_pattern%unk_id         
            else:
                assert False
            
            translated = tgt_indexer.deconvert(t, unk_tag = unk_replacer)
            
            yield src_data[num_t], translated, t, score, attn 
        print >>sys.stderr

def translate_to_file_with_beam_search(dest_fn, gpu, encdec, eos_idx, src_data, beam_width, beam_pruning_margin, nb_steps, 
       nb_steps_ratio, post_score_length_normalization, length_normalization_strength, 
       groundhog,
       tgt_unk_id, tgt_indexer, force_finish = False,
       prob_space_combination = False, reverse_encdec = None, 
       generate_attention_html = None, src_indexer = None, rich_output_filename = None,
       use_unfinished_translation_if_none_found = False):
    
    log.info("writing translation to %s "% dest_fn)
    out = codecs.open(dest_fn, "w", encoding = "utf8")
    
    translation_iterator = beam_search_all(gpu, encdec, eos_idx, src_data, beam_width, beam_pruning_margin, nb_steps, 
       nb_steps_ratio, post_score_length_normalization, length_normalization_strength, 
       groundhog,
       tgt_unk_id, tgt_indexer, force_finish = force_finish,
       prob_space_combination = prob_space_combination, reverse_encdec = reverse_encdec,
       use_unfinished_translation_if_none_found = use_unfinished_translation_if_none_found)
    
    attn_vis = None
    if generate_attention_html is not None:
        attn_vis = AttentionVisualizer()
        assert src_indexer is not None
        
    rich_output = None
    if rich_output_filename is not None:
        rich_output = RichOutputWriter(rich_output_filename)
        
    for src, translated, t, score, attn in translation_iterator:
        if rich_output is not None:
            rich_output.add_info(src, translated, t, score, attn)
        if attn_vis is not None:
            attn_vis.add_plot(src_indexer.deconvert(src), translated, attn)
        ct = " ".join(translated)
        out.write(ct + "\n")
        
    if rich_output is not None:
        rich_output.finish()
    
    if attn_vis is not None:
        attn_vis.make_plot(generate_attention_html)
    
def create_and_load_encdec_from_files(config_training_fn, trained_model):
    log.info("loading model config from %s" % config_training_fn)
    
    config_training = train.load_config_train(config_training_fn)
    encdec, eos_idx, src_indexer, tgt_indexer = train.create_encdec_and_indexers_from_config_dict(config_training)
        
    log.info("loading model from %s" % trained_model)
    serializers.load_npz(trained_model, encdec)
    
    return encdec, eos_idx, src_indexer, tgt_indexer
    
_CONFIG_SECTION_TO_DESCRIPTION = {"method": "Translation Method",
                                  "output": "Output Options",
                                  "process": "Translation Process Options"}
    
def define_parser(parser):
    parser.add_argument("training_config", nargs = "?", help = "prefix of the trained model",
                        action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    parser.add_argument("trained_model", nargs = "?", help = "prefix of the trained model",
                        action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    parser.add_argument("src_fn", nargs = "?", help = "source text",
                        action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    parser.add_argument("dest_fn", nargs = "?", help = "destination file",
                        action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    
    translation_method_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["method"])
    translation_method_group.add_argument("--mode", default = "translate", 
                        choices = ["translate", "align", "translate_attn", "beam_search", "eval_bleu",
                                   "score_nbest"], help = "target text")
    translation_method_group.add_argument("--beam_width", type = int, default= 20, help = "beam width")
    translation_method_group.add_argument("--beam_pruning_margin", type = float, default= None, help = "beam pruning margin")
    translation_method_group.add_argument("--nb_steps", type = int, default= 50, help = "nb_steps used in generation")
    translation_method_group.add_argument("--nb_steps_ratio", type = float, help = "nb_steps used in generation as a ratio of input length")
#     translation_method_group.add_argument("--beam_opt", default = False, action = "store_true")
    translation_method_group.add_argument("--groundhog", default = False, action = "store_true")
    translation_method_group.add_argument("--force_finish", default = False, action = "store_true")
    translation_method_group.add_argument("--post_score_length_normalization", choices = ['none', 'simple', 'google'], default = 'simple')
    translation_method_group.add_argument("--length_normalization_strength", type = float, default = 0.2)
    translation_method_group.add_argument("--prob_space_combination", default = False, action = "store_true")
    translation_method_group.add_argument("--additional_training_config", nargs = "*", help = "prefix of the trained model")
    translation_method_group.add_argument("--additional_trained_model", nargs = "*", help = "prefix of the trained model")
    translation_method_group.add_argument("--reverse_training_config", help = "prefix of the trained model")
    translation_method_group.add_argument("--reverse_trained_model", help = "prefix of the trained model")
    
    
    output_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["output"])
    output_group.add_argument("--tgt_fn", help = "target text")
    output_group.add_argument("--nbest_to_rescore", help = "nbest list in moses format")
    output_group.add_argument("--ref", help = "target text")
    output_group.add_argument("--tgt_unk_id", choices = ["align", "id"], default = "align")
    output_group.add_argument("--generate_attention_html", help = "generate a html file with attention information")
    output_group.add_argument("--rich_output_filename", help = "generate a JSON file with attention information")
    # arguments for unk replace
    output_group.add_argument("--dic")
    output_group.add_argument("--remove_unk", default = False, action = "store_true")
    output_group.add_argument("--normalize_unicode_unk", default = False, action = "store_true")
    output_group.add_argument("--attempt_to_relocate_unk_source", default = False, action = "store_true")
    
    management_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["process"])
    management_group.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
    management_group.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
    management_group.add_argument("--mb_size", type = int, default= 80, help = "Minibatch size")
    management_group.add_argument("--nb_batch_to_sort", type = int, default= 20, help = "Sort this many batches by size.")
    management_group.add_argument("--load_model_config", nargs = "+", help = "gives a list of models to be used for translation")
    management_group.add_argument("--src_fn", nargs = "?", help = "source text",
                                  action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    management_group.add_argument("--dest_fn", nargs = "?", help = "destination file",
                                  action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone)
    
#     management_group.add_argument("--config", help = "load eval config file")
    
def get_parse_option_orderer():
    description_to_config_section = dict( (v, k) for (k,v) in _CONFIG_SECTION_TO_DESCRIPTION.iteritems())
    por = argument_parsing_tools.ParseOptionRecorder(group_title_to_section = description_to_config_section,
                                                     ignore_positional_arguments = set(["src_fn", "dest_fn"]))
    define_parser(por)
    return por
    
def command_line(arguments = None):
    
    import argparse
    parser = argparse.ArgumentParser(description= "Use a RNNSearch model", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    define_parser(parser)
    
    args = parser.parse_args(args = arguments)
    
    do_eval(args)
    
    
def check_if_vocabulary_info_compatible(this_eos_idx, this_src_indexer, this_tgt_indexer, eos_idx, src_indexer, tgt_indexer):
    if eos_idx != this_eos_idx:
        raise Exception("incompatible models")
    if len(src_indexer) != len(this_src_indexer):
        raise Exception("incompatible models")
    if len(tgt_indexer) != len(this_tgt_indexer):
        raise Exception("incompatible models")
    
def make_config_eval(args):
    parse_option_orderer = get_parse_option_orderer()
    config_eval = parse_option_orderer.convert_args_to_ordered_dict(args)
    config_eval.add_metadata_infos(version_num = 1)
    config_eval.set_readonly()
    
    if config_eval.process.src_fn is None or config_eval.process.dest_fn is None:
        raise CommandLineValuesException("src_fn and dest_fn need to be set either on the command line or in a config file")
    
    return config_eval

def do_eval(args):
    config_eval = make_config_eval(args)
    
    save_eval_config_fn = config_eval.process.dest_fn + ".eval.config.json"
    log.info("Saving eval config to %s" % save_eval_config_fn)
    config_eval.save_to(save_eval_config_fn)
#     json.dump(config_eval, open(save_eval_config_fn, "w"), indent=2, separators=(',', ': '))
    
    encdec_list = []
    eos_idx, src_indexer, tgt_indexer = None, None, None
    
    if args.training_config is not None:
        if args.trained_model is None:
            raise CommandLineValuesException("If specifying a model via the training_config argument, you also need to specify the trained_model argument")
        
        encdec, eos_idx, src_indexer, tgt_indexer = create_and_load_encdec_from_files(
                                args.training_config, args.trained_model)
        
        encdec_list.append(encdec)
    
    if args.load_model_config is not None:
        for config_filename in args.load_model_config:
            log.info("loading model and parameters from config %s" % config_filename)
            config_training = train.load_config_train(config_filename)
            encdec, this_eos_idx, this_src_indexer, this_tgt_indexer = train.create_encdec_and_indexers_from_config_dict(config_training, load_config_model= "yes")
            
            if eos_idx is None:
                assert len(encdec_list) == 0
                assert src_indexer is None
                assert tgt_indexer is None
                eos_idx, src_indexer, tgt_indexer = this_eos_idx, this_src_indexer, this_tgt_indexer
            else:
                check_if_vocabulary_info_compatible(this_eos_idx, this_src_indexer, this_tgt_indexer, eos_idx, src_indexer, tgt_indexer)
            
            encdec_list.append(encdec)
            
    if len(encdec_list) == 0:
        raise CommandLineValuesException("You need to specify either the training_config positional argument, or the load_model_config option, or both")
    
            
    if args.additional_training_config is not None:
        assert len(args.additional_training_config) == len(args.additional_trained_model)
    
        
        for (config_training_fn, trained_model_fn) in zip(args.additional_training_config, 
                                                          args.additional_trained_model):
            this_encdec, this_eos_idx, this_src_indexer, this_tgt_indexer = create_and_load_encdec_from_files(
                            config_training_fn, trained_model_fn)
        
            check_if_vocabulary_info_compatible(this_eos_idx, this_src_indexer, this_tgt_indexer, eos_idx, src_indexer, tgt_indexer)
                              
#             if args.gpu is not None:
#                 this_encdec = this_encdec.to_gpu(args.gpu)
            
            encdec_list.append(this_encdec)
            
            
    if args.gpu is not None:
        encdec_list = [encdec.to_gpu(args.gpu) for encdec in encdec_list]
            
            
    if args.reverse_training_config is not None:
        reverse_encdec, reverse_eos_idx, reverse_src_indexer, reverse_tgt_indexer = create_and_load_encdec_from_files(
                            args.reverse_training_config, args.reverse_trained_model)
        
        if eos_idx != reverse_eos_idx:
            raise Exception("incompatible models")
            
        if len(src_indexer) != len(reverse_src_indexer):
            raise Exception("incompatible models")
          
        if len(tgt_indexer) != len(reverse_tgt_indexer):
            raise Exception("incompatible models")
                          
        if args.gpu is not None:
            reverse_encdec = reverse_encdec.to_gpu(args.gpu)
    else:
        reverse_encdec = None    
        
    log.info("opening source file %s" % args.src_fn)
    src_data, dic_src, make_data_infos = build_dataset_one_side(args.src_fn, 
                                    src_voc_limit = None, max_nb_ex = args.max_nb_ex, dic_src = src_indexer)
    log.info("%i sentences loaded" % make_data_infos.nb_ex)
    log.info("#tokens src: %i   of which %i (%f%%) are unknown"%(make_data_infos.total_token, 
                                                                 make_data_infos.total_count_unk, 
                                                                 float(make_data_infos.total_count_unk * 100) / 
                                                                    make_data_infos.total_token))
    assert dic_src == src_indexer
    
    tgt_data = None
    if args.tgt_fn is not None:
        log.info("opening target file %s" % args.tgt_fn)
        tgt_data, dic_tgt, make_data_infos = build_dataset_one_side(args.tgt_fn, 
                                    src_voc_limit = None, max_nb_ex = args.max_nb_ex, dic_src = tgt_indexer)
        log.info("%i sentences loaded"%make_data_infos.nb_ex)
        log.info("#tokens src: %i   of which %i (%f%%) are unknown"%(make_data_infos.total_token, 
                                                                 make_data_infos.total_count_unk, 
                                                                 float(make_data_infos.total_count_unk * 100) / 
                                                                    make_data_infos.total_token))
        assert dic_tgt == tgt_indexer

    
#     translations = greedy_batch_translate(encdec, eos_idx, src_data, batch_size = args.mb_size, gpu = args.gpu)
    
    if args.mode == "translate":
        log.info("writing translation of to %s"% args.dest_fn)
        with cuda.get_device(args.gpu):
            translations = greedy_batch_translate(
                                        encdec, eos_idx, src_data, batch_size = args.mb_size, gpu = args.gpu, nb_steps = args.nb_steps)
        out = codecs.open(args.dest_fn, "w", encoding = "utf8")
        for t in translations:
            if t[-1] == eos_idx:
                t = t[:-1]
            ct = " ".join(tgt_indexer.deconvert(t, unk_tag = "#T_UNK#"))
#             ct = convert_idx_to_string(t, tgt_voc + ["#T_UNK#"])
            out.write(ct + "\n")

    elif args.mode == "beam_search":
        translate_to_file_with_beam_search(args.dest_fn, args.gpu, encdec_list, eos_idx, src_data, args.beam_width, args.beam_pruning_margin,
                                           args.nb_steps,
                                           args.nb_steps_ratio, args.post_score_length_normalization, args.length_normalization_strength,
                                           args.groundhog,
                                           args.tgt_unk_id, tgt_indexer, force_finish = args.force_finish,
                                           prob_space_combination = args.prob_space_combination,
                                           reverse_encdec = reverse_encdec,
                                           generate_attention_html = args.generate_attention_html,
                                           src_indexer = src_indexer,
                                           rich_output_filename = args.rich_output_filename,
                                           use_unfinished_translation_if_none_found = True)
            
    elif args.mode == "eval_bleu":
#         assert args.ref is not None
        translate_to_file_with_beam_search(args.dest_fn, args.gpu, encdec_list, eos_idx, src_data, args.beam_width, args.beam_pruning_margin,
                                           args.nb_steps,
                                           args.nb_steps_ratio, args.post_score_length_normalization, args.length_normalization_strength,
                                           args.groundhog,
                                           args.tgt_unk_id, tgt_indexer, force_finish = args.force_finish,
                                           prob_space_combination = args.prob_space_combination,
                                           reverse_encdec = reverse_encdec,
                                           generate_attention_html = args.generate_attention_html,
                                           src_indexer = src_indexer,
                                           rich_output_filename = args.rich_output_filename,
                                           use_unfinished_translation_if_none_found = True)
        
        if args.ref is not None:
            bc = bleu_computer.get_bc_from_files(args.ref, args.dest_fn)
            print "bleu before unk replace:", bc
        else:
            print "bleu before unk replace: No Ref Provided"
        
        from nmt_chainer.utilities import replace_tgt_unk
        replace_tgt_unk.replace_unk(args.dest_fn, args.src_fn, args.dest_fn + ".unk_replaced", args.dic, args.remove_unk, 
                args.normalize_unicode_unk,
                args.attempt_to_relocate_unk_source)
         
        if args.ref is not None: 
            bc = bleu_computer.get_bc_from_files(args.ref, args.dest_fn + ".unk_replaced")
            print "bleu after unk replace:", bc 
        else:
            print "bleu before unk replace: No Ref Provided"   
            
    elif args.mode == "translate_attn":
        log.info("writing translation + attention as html to %s"% args.dest_fn)
        with cuda.get_device(args.gpu):
            translations, attn_all = greedy_batch_translate(
                                        encdec, eos_idx, src_data, batch_size = args.mb_size, gpu = args.gpu,
                                        get_attention = True, nb_steps = args.nb_steps)
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
                
            src_w = src_indexer.deconvert(src_idx_list, unk_tag = "#S_UNK#") + ["SUM_ATTN"]
            tgt_w = tgt_indexer.deconvert(tgt_idx_list, unk_tag = "#T_UNK#")
#             src_w = [src_voc_with_unk[idx] for idx in src_idx_list] + ["SUM_ATTN"]
#             tgt_w = [tgt_voc_with_unk[idx] for idx in tgt_idx_list]
#             for j in xrange(len(tgt_idx_list)):
#                 tgt_idx_list.append(tgt_voc_with_unk[t_and_attn[j][0]])
#             
    #         print [src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list
        
            attn_vis.add_plot(src_w, tgt_w, attn)
        
        attn_vis.make_plot(args.dest_fn)
        
    elif args.mode == "align":
        assert tgt_data is not None
        assert len(tgt_data) == len(src_data)
        log.info("writing alignment as html to %s"% args.dest_fn)
        with cuda.get_device(args.gpu):
            loss, attn_all = batch_align(
                                        encdec, eos_idx, zip(src_data, tgt_data), batch_size = args.mb_size, gpu = args.gpu)
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
            sum_al =[0] * len(tgt_idx_list)
            for i in xrange(len(src_idx_list)):
                for j in xrange(len(tgt_idx_list)):
                    alignment[i,j] = attn[j][i]
                    sum_al[j] += alignment[i,j]
            for j in xrange(len(tgt_idx_list)):        
                alignment[len(src_idx_list), j] =  sum_al[j]
                
            src_w = src_indexer.deconvert(src_idx_list, unk_tag = "#S_UNK#") + ["SUM_ATTN"]
            tgt_w = tgt_indexer.deconvert(tgt_idx_list, unk_tag = "#T_UNK#")
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
        visualisation.output_file(args.dest_fn)
        visualisation.show(p_all)
#     for t in translations_with_attn:
#         for x, attn in t:
#             print x, attn
            
            
#         out.write(convert_idx_to_string([x for x, attn in t], tgt_voc + ["#T_UNK#"]) + "\n")

    elif args.mode == "score_nbest":
        log.info("opening nbest file %s" % args.nbest_to_rescore)
        nbest_f = codecs.open(args.nbest_to_rescore, encoding = "utf8")
        nbest_list = [[]]
        for line in nbest_f:
            line = line.strip().split("|||")
            num_src = int(line[0].strip())
            if num_src >= len(nbest_list):
                assert num_src == len(nbest_list)
                if args.max_nb_ex is not None and num_src >= args.max_nb_ex:
                    break
                nbest_list.append([])
            else:
                assert num_src == len(nbest_list) -1
            sentence = line[1].strip()
            nbest_list[-1].append(sentence.split(" "))
        
        log.info("found nbest lists for %i source sentences"%len(nbest_list))
        nbest_converted, make_data_infos = make_data.build_dataset_for_nbest_list_scoring(tgt_indexer, nbest_list)
        log.info("total %i sentences loaded"%make_data_infos.nb_ex)
        log.info("#tokens src: %i   of which %i (%f%%) are unknown"%(make_data_infos.total_token, 
                                                                 make_data_infos.total_count_unk, 
                                                                 float(make_data_infos.total_count_unk * 100) / 
                                                                    make_data_infos.total_token))
        if len(nbest_list) != len(src_data[:args.max_nb_ex]):
            log.warn("mismatch in lengths nbest vs src : %i != %i"%(len(nbest_list), len(src_data[:args.max_nb_ex])))
            assert len(nbest_list) == len(src_data[:args.max_nb_ex])
        
        
        log.info("starting scoring")
        from nmt_chainer.utilities import utils
        res = []
        for num in xrange(len(nbest_converted)):
            if num%200 == 0:
                print  >>sys.stderr, num,
            elif num %50 == 0:
                print  >>sys.stderr, "*",
                
            res.append([])
            src, tgt_list = src_data[num], nbest_converted[num]
            src_batch, src_mask = utils.make_batch_src([src], gpu = args.gpu, volatile = "on")
            
            scorer = encdec.nbest_scorer(src_batch, src_mask)
            
            nb_batches = (len(tgt_list) + args.mb_size -1)/ args.mb_size
            for num_batch in xrange(nb_batches):
                tgt_batch, arg_sort = utils.make_batch_tgt(tgt_list[num_batch * nb_batches : (num_batch + 1) * nb_batches],
                                eos_idx = eos_idx, gpu =  args.gpu, volatile = "on", need_arg_sort = True)
                scores, attn = scorer(tgt_batch)
                scores, _ = scores
                scores = scores.data
                
                assert len(arg_sort) == len(scores)
                de_sorted_scores = [None] * len(scores)
                for xpos in xrange(len(arg_sort)):
                    original_pos = arg_sort[xpos]
                    de_sorted_scores[original_pos] = scores[xpos]
                res[-1] += de_sorted_scores
        print  >>sys.stderr
        log.info("writing scores to %s"%args.dest_fn)
        out = codecs.open(args.dest_fn, "w", encoding = "utf8")
        for num in xrange(len(res)):
            for score in res[num]:
                out.write("%i %f\n"%(num, score))
    
if __name__ == '__main__':
    command_line() 
    
    
