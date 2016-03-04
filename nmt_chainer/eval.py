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

import models
from make_data import Indexer, build_dataset_one_side
# from utils import make_batch_src, make_batch_src_tgt, minibatch_provider, compute_bleu_with_unk_as_wrong, de_batch
from evaluation import (greedy_batch_translate, 
                        convert_idx_to_string, 
                        batch_align, 
                        beam_search_translate, 
                        convert_idx_to_string_with_attn)

import visualisation

import logging
import codecs
# import h5py

logging.basicConfig()
log = logging.getLogger("rnns:eval")
log.setLevel(logging.INFO)

def commandline():
    
    import argparse
    parser = argparse.ArgumentParser(description= "Use a RNNSearch model", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("training_config", help = "prefix of the trained model")
    parser.add_argument("trained_model", help = "prefix of the trained model")
    parser.add_argument("src_fn", help = "source text")
    parser.add_argument("dest_fn", help = "destination file")
    
    parser.add_argument("--tgt_fn", help = "target text")
    parser.add_argument("--mode", default = "translate", 
                        choices = ["translate", "align", "translate_attn", "beam_search"], help = "target text")
    parser.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
    
    parser.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
    parser.add_argument("--mb_size", type = int, default= 80, help = "Minibatch size")
    parser.add_argument("--beam_width", type = int, default= 20, help = "beam width")
    parser.add_argument("--nb_steps", type = int, default= 50, help = "nb_steps used in generation")
    parser.add_argument("--nb_steps_ratio", type = float, help = "nb_steps used in generation as a ratio of input length")
    parser.add_argument("--nb_batch_to_sort", type = int, default= 20, help = "Sort this many batches by size.")
    parser.add_argument("--beam_opt", default = False, action = "store_true")
    args = parser.parse_args()
    
    config_training_fn = args.training_config #args.model_prefix + ".train.config"
    
    log.info("loading model config from %s" % config_training_fn)
    config_training = json.load(open(config_training_fn))

    voc_fn = config_training["voc"]
    log.info("loading voc from %s"% voc_fn)
    src_voc, tgt_voc = json.load(open(voc_fn))
    
    Vi = len(src_voc) + 1 # + UNK
    Vo = len(tgt_voc) + 1 # + UNK
    
    print config_training
    
    Ei = config_training["command_line"]["Ei"]
    Hi = config_training["command_line"]["Hi"]
    Eo = config_training["command_line"]["Eo"]
    Ho = config_training["command_line"]["Ho"]
    Ha = config_training["command_line"]["Ha"]
    Hl = config_training["command_line"]["Hl"]
    
    eos_idx = Vo
    encdec = models.EncoderDecoder(Vi, Ei, Hi, Vo + 1, Eo, Ho, Ha, Hl)
    
    log.info("loading model from %s" % args.trained_model)
    serializers.load_npz(args.trained_model, encdec)
    
    if args.gpu is not None:
        encdec = encdec.to_gpu(args.gpu)
        
        
    src_indexer = Indexer.make_from_list(src_voc)
    
    
    log.info("opening source file %s" % args.src_fn)
    src_data, dic_src, total_count_unk_src, total_token_src, num_ex = build_dataset_one_side(args.src_fn, 
                                    src_voc_limit = None, max_nb_ex = args.max_nb_ex, dic_src = src_indexer)
    
    log.info("%i sentences loaded"%num_ex)
    log.info("#tokens src: %i   of which %i (%f%%) are unknown"%(total_token_src, total_count_unk_src, 
                                                                 float(total_count_unk_src * 100) / total_token_src))
    assert dic_src == src_indexer
    
    tgt_data = None
    if args.tgt_fn is not None:
        log.info("opening target file %s" % args.tgt_fn)
        tgt_indexer = Indexer.make_from_list(tgt_voc)
        tgt_data, dic_tgt, total_count_unk_tgt, total_token_tgt, num_ex = build_dataset_one_side(args.tgt_fn, 
                                    src_voc_limit = None, max_nb_ex = args.max_nb_ex, dic_src = tgt_indexer)
        log.info("%i sentences loaded"%num_ex)
        log.info("#tokens tgt: %i   of which %i (%f%%) are unknown"%(total_token_tgt, total_count_unk_tgt, 
                                                                     float(total_count_unk_tgt * 100) / total_token_tgt))
        assert dic_tgt == tgt_indexer

    
#     translations = greedy_batch_translate(encdec, eos_idx, src_data, batch_size = args.mb_size, gpu = args.gpu)
    
    if args.mode == "translate":
        log.info("writing translation of to %s"% args.dest_fn)
        with cuda.cupy.cuda.Device(args.gpu):
            translations = greedy_batch_translate(
                                        encdec, eos_idx, src_data, batch_size = args.mb_size, gpu = args.gpu, nb_steps = args.nb_steps)
        out = codecs.open(args.dest_fn, "w", encoding = "utf8")
        for t in translations:
            ct = convert_idx_to_string(t[:-1], tgt_voc + ["#T_UNK#"])
            out.write(ct + "\n")

    elif args.mode == "beam_search":
        log.info("writing translation of to %s"% args.dest_fn)
        out = codecs.open(args.dest_fn, "w", encoding = "utf8")
        with cuda.cupy.cuda.Device(args.gpu):
            translations_gen = beam_search_translate(
                        encdec, eos_idx, src_data, beam_width = args.beam_width, nb_steps = args.nb_steps, 
                                                 gpu = args.gpu, beam_opt = args.beam_opt, nb_steps_ratio = args.nb_steps_ratio,
                                                 need_attention = True)
            
    #         for num_t in range(len(translations)):
    #             print num_t
    #             for t, score in translations[num_t]:
    #                 ct = convert_idx_to_string(t[:-1], tgt_voc + ["#T_UNK#"])
    #                 print ct, score
    #                 out.write(ct + "\n")
            for t, score, attn in translations_gen:
#                 t, score = bests[1]
#                 ct = convert_idx_to_string(t, tgt_voc + ["#T_UNK#"])
                ct = convert_idx_to_string_with_attn(t, tgt_voc, attn, unk_idx = len(tgt_voc))
#                 print convert_idx_to_string(bests[0][0], tgt_voc + ["#T_UNK#"]) , bests[0][1]
#                 print convert_idx_to_string(bests[1][0], tgt_voc + ["#T_UNK#"]), bests[1][1], bests[1][1] / len(bests[1][0])
                out.write(ct + "\n")
            
    elif args.mode == "translate_attn":
        log.info("writing translation + attention as html to %s"% args.dest_fn)
        with cuda.cupy.cuda.Device(args.gpu):
            translations, attn_all = greedy_batch_translate(
                                        encdec, eos_idx, src_data, batch_size = args.mb_size, gpu = args.gpu,
                                        get_attention = True, nb_steps = args.nb_steps)
        tgt_voc_with_unk = tgt_voc + ["#T_UNK#"]
        src_voc_with_unk = src_voc + ["#S_UNK#"]
        assert len(translations) == len(src_data)
        assert len(attn_all) == len(src_data)
        plots_list = []
        for num_t in xrange(len(src_data)):
            src_idx_list = src_data[num_t]
            tgt_idx_list = translations[num_t][:-1]
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
            src_w = [src_voc_with_unk[idx] for idx in src_idx_list] + ["SUM_ATTN"]
            tgt_w = [tgt_voc_with_unk[idx] for idx in tgt_idx_list]
#             for j in xrange(len(tgt_idx_list)):
#                 tgt_idx_list.append(tgt_voc_with_unk[t_and_attn[j][0]])
#             
    #         print [src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list
            p1 = visualisation.make_alignment_figure(
                            src_w, tgt_w, alignment)
#             p2 = visualisation.make_alignment_figure(
#                             [src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list, alignment)
            plots_list.append(p1)
        p_all = visualisation.vplot(*plots_list)
        visualisation.output_file(args.dest_fn)
        visualisation.show(p_all)
        
    elif args.mode == "align":
        assert tgt_data is not None
        assert len(tgt_data) == len(src_data)
        log.info("writing alignment as html to %s"% args.dest_fn)
        with cuda.cupy.cuda.Device(args.gpu):
            loss, attn_all = batch_align(
                                        encdec, eos_idx, zip(src_data, tgt_data), batch_size = args.mb_size, gpu = args.gpu)
        tgt_voc_with_unk = tgt_voc + ["#T_UNK#"]
        src_voc_with_unk = src_voc + ["#S_UNK#"]
        
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
            src_w = [src_voc_with_unk[idx] for idx in src_idx_list] + ["SUM_ATTN"]
            tgt_w = [tgt_voc_with_unk[idx] for idx in tgt_idx_list]
#             for j in xrange(len(tgt_idx_list)):
#                 tgt_idx_list.append(tgt_voc_with_unk[t_and_attn[j][0]])
#             
    #         print [src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list
            p1 = visualisation.make_alignment_figure(
                            src_w, tgt_w, alignment)
#             p2 = visualisation.make_alignment_figure(
#                             [src_voc_with_unk[idx] for idx in src_idx_list], tgt_idx_list, alignment)
            plots_list.append(p1)
        p_all = visualisation.vplot(*plots_list)
        visualisation.output_file(args.dest_fn)
        visualisation.show(p_all)
#     for t in translations_with_attn:
#         for x, attn in t:
#             print x, attn
            
            
#         out.write(convert_idx_to_string([x for x, attn in t], tgt_voc + ["#T_UNK#"]) + "\n")
    
if __name__ == '__main__':
    commandline() 
    
    