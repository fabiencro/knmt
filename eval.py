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
log = logging.getLogger("rnns:eval")
log.setLevel(logging.INFO)

def translate_to_file(encdec, eos_idx, test_src_data, mb_size, tgt_voc, 
                   translations_fn, test_references = None, control_src_fn = None, src_voc = None, gpu = None):
    
    log.info("computing translations")
    translations = greedy_batch_translate(encdec, eos_idx, test_src_data, batch_size = mb_size, gpu = gpu)
    
    log.info("writing translation of set to %s"% translations_fn)
    out = codecs.open(translations_fn, "w", encoding = "utf8")
    for t in translations:
        out.write(convert_idx_to_string(t, tgt_voc) + "\n")
    
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
        bc = compute_bleu_with_unk_as_wrong(test_references, translations, unk_id, new_unk_id_ref, new_unk_id_cand)
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
        src_batch, src_mask = make_batch_src(current_batch_raw_data, eos_idx = eos_idx, gpu = gpu, volatile = "auto")
        sample_greedy, score, attn_list = encdec(src_batch, 50, src_mask, use_best_for_sample = True, 
                                                 keep_attn_values = get_attention)
        deb = de_batch(sample_greedy, mask = None, eos_idx = eos_idx, is_variable = False)
        res += deb
        if get_attention:
            deb_attn = de_batch(attn_list, mask = None, eos_idx = None, is_variable = True, raw = True)
            attn_all += deb_attn 
#         for sent_num in xrange(len(src_batch[0].data)):
#             res.append([])
#             for smp_pos in range(len(sample_greedy)):
#                 idx_smp = cuda.to_cpu(sample_greedy[smp_pos][sent_num])
#                 if idx_smp == eos_idx:
#                     break
#                 res[-1].append(idx_smp)
    if get_attention:
        return res, attn_all
    else:
        return res
     
# def greedy_batch_translate_with_attn(encdec, eos_idx, src_data, batch_size = 80, gpu = None):
#     nb_ex = len(src_data)
#     nb_batch = nb_ex / batch_size + (1 if nb_ex % batch_size != 0 else 0)
#     res = []
#     attn_all = []
#     for i in range(nb_batch):
#         current_batch_raw_data = src_data[i * batch_size : (i + 1) * batch_size]
#         src_batch, src_mask = make_batch_src(current_batch_raw_data, eos_idx = eos_idx, gpu = gpu, volatile = "auto")
#         
#         sample_greedy, score, attn_list = encdec(src_batch, 50, src_mask, use_best_for_sample = True, 
#                                                  keep_attn_values = True)
#         deb = de_batch(sample_greedy, mask = None, eos_idx = eos_idx, is_variable = False)
#         res += deb
#         deb_attn = de_batch(attn_list, mask = None, eos_idx = None, is_variable = True, raw = True)
#         attn_all += deb_attn
#         de_batch(batch, mask, eos_idx, is_variable)
#         for sent_num in xrange(len(src_batch[0].data)):
#             res.append([])
#             for smp_pos in range(len(sample_greedy)):
#                 idx_smp = cuda.to_cpu(sample_greedy[smp_pos][sent_num])
#                 if idx_smp == eos_idx:
#                     break
#                 
#                 attention_vals = cuda.to_cpu(attn_list[smp_pos].data[sent_num])
#                 res[-1].append((idx_smp, attention_vals))
    return res, attn_all
   
def batch_align(encdec, eos_idx, src_tgt_data, batch_size = 80, gpu = None):
    nb_ex = len(src_tgt_data)
    nb_batch = nb_ex / batch_size + (1 if nb_ex % batch_size != 0 else 0)
    res = []
    for i in range(nb_batch):
        current_batch_raw_data = src_tgt_data[i * batch_size : (i + 1) * batch_size]
        src_batch, tgt_batch, src_mask = make_batch_src_tgt(current_batch_raw_data, eos_idx = eos_idx, gpu = gpu, volatile = "auto")
        
        loss, attn_list = encdec(src_batch, tgt_batch, src_mask, keep_attn_values = True)
        for sent_num in xrange(len(src_batch[0].data)):
            res.append([])
            for smp_pos in range(len(tgt_batch)):
                idx_smp = cuda.to_cpu(tgt_batch[smp_pos][sent_num])
                if idx_smp == eos_idx:
                    break
                
                attention_vals = cuda.to_cpu(attn_list[smp_pos].data[sent_num])
                res[-1].append((idx_smp, attention_vals))
    return res
            
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

def sample_once(encdec, src_batch, tgt_batch, src_mask, src_voc, tgt_voc, eos_idx):
    print "sample"
    sample_greedy, score, attn_list = encdec(src_batch, 50, src_mask, use_best_for_sample = True)
#                 sample, score = encdec(src_batch, 50, src_mask, use_best_for_sample = False)
    assert len(src_batch[0].data) == len(tgt_batch[0].data)
    assert len(sample_greedy[0]) == len(src_batch[0].data)
    
    debatched_src = de_batch(src_batch, mask = src_mask, eos_idx = None, is_variable= True)
    debatched_tgt = de_batch(tgt_batch, eos_idx = eos_idx, is_variable= True)
    debatched_sample = de_batch(sample_greedy, eos_idx = eos_idx)
    
    for sent_num in xrange(len(debatched_src)):
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
                        choices = ["translate", "align", "translate_attn"], help = "target text")
    parser.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
    
    parser.add_argument("--max_nb_ex", type = int, help = "only use the first MAX_NB_EX examples")
    parser.add_argument("--mb_size", type = int, default= 80, help = "Minibatch size")
    parser.add_argument("--nb_batch_to_sort", type = int, default= 20, help = "Sort this many batches by size.")
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
    
    log.info("writing translation of test set to %s"% args.dest_fn)
#     translations = greedy_batch_translate(encdec, eos_idx, src_data, batch_size = args.mb_size, gpu = args.gpu)
    
    
    if args.mode == "translate":
        translations = greedy_batch_translate(
                                        encdec, eos_idx, src_data, batch_size = args.mb_size, gpu = args.gpu)
        out = codecs.open(args.dest_fn, "w", encoding = "utf8")
        for t in translations:
            ct = convert_idx_to_string(t[:-1], tgt_voc + ["#T_UNK#"])
            out.write(ct + "\n")

    elif args.mode == "translate_attn":
        translations, attn_all = greedy_batch_translate(
                                        encdec, eos_idx, src_data, batch_size = args.mb_size, gpu = args.gpu,
                                        get_attention = True)
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
            import visualisation
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
    
    