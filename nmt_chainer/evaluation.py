#!/usr/bin/env python
"""eval.py: Use a RNNSearch Model"""

__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

from chainer import cuda

from utils import make_batch_src, make_batch_src_tgt, minibatch_provider, compute_bleu_with_unk_as_wrong, de_batch
import logging
import codecs
import operator
# import h5py

logging.basicConfig()
log = logging.getLogger("rnns:evaluation")
log.setLevel(logging.INFO)

def translate_to_file(encdec, eos_idx, test_src_data, mb_size, tgt_indexer, 
                   translations_fn, test_references = None, control_src_fn = None, src_indexer = None, gpu = None, nb_steps = 50,
                   reverse_src = False, reverse_tgt = False,
                   s_unk_tag = "#S_UNK#", t_unk_tag = "#T_UNK#"):
    
    log.info("computing translations")
    translations = greedy_batch_translate(encdec, eos_idx, test_src_data, 
                        batch_size = mb_size, gpu = gpu, nb_steps = nb_steps, 
                        reverse_src = reverse_src, reverse_tgt = reverse_tgt)
    
    log.info("writing translation of set to %s"% translations_fn)
    out = codecs.open(translations_fn, "w", encoding = "utf8")
    for t in translations:
        if t[-1] == eos_idx:
            t = t[:-1]
#         out.write(convert_idx_to_string(t, tgt_voc) + "\n")
        out.write(" ".join(tgt_indexer.deconvert(t, unk_tag = t_unk_tag)) + "\n")
        
    
    if control_src_fn is not None:
        assert src_indexer is not None
        control_out = codecs.open(control_src_fn, "w", encoding = "utf8")
        log.info("writing src of test set to %s"% control_src_fn)
        for s in test_src_data:
#             control_out.write(convert_idx_to_string(s, src_voc) + "\n")
            control_out.write(" ".join(src_indexer.deconvert(s, unk_tag = s_unk_tag)) + "\n")
        
    if test_references is not None:
#         unk_id = tgt_indexer.get_unk_idx()  #len(tgt_voc) - 1
        new_unk_id_ref = len(tgt_indexer) + 7777
        new_unk_id_cand = len(tgt_indexer) + 9999
        bc = compute_bleu_with_unk_as_wrong(test_references, [t[:-1] for t in translations], tgt_indexer.is_unk_idx, new_unk_id_ref, new_unk_id_cand)
        log.info("bleu: %r"%bc)
        return bc
    else:
        return None


def compute_loss_all(encdec, test_data, eos_idx, mb_size, gpu = None, reverse_src = False, reverse_tgt = False):
    mb_provider_test = minibatch_provider(test_data, eos_idx, mb_size, nb_mb_for_sorting = -1, loop = False,
                                          gpu = gpu, volatile = "on",
                                          reverse_src = reverse_src, reverse_tgt = reverse_tgt)
    test_loss = 0
    test_nb_predictions = 0
    for src_batch, tgt_batch, src_mask in mb_provider_test:
        loss, attn = encdec(src_batch, tgt_batch, src_mask, raw_loss_info = True)
        test_loss += loss[0].data
        test_nb_predictions += loss[1]
    test_loss /= test_nb_predictions
    return test_loss

def greedy_batch_translate(encdec, eos_idx, src_data, batch_size = 80, gpu = None, get_attention = False, nb_steps = 50,
                           reverse_src = False, reverse_tgt = False):
    nb_ex = len(src_data)
    nb_batch = nb_ex / batch_size + (1 if nb_ex % batch_size != 0 else 0)
    res = []
    attn_all = []
    for i in range(nb_batch):
        current_batch_raw_data = src_data[i * batch_size : (i + 1) * batch_size]
        
        if reverse_src:
            current_batch_raw_data_new = []
            for src_side in current_batch_raw_data:
                current_batch_raw_data_new.append(src_side[::-1])
            current_batch_raw_data = current_batch_raw_data_new
            
        src_batch, src_mask = make_batch_src(current_batch_raw_data, gpu = gpu, volatile = "on")
        sample_greedy, score, attn_list = encdec(src_batch, nb_steps, src_mask, use_best_for_sample = True, 
                                                 keep_attn_values = get_attention)
        deb = de_batch(sample_greedy, mask = None, eos_idx = eos_idx, is_variable = False)
        res += deb
        if get_attention:
            deb_attn = de_batch(attn_list, mask = None, eos_idx = None, is_variable = True, raw = True,
                       reverse = reverse_tgt)
            attn_all += deb_attn 
            
    if reverse_tgt:
        new_res = []
        for t in res:
            if t[-1] == eos_idx:
                new_res.append(t[:-1][::-1] + [t[-1]])
            else:
                new_res.append(t[::-1])
                
        res = new_res
        
    if get_attention:
        assert not reverse_tgt, "not implemented"
        return res, attn_all
    else:
        return res
     
def beam_search_translate(encdec, eos_idx, src_data, beam_width = 20, nb_steps = 50, gpu = None, beam_opt = False,
                          need_attention = False, nb_steps_ratio = None, score_is_divided_by_length = True, 
                          groundhog = False):
    nb_ex = len(src_data)
#     res = []
    for i in range(nb_ex):
        src_batch, src_mask = make_batch_src([src_data[i]], gpu = gpu, volatile = "on")
        assert len(src_mask) == 0
        if nb_steps_ratio is not None:
            nb_steps = int(len(src_data[i]) * nb_steps_ratio) + 1
        translations = encdec.beam_search(src_batch, src_mask, nb_steps = nb_steps, eos_idx = eos_idx, 
                                          beam_width = beam_width,
                                          beam_opt = beam_opt, need_attention = need_attention,
                                    groundhog = groundhog)
#         print "nb_trans", len(translations), [score for _, score in translations]
#         bests = []
#         translations.sort(key = itemgetter(1), reverse = True)
#         bests.append(translations[0])
        if score_is_divided_by_length:
            translations.sort(key = lambda x:x[1]/(len(x[0])+1), reverse = True)
        else:
            translations.sort(key = operator.itemgetter(1), reverse = True)
#         bests.append(translations[0])
#         yield bests
        yield translations[0]
#         res.append(bests)
#     return res
   
def batch_align(encdec, eos_idx, src_tgt_data, batch_size = 80, gpu = None):
    nb_ex = len(src_tgt_data)
    nb_batch = nb_ex / batch_size + (1 if nb_ex % batch_size != 0 else 0)
    sum_loss = 0
    attn_all = []
    for i in range(nb_batch):
        current_batch_raw_data = src_tgt_data[i * batch_size : (i + 1) * batch_size]
#         print current_batch_raw_data
        src_batch, tgt_batch, src_mask, arg_sort = make_batch_src_tgt(
                    current_batch_raw_data, eos_idx = eos_idx, gpu = gpu, volatile = "on", need_arg_sort = True)
        loss, attn_list = encdec(src_batch, tgt_batch, src_mask, keep_attn_values = True)
        deb_attn = de_batch(attn_list, mask = None, eos_idx = None, is_variable = True, raw = True)
        
        assert len(arg_sort) == len(deb_attn)
        de_sorted_attn = [None] * len(deb_attn)
        for xpos in xrange(len(arg_sort)):
            original_pos = arg_sort[xpos]
            de_sorted_attn[original_pos] = deb_attn[xpos]
        
        attn_all += de_sorted_attn 
        sum_loss += float(loss.data)
    return sum_loss, attn_all
            
# def convert_idx_to_string(seq, voc, eos_idx = None):
#     trans = []
#     for idx_tgt in seq:
#         if eos_idx is not None and idx_tgt == eos_idx:
#             trans.append("#EOS#")
#         else:
#             if idx_tgt >= len(voc):
#                 log.warn("found unknown idx in tgt : %i / %i"% (idx_tgt, len(voc)))
#             else:
#                 trans.append(voc[idx_tgt])
#     return " ".join(trans)
# 
# def convert_idx_to_string_with_attn(seq, voc, attn, unk_idx, unk_pattern = "#T_UNK_%i#"):
#     trans = []
#     for num, idx_tgt in enumerate(seq):
#         if idx_tgt == unk_idx:
#             a = attn[num]
#             xp = cuda.get_array_module(a)
#             src_pos = int(xp.argmax(a))
#             trans.append(unk_pattern%src_pos)
#         else:
#             if idx_tgt >= len(voc):
#                 log.warn("found unknown idx in tgt : %i / %i"% (idx_tgt, len(voc)))
#             else:
#                 trans.append(voc[idx_tgt])
#     return " ".join(trans)

def sample_once(encdec, src_batch, tgt_batch, src_mask, src_indexer, tgt_indexer, eos_idx, max_nb = None,
                s_unk_tag = "#S_UNK#", t_unk_tag = "#T_UNK#"):
    print "sample"
    sample_greedy, score, attn_list = encdec(src_batch, 50, src_mask, use_best_for_sample = True, need_score = True)
#                 sample, score = encdec(src_batch, 50, src_mask, use_best_for_sample = False)
    assert len(src_batch[0].data) == len(tgt_batch[0].data)
    assert len(sample_greedy[0]) == len(src_batch[0].data)
    
    debatched_src = de_batch(src_batch, mask = src_mask, eos_idx = None, is_variable= True)
    debatched_tgt = de_batch(tgt_batch, eos_idx = eos_idx, is_variable= True)
    debatched_sample = de_batch(sample_greedy, eos_idx = eos_idx)
    
    for sent_num in xrange(len(debatched_src)):
        if max_nb is not None and sent_num > max_nb:
            break
        src_idx_seq = debatched_src[sent_num]
        tgt_idx_seq = debatched_tgt[sent_num]
        sample_idx_seq = debatched_sample[sent_num]
        print "sent num", sent_num
        print "src idx:", src_idx_seq
        print "src:", " ".join(src_indexer.deconvert(src_idx_seq, unk_tag = s_unk_tag)) #convert_idx_to_string(src_idx_seq, src_voc)
        print "tgt idx:", tgt_idx_seq
        print "tgt:", " ".join(tgt_indexer.deconvert(tgt_idx_seq, unk_tag = t_unk_tag, eos_idx = eos_idx)) # convert_idx_to_string(tgt_idx_seq, tgt_voc, eos_idx = eos_idx)
        print "sample idx:", sample_idx_seq
        print "sample:", " ".join(tgt_indexer.deconvert(sample_idx_seq, unk_tag = t_unk_tag, eos_idx = eos_idx)) #convert_idx_to_string(sample_idx_seq, tgt_voc, eos_idx = eos_idx)