#!/usr/bin/env python
"""beam_search.py: Implementation of beam_search"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

from _collections import defaultdict
import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import math, random

import logging
logging.basicConfig()
log = logging.getLogger("rnns:beam_search")
log.setLevel(logging.INFO)

def ensemble_beam_search(model_list, src_batch, src_mask, nb_steps, eos_idx, beam_width = 20, need_attention = False):
    
    fb_concat_list = [model.enc(src_batch, src_mask) for model in model_list]
    
    mb_size, nb_elems, Hi = fb_concat_list[0].data.shape
    assert Hi == model_list[0].dec.Hi, "%i != %i"%(Hi, model_list[0].dec.Hi)
    assert len(model_list) >= 1
    xp = cuda.get_array_module(model_list[0].dec.initial_state.data)

    dec_list = [model.dec for model in model_list]

    compute_ctxt_list = [model.attn_module.compute_ctxt_demux(fb_concat, src_mask) for (model, fb_concat) in 
                         zip(dec_list, fb_concat_list)]

    assert mb_size == 1
    finished_translations = []
    current_translations_states = (
                            [[]], # translations
                            xp.array([0]), # scores
                            [F.reshape(model.initial_state, (1, -1)) for model in dec_list],  #previous states
                            None, #previous words
                            [[]] #attention
                            )
    for i in xrange(nb_steps):
        current_translations, current_scores, current_states_list, current_words, current_attentions = current_translations_states

        ci_list, attn_list = zip(*[compute_ctxt(current_states) for (compute_ctxt, current_states) in zip(
                    compute_ctxt_list, current_states_list) ])
        if current_words is not None:
            prev_y_list = [model.emb(current_words) for model in dec_list]
        else:
            prev_y_list = [F.reshape(model.bos_embeding, (1, -1)) for model in dec_list]

        concatenated_list = [F.concat( (prev_y, ci) ) for (prev_y, ci) in zip(prev_y_list, ci_list)]
        new_state_list = [model.gru(current_states, concatenated) for 
                          (model, current_states, concatenated) in zip(dec_list, current_states_list, concatenated_list)]

        all_concatenated_list = [F.concat((concatenated, new_state)) for (concatenated, new_state)
                                 in zip(concatenated_list, new_state_list)]
                                 
                                 
        logits_list = [model.lin_o(model.maxo(all_concatenated)) for (model, all_concatenated) in
                                   zip(dec_list, all_concatenated_list)]
#         probs_v = F.softmax(logits)
#         log_probs_v = F.log(probs_v) # TODO replace wit a logsoftmax if implemented
        
        combined_scores = xp.zeros((logits_list[0].data.shape), dtype = xp.float32)
        for logits in logits_list:
            combined_scores += xp.log(F.softmax(logits).data)
        combined_scores /= len(model_list)
#         xp.sum([xp.log(F.softmax(logits).data) for logits in logits_list], axis = 0)/ len(model_list)
        
#         nb_cases, v_size = probs_v.data.shape
        nb_cases, v_size = combined_scores.shape
        assert nb_cases <= beam_width

        new_scores = current_scores[:, xp.newaxis] + combined_scores #log_probs_v.data
        new_costs_flattened =  cuda.to_cpu( - new_scores).ravel()

        # TODO replace wit a cupy argpartition when/if implemented
        best_idx = np.argpartition( new_costs_flattened, beam_width)[:beam_width]

        all_num_cases = best_idx / v_size
        all_idx_in_cases = best_idx % v_size

        next_states_list = []
        next_words_list = []
        next_score_list = []
        next_translations_list = []
        next_attentions_list = []
        for num in xrange(len(best_idx)):
            idx = best_idx[num]
            num_case = all_num_cases[num]
            idx_in_case = all_idx_in_cases[num]
            if idx_in_case == eos_idx:
                if need_attention:
                    finished_translations.append((current_translations[num_case], 
                                              -new_costs_flattened[idx],
                                              current_attentions[num_case]
                                              ))
                else:
                    finished_translations.append((current_translations[num_case], 
                                              -new_costs_flattened[idx]))
            else:
                next_states_list.append(
                    
                    [Variable(new_state.data[num_case].reshape(1,-1), volatile = "auto")
                     for new_state in new_state_list
                     ]
                     )
                
                
                next_words_list.append(idx_in_case)
                next_score_list.append(-new_costs_flattened[idx])
                next_translations_list.append(current_translations[num_case] + [idx_in_case])
                if need_attention:
                    attn_summed = xp.zeros((attn_list[0].data.shape), dtype = xp.float32)
                    for attn in attn_list:
                        attn_summed += attn.data[num_case]
                    attn_summed /= len(model_list)
                    next_attentions_list.append(current_attentions[num_case] + [attn_summed])
                if len(next_states_list) >= beam_width:
                    break

        if len(next_states_list) == 0:
            break

        next_words_array = np.array(next_words_list, dtype = np.int32)
        if model_list[0].xp is not np:
            next_words_array = cuda.to_gpu(next_words_array)

        current_translations_states = (next_translations_list,
                                    xp.array(next_score_list),
                                    [F.concat(next_states_list_one_model, axis = 0) for 
                                    next_states_list_one_model in zip(*next_states_list)],
                                    Variable(next_words_array, volatile = "auto"),
                                    next_attentions_list
                                    )

    if len (finished_translations) == 0:
        if need_attention:
            finished_translations.append(([], 0, []))
        else:
            finished_translations.append(([], 0))
    return finished_translations