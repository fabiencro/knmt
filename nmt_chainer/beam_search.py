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

def iterate_best_score(new_scores, beam_width):
    nb_cases, v_size = new_scores.shape
    new_costs_flattened =  cuda.to_cpu( - new_scores).ravel()

    # TODO replace wit a cupy argpartition when/if implemented
    best_idx = np.argpartition( new_costs_flattened, beam_width)[:beam_width]

    all_num_cases = best_idx / v_size
    all_idx_in_cases = best_idx % v_size

    for num in xrange(len(best_idx)):
        idx = best_idx[num]
        num_case = all_num_cases[num]
        idx_in_case = all_idx_in_cases[num]
        yield num_case, idx_in_case, new_costs_flattened[idx]

def iterate_eos_scores(new_scores, eos_idx):
    nb_cases, v_size = new_scores.shape

    for num_case in xrange(nb_cases):
        idx_in_case = eos_idx
        yield num_case, idx_in_case, cuda.to_cpu(new_scores[num_case, eos_idx])

def update_next_lists(num_case, idx_in_case, new_cost, eos_idx, new_state_ensemble, finished_translations, current_translations, current_attentions,
           next_states_list, next_words_list, next_score_list, next_translations_list, 
           attn_ensemble, next_attentions_list, need_attention = False):
    if idx_in_case == eos_idx:
        if need_attention:
            finished_translations.append((current_translations[num_case], 
                                      -new_cost,
                                      current_attentions[num_case]
                                      ))
        else:
            finished_translations.append((current_translations[num_case], 
                                      -new_cost))
    else:
        next_states_list.append(
            
            [
               (Variable(new_state.data[num_case].reshape(1,-1), volatile = "auto")
                if (not isinstance(new_state, tuple)) else
                
                (Variable(new_state[0].data[num_case].reshape(1,-1), volatile = "auto"),
                 Variable(new_state[1].data[num_case].reshape(1,-1), volatile = "auto")
                 )
                )
             for new_state in new_state_ensemble
             ]
             )
        
        next_words_list.append(idx_in_case)
        next_score_list.append(-new_cost)
        next_translations_list.append(current_translations[num_case] + [idx_in_case])
        if need_attention:
            xp = cuda.get_array_module(attn_ensemble[0].data)
            attn_summed = xp.zeros((attn_ensemble[0].data.shape), dtype = xp.float32)
            for attn in attn_ensemble:
                attn_summed += attn.data[num_case]
            attn_summed /= len(attn_ensemble)
            next_attentions_list.append(current_attentions[num_case] + [attn_summed])
            
def compute_next_lists(new_state_ensemble, new_scores, beam_width, eos_idx, current_translations, finished_translations, 
      current_attentions, attn_ensemble, force_finish = False, need_attention = False):

    next_states_list = []
    next_words_list = []
    next_score_list = []
    next_translations_list = []
    next_attentions_list = []
    
    if force_finish:
        score_iterator = iterate_eos_scores(new_scores, eos_idx)
    else:
        score_iterator = iterate_best_score(new_scores, beam_width)
        
    for num_case, idx_in_case, new_cost in score_iterator:
        update_next_lists(num_case, idx_in_case, new_cost, eos_idx, new_state_ensemble, 
                          finished_translations, current_translations, current_attentions,
           next_states_list, next_words_list, next_score_list, next_translations_list, 
           attn_ensemble, next_attentions_list, need_attention = need_attention)
        assert len(next_states_list) <= beam_width
#             if len(next_states_list) >= beam_width:
#                 break

    return next_states_list, next_words_list, next_score_list, next_translations_list, next_attentions_list


def compute_next_states_and_scores(dec_ensemble, compute_ctxt_ensemble, current_states_ensemble, current_words):
    xp = cuda.get_array_module(dec_ensemble[0].initial_state.data)
    current_states_ensemble_no_cell = [(cs[1] if isinstance(cs, tuple) else cs) for cs in current_states_ensemble]

    ci_ensemble, attn_ensemble = zip(*[compute_ctxt(current_states) for (compute_ctxt, current_states) in zip(
                compute_ctxt_ensemble, current_states_ensemble_no_cell) ])
    
    if current_words is not None:
        prev_y_ensemble = [model.emb(current_words) for model in dec_ensemble]
    else:
        prev_y_ensemble = [F.reshape(model.bos_embeding, (1, -1)) for model in dec_ensemble]

    concatenated_ensemble = [F.concat( (prev_y, ci) ) for (prev_y, ci) in zip(prev_y_ensemble, ci_ensemble)]
    
    new_state_ensemble = []
    for model, current_states, concatenated in zip(dec_ensemble, current_states_ensemble, concatenated_ensemble):
        if model.cell_type == "lstm":
            new_state_for_this_model = model.gru(current_states[0], current_states[1], concatenated)
        else:
            new_state_for_this_model = model.gru(current_states, concatenated)
        new_state_ensemble.append(new_state_for_this_model)

    all_concatenated_ensemble = [ (F.concat((concatenated, new_state)) 
                               if (not isinstance(new_state, tuple)) else
                               F.concat((concatenated, new_state[1])) 
                             )
                             for (concatenated, new_state)
                             in zip(concatenated_ensemble, new_state_ensemble)]
                             
                             
    logits_ensemble = [model.lin_o(model.maxo(all_concatenated)) for (model, all_concatenated) in
                               zip(dec_ensemble, all_concatenated_ensemble)]
    
    # Combine the scores of the ensembled models
    combined_scores = xp.zeros((logits_ensemble[0].data.shape), dtype = xp.float32)
    for logits in logits_ensemble:
        combined_scores += xp.log(F.softmax(logits).data)
    combined_scores /= len(dec_ensemble)
    
    return combined_scores, new_state_ensemble, attn_ensemble
    
def advance_one_step(dec_ensemble, compute_ctxt_ensemble, eos_idx, current_translations_states, beam_width, finished_translations,
                     force_finish = False, need_attention = False):
    xp = cuda.get_array_module(dec_ensemble[0].initial_state.data)
    
    current_translations, current_scores, current_states_ensemble, current_words, current_attentions = current_translations_states

    # Compute the next states and associated next word scores
    combined_scores, new_state_ensemble, attn_ensemble = compute_next_states_and_scores(
                dec_ensemble, compute_ctxt_ensemble, current_states_ensemble, current_words)
    
    nb_cases, v_size = combined_scores.shape
    assert nb_cases <= beam_width
    
    # Add the new scores to the previous ones for each states in the beam
    new_scores = current_scores[:, xp.newaxis] + combined_scores
    
    # Compute the list of new translation states after pruning
    next_states_list, next_words_list, next_score_list, next_translations_list, next_attentions_list = compute_next_lists(
                new_state_ensemble, new_scores, beam_width, eos_idx, 
                current_translations, finished_translations, 
                current_attentions, attn_ensemble, force_finish = force_finish, need_attention = need_attention)

    if len(next_states_list) == 0:
        return None # We only found finished translations

    # Create the new translation states
    
    next_words_array = np.array(next_words_list, dtype = np.int32)
    if xp is not np:
        next_words_array = cuda.to_gpu(next_words_array)
    
    concatenated_next_states_list = []
    for next_states_list_one_model in zip(*next_states_list):
        if isinstance(next_states_list_one_model[0], tuple):
            next_cells_this_model, next_states_this_model = zip(*next_states_list_one_model)
            concatenated_next_states_list.append(
                    (F.concat(next_cells_this_model, axis = 0),
                     F.concat(next_states_this_model, axis = 0)
                     )
                     )
        else:
            concatenated_next_states_list.append(F.concat(next_states_list_one_model, axis = 0))
            
    current_translations_states = (next_translations_list,
                                xp.array(next_score_list),
                                concatenated_next_states_list,
                                Variable(next_words_array, volatile = "auto"),
                                next_attentions_list
                                )
    
    return current_translations_states

def ensemble_beam_search(model_ensemble, src_batch, src_mask, nb_steps, eos_idx, beam_width = 20, need_attention = False,
                         force_finish = False):
    
    fb_concat_list = [model.enc(src_batch, src_mask) for model in model_ensemble]
    
    mb_size, nb_elems, Hi = fb_concat_list[0].data.shape
    assert Hi == model_ensemble[0].dec.Hi, "%i != %i"%(Hi, model_ensemble[0].dec.Hi)
    assert len(model_ensemble) >= 1
    xp = cuda.get_array_module(model_ensemble[0].dec.initial_state.data)

    dec_ensemble = [model.dec for model in model_ensemble]

    compute_ctxt_ensemble = [model.attn_module.compute_ctxt_demux(fb_concat, src_mask) for (model, fb_concat) in 
                         zip(dec_ensemble, fb_concat_list)]

    assert mb_size == 1
    finished_translations = []
    
    # Create the initial Translation state
    previous_states_ensemble = []
    for model in dec_ensemble:
        initial_state_for_this_model =  F.reshape(model.initial_state, (1, -1))
        if model.cell_type == "lstm":
            initial_cell_for_this_model = Variable(
                        model.xp.broadcast_to(model.initial_cell, (1, model.Ho)), volatile = "off")
            previous_states_ensemble.append( (initial_cell_for_this_model, initial_state_for_this_model) )
        else:
            previous_states_ensemble.append( initial_state_for_this_model )
    
    current_translations_states = (
                            [[]], # translations
                            xp.array([0]), # scores
                            previous_states_ensemble,  #previous states
                            None, #previous words
                            [[]] #attention
                            )
    
    # Proceed with the search
    for num_step in xrange(nb_steps):
        current_translations_states = advance_one_step(
                            dec_ensemble, 
                            compute_ctxt_ensemble, 
                            eos_idx, 
                            current_translations_states, 
                            beam_width, 
                            finished_translations,
                            force_finish = force_finish and num_step == (nb_steps -1),
                            need_attention = need_attention)
        if current_translations_states is None:
            break
        
#     print finished_translations, need_attention
        
    # Return finished translations
    if len (finished_translations) == 0:
        if need_attention:
            finished_translations.append(([], 0, []))
        else:
            finished_translations.append(([], 0))
    return finished_translations

def ensemble_beam_search_old(model_list, src_batch, src_mask, nb_steps, eos_idx, beam_width = 20, need_attention = False,
                         force_finish = False):
    
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
    
    previous_states = []
    for model in dec_list:
        initial_state_for_this_model =  F.reshape(model.initial_state, (1, -1))
        if model.cell_type == "lstm":
            initial_cell_for_this_model = Variable(
                        model.xp.broadcast_to(model.initial_cell, (1, model.Ho)), volatile = "off")
            previous_states.append( (initial_cell_for_this_model, initial_state_for_this_model) )
        else:
            previous_states.append( initial_state_for_this_model )
    
    current_translations_states = (
                            [[]], # translations
                            xp.array([0]), # scores
                            previous_states,  #previous states
                            None, #previous words
                            [[]] #attention
                            )
    for num_step in xrange(nb_steps):
        current_translations, current_scores, current_states_list, current_words, current_attentions = current_translations_states

        current_states_list_no_cell = [(cs[1] if isinstance(cs, tuple) else cs) for cs in current_states_list]

        ci_list, attn_list = zip(*[compute_ctxt(current_states) for (compute_ctxt, current_states) in zip(
                    compute_ctxt_list, current_states_list_no_cell) ])
        if current_words is not None:
            prev_y_list = [model.emb(current_words) for model in dec_list]
        else:
            prev_y_list = [F.reshape(model.bos_embeding, (1, -1)) for model in dec_list]



        concatenated_list = [F.concat( (prev_y, ci) ) for (prev_y, ci) in zip(prev_y_list, ci_list)]
        
        
        new_state_list = []
        for model, current_states, concatenated in zip(dec_list, current_states_list, concatenated_list):
            if model.cell_type == "lstm":
#                 print current_states[0].volatile, current_states[1].volatile, concatenated.volatile
                new_state_for_this_model = model.gru(current_states[0], current_states[1], concatenated)
            else:
                new_state_for_this_model = model.gru(current_states, concatenated)
            new_state_list.append(new_state_for_this_model)
                
            
#         new_state_list = [( model.gru(current_states[0], current_states[1], concatenated)
#                            if model.cell_type == "lstm" else  
#                            model.gru(current_states, concatenated)
#                            )
#                           
#                           for 
#                           (model, current_states, concatenated) in zip(dec_list, current_states_list, concatenated_list)]



        all_concatenated_list = [ (F.concat((concatenated, new_state)) 
                                   if (not isinstance(new_state, tuple)) else
                                   F.concat((concatenated, new_state[1])) 
                                 )
                                 for (concatenated, new_state)
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
                    
                    [
                       (Variable(new_state.data[num_case].reshape(1,-1), volatile = "auto")
                        if (not isinstance(new_state, tuple)) else
                        
                        (Variable(new_state[0].data[num_case].reshape(1,-1), volatile = "auto"),
                         Variable(new_state[1].data[num_case].reshape(1,-1), volatile = "auto")
                         )
                        )
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

        concatenated_next_states_list = []
        for next_states_list_one_model in zip(*next_states_list):
            if isinstance(next_states_list_one_model[0], tuple):
                next_cells_this_model, next_states_this_model = zip(*next_states_list_one_model)
#                 assert len(next_states_list_one_model) == 2 #lstm case
                concatenated_next_states_list.append(
                        (F.concat(next_cells_this_model, axis = 0),
                         F.concat(next_states_this_model, axis = 0)
                         )
                         )
            else:
                concatenated_next_states_list.append(F.concat(next_states_list_one_model, axis = 0))
                
        current_translations_states = (next_translations_list,
                                    xp.array(next_score_list),
                                    
                                    concatenated_next_states_list,
                                       
                                       
                                    Variable(next_words_array, volatile = "auto"),
                                    next_attentions_list
                                    )

    if len (finished_translations) == 0:
        if need_attention:
            finished_translations.append(([], 0, []))
        else:
            finished_translations.append(([], 0))
    return finished_translations
