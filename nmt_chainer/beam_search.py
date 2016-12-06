#!/usr/bin/env python
"""beam_search.py: Implementation of beam_search"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


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
            [tuple([Variable(substates.data[num_case].reshape(1,-1), volatile = "auto") for substates in new_state])
                    for new_state in new_state_ensemble]
            
#             [
#                (Variable(new_state.data[num_case].reshape(1,-1), volatile = "auto")
#                 if (not isinstance(new_state, tuple)) else
#                 
#                 (Variable(new_state[0].data[num_case].reshape(1,-1), volatile = "auto"),
#                  Variable(new_state[1].data[num_case].reshape(1,-1), volatile = "auto")
#                  )
#                 )
#              for new_state in new_state_ensemble
#              ]
             )
        
        next_words_list.append(idx_in_case)
        next_score_list.append(-new_cost)
        next_translations_list.append(current_translations[num_case] + [idx_in_case])
        if need_attention:
            xp = cuda.get_array_module(attn_ensemble[0].data)
            attn_summed = xp.zeros((attn_ensemble[0].data[0].shape), dtype = xp.float32)
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

def compute_next_states_and_scores(dec_cell_ensemble, current_states_ensemble, current_words,
                                   prob_space_combination = False):
#     xp = cuda.get_array_module(dec_ensemble[0].initial_state.data)
    xp = dec_cell_ensemble[0].xp
    
    if current_words is not None:
        states_logits_attn_ensemble = [dec_cell(states, current_words) for (dec_cell,states) in zip(
                                                            dec_cell_ensemble, current_states_ensemble)]
    else:
        assert all(x is None for x in current_states_ensemble)
        states_logits_attn_ensemble = [dec_cell.get_initial_logits(1) for dec_cell in dec_cell_ensemble]
    
    
    new_state_ensemble, logits_ensemble, attn_ensemble = zip(*states_logits_attn_ensemble)

    # Combine the scores of the ensembled models
    combined_scores = xp.zeros((logits_ensemble[0].data.shape), dtype = xp.float32)
    
    if not prob_space_combination:
        for logits in logits_ensemble:
            combined_scores += xp.log(F.softmax(logits).data)
        combined_scores /= len(dec_cell_ensemble)
    else:
        for logits in logits_ensemble:
            combined_scores += F.softmax(logits).data
        combined_scores /= len(dec_cell_ensemble)
        combined_scores = xp.log(combined_scores)
        
        
    return combined_scores, new_state_ensemble, attn_ensemble
    
def advance_one_step(dec_cell_ensemble, eos_idx, current_translations_states, beam_width, finished_translations,
                     force_finish = False, need_attention = False,
                     prob_space_combination = False):
#     xp = cuda.get_array_module(dec_ensemble[0].initial_state.data)
    xp = dec_cell_ensemble[0].xp
    current_translations, current_scores, current_states_ensemble, current_words, current_attentions = current_translations_states

    # Compute the next states and associated next word scores
    combined_scores, new_state_ensemble, attn_ensemble = compute_next_states_and_scores(
                dec_cell_ensemble, current_states_ensemble, current_words,
                prob_space_combination = prob_space_combination)
    
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
        concatenated_next_states_list.append(
            tuple([F.concat(substates, axis = 0) for substates in zip(*next_states_list_one_model)])
            )
#         if isinstance(next_states_list_one_model[0], tuple):
#             next_cells_this_model, next_states_this_model = zip(*next_states_list_one_model)
#             concatenated_next_states_list.append(
#                     (F.concat(next_cells_this_model, axis = 0),
#                      F.concat(next_states_this_model, axis = 0)
#                      )
#                      )
#         else:
#             concatenated_next_states_list.append(F.concat(next_states_list_one_model, axis = 0))
            
    current_translations_states = (next_translations_list,
                                xp.array(next_score_list),
                                concatenated_next_states_list,
                                Variable(next_words_array, volatile = "auto"),
                                next_attentions_list
                                )
    
    return current_translations_states

def ensemble_beam_search(model_ensemble, src_batch, src_mask, nb_steps, eos_idx, beam_width = 20, need_attention = False,
                         force_finish = False,
                         prob_space_combination = False, use_unfinished_translation_if_none_found = False):
    
    mb_size = src_batch[0].data.shape[0]
    assert len(model_ensemble) >= 1
    xp = model_ensemble[0].xp
    
    dec_cell_ensemble = [model.give_conditionalized_cell(src_batch, src_mask, noise_on_prev_word = False,
                    mode = "test", demux = True) for model in model_ensemble]
    

    assert mb_size == 1
    finished_translations = []
    
    # Create the initial Translation state
    previous_states_ensemble = [None] * len(model_ensemble)
    
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
                            dec_cell_ensemble, 
                            eos_idx, 
                            current_translations_states, 
                            beam_width, 
                            finished_translations,
                            force_finish = force_finish and num_step == (nb_steps -1),
                            need_attention = need_attention,
                            prob_space_combination = prob_space_combination)
        if current_translations_states is None:
            break
        
#     print finished_translations, need_attention
        
    # Return finished translations
    if len (finished_translations) == 0:
        if use_unfinished_translation_if_none_found:
            assert current_translations_states is not None
            if need_attention:
                translations, scores, _, _, attentions = current_translations_states
                finished_translations.append((translations[0], scores[0], attentions[0]))
            else:
                finished_translations.append((translations[0], scores[0]))
        else:
            if need_attention:
                finished_translations.append(([], 0, []))
            else:
                finished_translations.append(([], 0))
    return finished_translations


