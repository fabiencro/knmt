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
    """
    Create generator over the beam_width best scores.

    Args:
        new_scores: a numpy/cupy 2-dimensional array
        beam_width: a positive integer

    Returns:
        a generator that yields tuples (num_case, idx_in_case, score) where
            score is one of the top beam_width values of new_scores
            num_case is the row of this score in new_scores
            idx_in_case is the column of this score
    """
    nb_cases, v_size = new_scores.shape
    new_costs_flattened = cuda.to_cpu(- new_scores).ravel()

    # TODO replace wit a cupy argpartition when/if implemented
    best_idx = np.argpartition(new_costs_flattened, beam_width)[:beam_width]

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


def update_next_lists(num_case, idx_in_case, new_cost, eos_idx, new_state_ensemble, finished_translations, current_translations,
                      current_attentions,
                      next_states_list, next_words_list, next_score_list, next_normalized_score_list, next_translations_list,
                      attn_ensemble, next_attentions_list, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, need_attention=False):
    """
    Updates the lists containing the infos on translations in current beam

    Args:
        num_case: the index of the translation in current_translations we are going to try and extend
        idx_in_case: vocabulary index of the word we want to add to current_translations[num_case]
        eos_idx: value of the EOS index (so that we can chexk for equality with idx_in_case)

        new_state_ensemble: list of states as returned by compute_next_states_and_scores

        finished_translations: list of finished translations (for which EOS was generated)
                    each item in the list is a tuple (translation, score) or (translation, score, attention) if need_attention = True

        current_translations: list of unfinished translations in the current beam
                                each item is a sequence of target word index

        current_attentions: list of attentions for the current beam
                                each item corresponds to an item in current_translations

        next_states_list: the states extracted from new_state_ensemble corresponding to the translations in next_translations_list
        next_words_list: list of target word index corresponding to the last words of each translation in next_translations_list
        next_score_list: list of scores corresponding to the translations in next_translations_list
        next_translations_list: partially constructed list of translations for the next beam
            same structure as current_translations
        next_attentions_list: attention list corresponding to the translations in next_translations_list

        need_attention: if True, keep the attention

    Return:
        Returns None.
        But the lists finished_translations, next_states_list, next_words_list, next_score_list, next_translations_list, next_attentions_list
            will be updated.
    """
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
            [tuple([Variable(substates.data[num_case].reshape(1, -1), volatile="auto") for substates in new_state])
             for new_state in new_state_ensemble]
        )

        next_words_list.append(idx_in_case)
        next_score_list.append(-new_cost)

        # Compute the normalized score if needed.
        if beam_score_coverage_penalty == "google":
            coverage_penalty = 0
            if len(current_attentions[num_case]) > 0:
                xp = cuda.get_array_module(attn_ensemble[0].data)
                log_of_min_of_sum_over_j = xp.log(xp.minimum(
                    sum(current_attentions[num_case]), xp.array(1.0)))
                coverage_penalty = beam_score_coverage_penalty_strength * \
                    xp.sum(log_of_min_of_sum_over_j)
            normalized_score = -new_cost + coverage_penalty
            next_normalized_score_list.append(normalized_score)

        next_translations_list.append(
            current_translations[num_case] + [idx_in_case])
        if need_attention:
            xp = cuda.get_array_module(attn_ensemble[0].data)
            attn_summed = xp.zeros((attn_ensemble[0].data[0].shape), dtype=xp.float32)
            for attn in attn_ensemble:
                attn_summed += attn.data[num_case]
            attn_summed /= len(attn_ensemble)
            next_attentions_list.append(current_attentions[num_case] + [attn_summed])


def compute_next_lists(new_state_ensemble, new_scores, beam_width, beam_pruning_margin,
                       beam_score_length_normalization, beam_score_length_normalization_strength,
                       beam_score_coverage_penalty, beam_score_coverage_penalty_strength,
                       eos_idx,
                       current_translations,
                       finished_translations,
                       current_attentions,
                       attn_ensemble,
                       force_finish=False,
                       need_attention=False):
    """
        Compute the informations for the next beam.

        Args:
            new_state_ensemble: list of states as returned by compute_next_states_and_scores
            new_scores: numpy/cupy array of float32 representing the scores for each augmented translation
                             new_scores[num_case][idx] is the partial score of the partial
                             translation  current_translations[num_case] augmented with the word whose
                             target vocabulary index is idx.
            beam_width: maximum number of translations in a beam
            beam_pruning_margin: maximum difference of scores for translations in the same beam
            eos_idx: index of EOS in the target vocabulary
            current_translations: list of partial translations in the current beam
                                    each item is a sequence of target vocabulary index
            finished_translations: list of finished translations found so far (ie for which EOS was generated)
                    each item in the list is a tuple (translation, score) or (translation, score, attention) if need_attention = True
            current_attentions: attention for each of the partial translation in current_translations
            attn_ensemble: attention generated when computing the states in new_state_ensemble,
                            as generated by compute_next_states_and_scores
            force_finish: force the generation of EOS if we did not find a translation after nb_steps steps
            need_attention: if True, attention is kept

        Return:
            A tuple (next_states_list, next_words_list, next_score_list, next_translations_list, next_attentions_list)
                containing the informations for the next beam
    """
    # lists that contain infos on the current beam
    next_states_list = []  # each item is a list of list
    next_words_list = []
    next_score_list = []
    next_normalized_score_list = []
    next_translations_list = []
    next_attentions_list = []

    if force_finish:
        score_iterator = iterate_eos_scores(new_scores, eos_idx)
    else:
        score_iterator = iterate_best_score(new_scores, beam_width)

    for num_case, idx_in_case, new_cost in score_iterator:
        if len(current_translations[num_case]) > 0:
            if beam_score_length_normalization == 'simple':
                new_cost /= len(current_translations[num_case])
            elif beam_score_length_normalization == 'google':
                new_cost /= (pow((len(current_translations[num_case]) + 5), beam_score_length_normalization_strength) / pow(6, beam_score_length_normalization_strength))
        update_next_lists(num_case, idx_in_case, new_cost, eos_idx, new_state_ensemble,
                          finished_translations, current_translations, current_attentions,
                          next_states_list, next_words_list, next_score_list, next_normalized_score_list, next_translations_list,
                          attn_ensemble, next_attentions_list, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, need_attention=need_attention)
        assert len(next_states_list) <= beam_width
#             if len(next_states_list) >= beam_width:
#                 break

    # Prune items that have a score worse than beam_pruning_margin below the
    # best score.
    if (beam_pruning_margin is not None and next_score_list):
        best_next_score = max(next_score_list)
        bad_score_indices = [
            idx for idx,
            elem in enumerate(next_score_list) if (
                best_next_score -
                elem > beam_pruning_margin)]

        for i in bad_score_indices[::-1]:
            del next_states_list[i]
            del next_words_list[i]
            del next_score_list[i]
            del next_translations_list[i]
            del next_attentions_list[i]
            if beam_score_coverage_penalty == "google":
                del next_normalized_score_list[i]

    # Prune items that have a normalized score worse than beam_pruning_margin
    # below the best normalized score.
    if (beam_score_coverage_penalty ==
            "google" and beam_pruning_margin is not None and next_normalized_score_list):
        best_next_normalized_score = max(next_normalized_score_list)
        bad_score_indices = [
            idx for idx,
            elem in enumerate(next_normalized_score_list) if (
                best_next_normalized_score -
                elem > beam_pruning_margin)]

        for i in bad_score_indices[::-1]:
            del next_states_list[i]
            del next_words_list[i]
            del next_score_list[i]
            del next_normalized_score_list[i]
            del next_translations_list[i]
            del next_attentions_list[i]

    return next_states_list, next_words_list, next_score_list, next_translations_list, next_attentions_list


def compute_next_states_and_scores(dec_cell_ensemble, current_states_ensemble, current_words,
                                   prob_space_combination=False):
    """
        Compute the next states and scores when giving current_words as input to the decoding cells in dec_cell_ensemble.

        Args:
            dec_cell_ensemble: list of decoder cells conditionalized on the input sentence
                if the length of this list is larger than one, then we will proceed to do ensemble decoding
            current_states_ensemble: list of decoder states
                                        one item for each decoder cell in dec_cell_ensemble
                                        each item can actually be a minibatch of states
            current_words: array of int32 representing the next words to give as input
                            if None, the decoder cell should use its BOS embedding as input
            prob_space_combination: if true, ensemble scores are combined by geometric average instead of arithmetic average

        Return:
            A tuple (combined_scores, new_state_ensemble, attn_ensemble) where:
                combined_scores is the log_probability returned by the decoder cell
                    or a combination of the log probabilities if there was more than one cell in dec_cell_ensemble
                new_state_ensemble is a list of states of same length as dec_cell_ensemble corresponding to the
                    new decoder states after the input of current_words
                attn_ensemble is a list attention generated by each decoder cell after being given the input current_words
    """
#     xp = cuda.get_array_module(dec_ensemble[0].initial_state.data)
    xp = dec_cell_ensemble[0].xp

    if current_words is not None:
        states_logits_attn_ensemble = [dec_cell(states, current_words) for (dec_cell, states) in zip(
            dec_cell_ensemble, current_states_ensemble)]
    else:
        assert all(x is None for x in current_states_ensemble)
        states_logits_attn_ensemble = [dec_cell.get_initial_logits(1) for dec_cell in dec_cell_ensemble]

    new_state_ensemble, logits_ensemble, attn_ensemble = zip(*states_logits_attn_ensemble)

    # Combine the scores of the ensembled models
    combined_scores = xp.zeros((logits_ensemble[0].data.shape), dtype=xp.float32)

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


def advance_one_step(dec_cell_ensemble, eos_idx, current_translations_states, beam_width, beam_pruning_margin,
                     beam_score_length_normalization,
                     beam_score_length_normalization_strength,
                     beam_score_coverage_penalty,
                     beam_score_coverage_penalty_strength,
                     finished_translations,
                     force_finish=False, need_attention=False,
                     prob_space_combination=False):
    """
        Generate the partial translations / decoder states in the next beam

        Args:
            dec_cell_ensemble: list of decoder cells conditionalized on the input sentence
                if the length of this list is larger than one, then we will proceed to do ensemble decoding
            eos_idx: the index of EOS element in the target vocabulary
            current_translations_states: a tuple representing the state of the current beam
                the tuple has the shape (translations, score, previous_states, previous_words, attention) where:
                    translations is a list of unfinished translations
                        each item is a sequence of target vocabulary index
                    score is a numpy/cupy array of same length as translations, giving the score for each item in translations
                    previous_states is a list of "states", with one state for each cell in dec_cell_ensemble
                        "states" $i$ in previous_states represents the state of cell $i$ in dec_cell_ensemble after
                            generating the translations in 'translations'. Each states thus actually represents up to
                            beam_width state, onr for each translation in 'translations'.
                            A value of None for a state indicate that the initial state of the decoder should be used.
                    previous_words is a numpy/cupy array of int32 containing the index of the last word of each translation
                        in 'translations'. If its value is None, it means the decoder should use its BOS embedding as input.
                    attention is the list of attentions generated for each of the translations in 'translations'
            finished_translations: list of finished translations
                each item in the list is a tuple (translation, score) or (translation, score, attention) if need_attention = True
            beam_width, beam_pruning_margin, force_finish, need_attention, prob_space_combination:
                see ensemble_beam_search documentation

        Returns:
            A tuple (translations, score, states, words, attention) similar to the input
                argument current_translations_states, but corresponding to the next beam.
    """

#     xp = cuda.get_array_module(dec_ensemble[0].initial_state.data)
    xp = dec_cell_ensemble[0].xp
    current_translations, current_scores, current_states_ensemble, current_words, current_attentions = current_translations_states

    # Compute the next states and associated next word scores
    combined_scores, new_state_ensemble, attn_ensemble = compute_next_states_and_scores(
        dec_cell_ensemble, current_states_ensemble, current_words,
        prob_space_combination=prob_space_combination)

    nb_cases, v_size = combined_scores.shape
    assert nb_cases <= beam_width

    # Add the new scores to the previous ones for each states in the beam
    new_scores = current_scores[:, xp.newaxis] + combined_scores

    # Compute the list of new translation states after pruning
    next_states_list, next_words_list, next_score_list, next_translations_list, next_attentions_list = compute_next_lists(
        new_state_ensemble, new_scores, beam_width, beam_pruning_margin,
        beam_score_length_normalization, beam_score_length_normalization_strength,
        beam_score_coverage_penalty, beam_score_coverage_penalty_strength,
        eos_idx,
        current_translations, finished_translations,
        current_attentions, attn_ensemble, force_finish=force_finish, need_attention=need_attention)

    if len(next_states_list) == 0:
        return None  # We only found finished translations

    # Create the new translation states

    next_words_array = np.array(next_words_list, dtype=np.int32)
    if xp is not np:
        next_words_array = cuda.to_gpu(next_words_array)

    concatenated_next_states_list = []
    for next_states_list_one_model in zip(*next_states_list):
        concatenated_next_states_list.append(
            tuple([F.concat(substates, axis=0) for substates in zip(*next_states_list_one_model)])
        )

    next_translations_states = (next_translations_list,
                                xp.array(next_score_list),
                                concatenated_next_states_list,
                                Variable(next_words_array, volatile="auto"),
                                next_attentions_list
                                )

    return next_translations_states


def ensemble_beam_search(model_ensemble, src_batch, src_mask, nb_steps, eos_idx,
                         beam_width=20, beam_pruning_margin=None,
                         beam_score_length_normalization=None,
                         beam_score_length_normalization_strength=0.2,
                         beam_score_coverage_penalty=None,
                         beam_score_coverage_penalty_strength=0.2,
                         need_attention=False,
                         force_finish=False,
                         prob_space_combination=False, use_unfinished_translation_if_none_found=False):
    """
    Compute translations using a beam-search algorithm.

    Args:
        model_ensemble: list of (trained) EncoderDecoder models
                    if the list contains more than one model, we do ensemble decoding : the scores given by each
                    models are combined to produce the best translation
        src_batch: input sentence in batch form, as generated by make_batch_src.
                    src_batch should only contains one sentence (ie minibatch size is one)
        src_mask: mask value returned by make_batch_src
        nb_steps: maximum length of the generated translation
        eos_idx: index of the EOS element in the target vocabulary
        beam_width: number of partial translation kept in each beam
        beam_pruning_margin: maximum score difference accepted within a beam
        need_attention: if True, will return the attention values for each translation
        force_finish: force the generation of EOS if we did not find a translation after nb_steps steps
        prob_space_combination: if true, ensemble scores are combined by geometric average instead of arithmetic average
        use_unfinished_translation_if_none_found: will ureturn unfinished translation if we did not find a translation after nb_steps steps

    Return:
        list of translations
            each item in the list is a tuple (translation, score) or (translation, score, attention) if need_attention = True
    """
    mb_size = src_batch[0].data.shape[0]
    assert len(model_ensemble) >= 1
    xp = model_ensemble[0].xp

    dec_cell_ensemble = [model.give_conditionalized_cell(src_batch, src_mask, noise_on_prev_word=False,
                                                         mode="test", demux=True) for model in model_ensemble]

    assert mb_size == 1
    # TODO: if mb_size == 1 then src_mask value unnecessary -> remove?

    finished_translations = []

    # Create the initial Translation state
    previous_states_ensemble = [None] * len(model_ensemble)

    # Current_translations_states will hold the information for the current beam
    current_translations_states = (
        [[]],  # translations
        xp.array([0]),  # scores
        previous_states_ensemble,  # previous states
        None,  # previous words
        [[]]  # attention
    )

    # Proceed with the search
    for num_step in xrange(nb_steps):
        current_translations_states = advance_one_step(
            dec_cell_ensemble,
            eos_idx,
            current_translations_states,
            beam_width,
            beam_pruning_margin,
            beam_score_length_normalization,
            beam_score_length_normalization_strength,
            beam_score_coverage_penalty,
            beam_score_coverage_penalty_strength,
            finished_translations,
            force_finish=force_finish and num_step == (nb_steps - 1),
            need_attention=need_attention,
            prob_space_combination=prob_space_combination)

        if current_translations_states is None:
            break

#     print finished_translations, need_attention

    # Return finished translations
    if len(finished_translations) == 0:
        if use_unfinished_translation_if_none_found:
            assert current_translations_states is not None
            if need_attention:
                translations, scores, _, _, attentions = current_translations_states
                finished_translations.append(
                    (translations[0], scores[0], attentions[0]))
            else:
                finished_translations.append((translations[0], scores[0]))
        else:
            if need_attention:
                finished_translations.append(([], 0, []))
            else:
                finished_translations.append(([], 0))
    return finished_translations
