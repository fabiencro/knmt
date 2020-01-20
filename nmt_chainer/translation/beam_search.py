#!/usr/bin/env python
"""beam_search.py: Implementation of beam_search"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import bisect
import dataclasses
import enum
import logging
import operator
import queue
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple, cast

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six
from chainer import Chain, ChainList, Link, Variable, cuda

from nmt_chainer.models.attention import AttentionModule

__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"




#from nmt_chainer.translation.astar_search_utils import AStarParams, Item, TranslationPriorityQueue

logging.basicConfig()
log = logging.getLogger("rnns:beam_search")
log.setLevel(logging.INFO)


@dataclass(eq = False)
class ATranslationState:
    translations: List[List[int]] = field(default_factory=lambda:[[]])
    scores: np.array = field(default_factory=lambda:np.zeros(1))
    previous_states_ensemble: List[np.array] = field(default_factory=list)
    previous_words: Optional[List[int]] = None
    attentions: List[List[np.array]] = field(default_factory=lambda:[[]])

    @classmethod
    def make_empty(cls, xp, ensemble_size):
        obj = cls(scores=xp.zeros(1), previous_states_ensemble = [None] * ensemble_size)
        return obj

class BSReturn(enum.Enum):
    OK = enum.auto()
    CONSTRAINT_VIOLATED = enum.auto()

@dataclass(order=False, frozen=True)
class AStarParams:
    astar_batch_size:int = 32
    astar_max_queue_size:int =1000
    astar_prune_margin:float = 10
    astar_prune_ratio:float = 10

def iterate_best_score(new_scores: np.ndarray, beam_width: int)->Iterator[Tuple[int, int, float]]:
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

    for num in six.moves.range(len(best_idx)):
        idx = best_idx[num]
        num_case = all_num_cases[num]
        idx_in_case = all_idx_in_cases[num]
        yield int(num_case), idx_in_case, new_costs_flattened[idx]


def iterate_eos_scores(new_scores, eos_idx)->Iterator[Tuple[int, int, float]]:
    nb_cases, v_size = new_scores.shape

    for num_case in six.moves.range(nb_cases):
        idx_in_case = eos_idx
        yield int(num_case), idx_in_case, cuda.to_cpu(new_scores[num_case, eos_idx])

def update_next_lists(num_case, idx_in_case, new_cost, eos_idx, new_state_ensemble, finished_translations, current_translations,
                      current_attentions,
                      next_states_list, next_words_list, next_score_list, next_normalized_score_list, next_translations_list,
                      attn_ensemble, next_attentions_list, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, 
                      need_attention=False,
                      constraints_fn=None) -> BSReturn:
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

        if constraints_fn is not None and constraints_fn(current_translations[num_case]) != 1:
            return BSReturn.CONSTRAINT_VIOLATED

        if need_attention:
            finished_translations.append((current_translations[num_case],
                                          -new_cost,
                                          current_attentions[num_case]
                                          ))
        else:
            finished_translations.append((current_translations[num_case],
                                          -new_cost))

        return BSReturn.OK

    else:
        new_translation = current_translations[num_case]+ [idx_in_case]
        if constraints_fn is not None and constraints_fn(new_translation) <0:
            return BSReturn.CONSTRAINT_VIOLATED

        next_states_list.append(
            [tuple([Variable(substates.data[num_case].reshape(1, -1)) for substates in new_state])
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

        next_translations_list.append(new_translation)

        if need_attention:
            xp = cuda.get_array_module(attn_ensemble[0].data)
            attn_summed = xp.zeros((attn_ensemble[0].data[0].shape), dtype=xp.float32)
            for attn in attn_ensemble:
                attn_summed += attn.data[num_case]
            attn_summed /= len(attn_ensemble)
            next_attentions_list.append(current_attentions[num_case] + [attn_summed])
        return BSReturn.OK


def compute_next_lists(new_state_ensemble, new_scores, beam_width, beam_pruning_margin,
                       beam_score_length_normalization, beam_score_length_normalization_strength,
                       beam_score_coverage_penalty, beam_score_coverage_penalty_strength,
                       eos_idx,
                       current_translations,
                       finished_translations,
                       current_attentions,
                       attn_ensemble,
                       force_finish=False,
                       need_attention=False,
                       constraints_fn=None):
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
                          attn_ensemble, next_attentions_list, beam_score_coverage_penalty, beam_score_coverage_penalty_strength, 
                          need_attention=need_attention, constraints_fn=constraints_fn)
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
        states_logits_attn_ensemble = [dec_cell(states, current_words) for (dec_cell, states) in six.moves.zip(
            dec_cell_ensemble, current_states_ensemble)]
    else:
        assert all(x is None for x in current_states_ensemble)
        states_logits_attn_ensemble = [dec_cell.get_initial_logits(1) for dec_cell in dec_cell_ensemble]

    new_state_ensemble, logits_ensemble, attn_ensemble = list(six.moves.zip(*states_logits_attn_ensemble))

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


def advance_one_step(dec_cell_ensemble, eos_idx, 
                     current_translations_states: ATranslationState,
                     beam_width, beam_pruning_margin,
                     beam_score_length_normalization,
                     beam_score_length_normalization_strength,
                     beam_score_coverage_penalty,
                     beam_score_coverage_penalty_strength,
                     finished_translations,
                     force_finish=False, need_attention=False,
                     prob_space_combination=False,
                     constraints_fn=None) -> Optional[ATranslationState]:
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
    current_translations, current_scores, current_states_ensemble, current_words, current_attentions = dataclasses.astuple(
                                                                                    current_translations_states)

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
        current_attentions, attn_ensemble, force_finish=force_finish, need_attention=need_attention,
        constraints_fn=constraints_fn)

    if len(next_states_list) == 0:
        return None  # We only found finished translations

    # Create the new translation states

    next_words_array = np.array(next_words_list, dtype=np.int32)
    if xp is not np:
        next_words_array = cuda.to_gpu(next_words_array)

    concatenated_next_states_list = []
    for next_states_list_one_model in six.moves.zip(*next_states_list):
        concatenated_next_states_list.append(
            tuple([F.concat(substates, axis=0) for substates in six.moves.zip(*next_states_list_one_model)])
        )

    next_translations_states = ATranslationState(next_translations_list,
                                xp.array(next_score_list),
                                concatenated_next_states_list,
                                Variable(next_words_array),
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
                         prob_space_combination=False, use_unfinished_translation_if_none_found=False,
                         constraints_fn=None,
                         use_astar: bool = False,
                         astar_params:AStarParams = AStarParams()):
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
        constraints_fn: function for enforcing constraints on translations. takes a translation as input. Return 1 if all constraints
                        are satisfied. -1 if constraints cannot be satisfied. 0<=v<1 if all constraints are not satisfied yet but can 
                        be satisfied (v being an hint on the number of constraints solved)
    Return:
        list of translations
            each item in the list is a tuple (translation, score) or (translation, score, attention) if need_attention = True
    """
    
    if use_astar:
        return ensemble_astar_search(
                         model_ensemble=model_ensemble, 
                         src_batch=src_batch, 
                         src_mask=src_mask,
                         nb_steps=nb_steps, 
                         eos_idx=eos_idx,
                         beam_width=beam_width, beam_pruning_margin=beam_pruning_margin,
                         beam_score_length_normalization=beam_score_length_normalization,
                         beam_score_length_normalization_strength=beam_score_length_normalization_strength,
                         beam_score_coverage_penalty=beam_score_coverage_penalty,
                         beam_score_coverage_penalty_strength=beam_score_coverage_penalty_strength,
                         need_attention=need_attention,
                         force_finish=force_finish,
                         prob_space_combination=prob_space_combination, 
                         use_unfinished_translation_if_none_found=use_unfinished_translation_if_none_found,
                         constraints_fn=constraints_fn,
                         astar_params=astar_params)

    with chainer.using_config("train", False), chainer.no_backprop_mode():
        mb_size = src_batch[0].data.shape[0]
        assert len(model_ensemble) >= 1
        xp = model_ensemble[0].xp
    
        dec_cell_ensemble = [model.give_conditionalized_cell(src_batch, src_mask, noise_on_prev_word=False,
                                                             demux=True) for model in model_ensemble]
    
        assert mb_size == 1
        # TODO: if mb_size == 1 then src_mask value unnecessary -> remove?
    
        finished_translations = []
    
        # Create the initial Translation state
        #previous_states_ensemble = [None] * len(model_ensemble)

        # Current_translations_states will hold the information for the current beam

        current_translations_states = ATranslationState.make_empty(xp, len(model_ensemble))
        #current_translations_states.previous_states_ensemble = previous_states_ensemble
        #current_translations_states.scores = xp.array([0])

        # current_translations_states = (
        #     [[]],  # translations
        #     xp.array([0]),  # scores
        #     previous_states_ensemble,  # previous states
        #     None,  # previous words
        #     [[]]  # attention
        # )
    
        # Proceed with the search
        for num_step in six.moves.range(nb_steps):
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
                prob_space_combination=prob_space_combination,
                constraints_fn=constraints_fn)
    
            if current_translations_states is None:
                break
    
    #     print(finished_translations, need_attention)
    
        # Return finished translations
        if len(finished_translations) == 0:
            if use_unfinished_translation_if_none_found:
                assert current_translations_states is not None
                if need_attention:
                    finished_translations.append(
                        (current_translations_states.translations[0], 
                        current_translations_states.scores[0], 
                        current_translations_states.attentions[0]))
                else:
                    finished_translations.append(
                        (current_translations_states.translations[0], 
                        current_translations_states.scores[0]))
            else:
                if need_attention:
                    finished_translations.append(([], 0, []))
                else:
                    finished_translations.append(([], 0))
        return finished_translations

###################
# Astar

@dataclass
class Item:
    score: float = 0
    state: Tuple[np.ndarray, ...] = ()
    last_word: Optional[int] = None
    current_translation: List[int] = field(default_factory=list)
    attention: Optional[List[np.array]] = field(default_factory=list)

    def __repr__(self):
        state_repr = []
        for x in self.state:
            state_repr.append("<")
            for y in x:
                state_repr.append(str(y.shape))
            state_repr.append(">")
        return f"Item[SC:{self.score:2f}, T:{self.current_translation}, LW:{self.last_word}, ST:{''.join(state_repr)}]"



@dataclass(order=True)
class PItem:
    priority: float
    item: Item = field(compare=False)

class TranslationPriorityQueue:
    def __init__(self):
        #self.queue = queue.PriorityQueue()
        self.queue :List[PItem] = []
        self.dirty = False

    def __repr__(self):
        content = []
        for pitem in self.queue:
            content.append(f"P:{pitem.priority:2f}  {repr(pitem.item)}")
        return f"L:{len(self.queue)} Dirty:{self.dirty}\n"+"\n".join(content)

    def prune_queue(self, ratio = None, margin = None, top_n = None):
        if len(self.queue) == 0:
            return

        initial_length = len(self.queue)
        if self.dirty:
            self.sort()

        if top_n is not None:
            self.queue = self.queue[:top_n]

        max_priority = float("-inf")
        min_priority = float("-inf")
        threshold1 = float("-inf")
        threshold2 = float("-inf")
        threshold = float("-inf")
        threshold_index = float("-inf")
        if ratio is not None or margin is not None:
            max_priority = self.queue[0].priority
            min_priority = self.queue[-1].priority
            if ratio is not None:
                threshold1 = max_priority + max_priority*ratio
            if margin is not None:
                threshold2 = max_priority - margin
            if threshold1 is not None and threshold2 is not None:
                threshold = max(threshold1, threshold2)
            elif threshold1 is not None:
                threshold = threshold1
            elif threshold2 is not None:
                threshold = threshold2
            
            if threshold > min_priority:
                threshold_index = bisect.bisect([-item.priority for item in self.queue], -threshold)
                self.queue = self.queue[:threshold_index]
        final_length = len(self.queue)
        #print(f"pruned {initial_length} -> {final_length} maxp:{max_priority:2f} minp:{min_priority:2f} th1:{threshold1:2f} th2:{threshold2:2f} th:{threshold:2f} thi:{threshold_index}")

    def put(self, item:Item, priority:float)->None:
        #self.queue.put(item)
        assert priority <= 0
        self.queue.append(PItem(priority, item))
        self.dirty = True

    def sort(self):
        self.queue.sort(reverse=True, key=operator.attrgetter("priority"))
        self.dirty = False

    def get_n(self, n:int) ->List[Item]:
        if self.dirty:
            self.sort()

        res = [x.item for x in self.queue[:n]]
        self.queue = self.queue[n:]
        return res
        # res = []
        # while not self.queue.empty() and len(res) < n:
        #     p_item = self.queue.get() 
        #     res.append(p_item.item)
        # return res


def make_item_list(next_states_list, next_words_list, 
                next_score_list, next_translations_list, next_attentions_list)->List[Item]:
    res = []
    for num_item in range(len(next_states_list)):
        next_translations_list[num_item]
        next_score_list[num_item]
        next_words_list[num_item]
        next_states_list[num_item]
        next_attentions_list[num_item]
        new_item = Item(score = next_score_list[num_item], state = next_states_list[num_item],
                        last_word=next_words_list[num_item], current_translation = next_translations_list[num_item],
                        attention=next_attentions_list[num_item])
        res.append(new_item)
    return res

def merge_items_into_TState(items_list: List[Item], xp) -> ATranslationState:
    if len(items_list) > 1:
        assert all(item.last_word is not None for item in items_list)
        assert all(item.state is not None for item in items_list)

    for item in items_list:
        assert item.last_word is not None

    translations = [item.current_translation for item in items_list]
    scores = xp.array([item.score for item in items_list])

    if len(items_list) == 1 and items_list[0].last_word is None:
        next_words_array = None
    else:
        last_words = cast(List[int], [item.last_word for item in items_list])

        next_words_array = np.array(last_words, dtype=np.int32)
        if xp is not np:
            next_words_array = cuda.to_gpu(next_words_array)

    attentions = [item.attention for item in items_list]

    next_states_list = cast(List[Tuple[np.ndarray, ...]], [item.state for item in items_list])
    concatenated_next_states_list: List[Tuple[np.ndarray,...]]= []
    for next_states_list_one_model in zip(*next_states_list):
        concatenated_next_states_list.append(
            tuple([F.concat(substates, axis=0) for substates in zip(*next_states_list_one_model)])
        )

    return ATranslationState(translations=translations, scores=scores, 
            previous_states_ensemble= concatenated_next_states_list, 
            previous_words = next_words_array, attentions = attentions)



def astar_update(dec_cell_ensemble, eos_idx, 
                     translations_priority_queue: TranslationPriorityQueue,
                     beam_width, beam_pruning_margin,
                     beam_score_length_normalization,
                     beam_score_length_normalization_strength,
                     beam_score_coverage_penalty,
                     beam_score_coverage_penalty_strength,
                     finished_translations,
                     force_finish=False, need_attention=False,
                     prob_space_combination=False,
                     constraints_fn=None,
                     astar_params:AStarParams = AStarParams()) -> bool:
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



    translations_priority_queue.prune_queue(
                                ratio = astar_params.astar_prune_ratio, 
                                margin = astar_params.astar_prune_margin, 
                                top_n = astar_params.astar_max_queue_size)
    items = translations_priority_queue.get_n(astar_params.astar_batch_size)

    if len(items) == 0:
        log.info("astar queue got empty early")
        return False

    xp = dec_cell_ensemble[0].xp

    current_translations_states = merge_items_into_TState(items, xp)


#     xp = cuda.get_array_module(dec_ensemble[0].initial_state.data)
    
    current_translations, current_scores, current_states_ensemble, current_words, current_attentions = dataclasses.astuple(
                                                                                    current_translations_states)

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
        current_attentions, attn_ensemble, force_finish=force_finish, need_attention=need_attention,
        constraints_fn=constraints_fn)

    if len(next_states_list) == 0:
        return False  # We only found finished translations

    # Create the new translation states

    next_words_array = np.array(next_words_list, dtype=np.int32)
    if xp is not np:
        next_words_array = cuda.to_gpu(next_words_array)

    # concatenated_next_states_list = []
    # for next_states_list_one_model in six.moves.zip(*next_states_list):
    #     concatenated_next_states_list.append(
    #         tuple([F.concat(substates, axis=0) for substates in six.moves.zip(*next_states_list_one_model)])
    #     )



    
    # next_translations_states = ATranslationState(next_translations_list,
    #                             xp.array(next_score_list),
    #                             concatenated_next_states_list,
    #                             Variable(next_words_array),
    #                             next_attentions_list
    #                             )
    item_list = make_item_list(next_states_list, next_words_list, 
                next_score_list, next_translations_list, next_attentions_list)
    #print("adding", len(item_list), "items")
    for item in item_list:
        translations_priority_queue.put(item, item.score/(1 + len(item.current_translation)))

    return True


def ensemble_astar_search(model_ensemble, src_batch, src_mask, nb_steps, eos_idx,
                         beam_width=20, beam_pruning_margin=None,
                         beam_score_length_normalization=None,
                         beam_score_length_normalization_strength=0.2,
                         beam_score_coverage_penalty=None,
                         beam_score_coverage_penalty_strength=0.2,
                         need_attention=False,
                         force_finish=False,
                         prob_space_combination=False, use_unfinished_translation_if_none_found=False,
                         constraints_fn=None,
                         astar_params:AStarParams = AStarParams()):
    """
    Compute translations using a astar-search algorithm.

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
        constraints_fn: function for enforcing constraints on translations. takes a translation as input. Return 1 if all constraints
                        are satisfied. -1 if constraints cannot be satisfied. 0<=v<1 if all constraints are not satisfied yet but can 
                        be satisfied (v being an hint on the number of constraints solved)
    Return:
        list of translations
            each item in the list is a tuple (translation, score) or (translation, score, attention) if need_attention = True
    """
    
    with chainer.using_config("train", False), chainer.no_backprop_mode():
        mb_size = src_batch[0].data.shape[0]
        assert len(model_ensemble) >= 1
        xp = model_ensemble[0].xp
    
        dec_cell_ensemble = [model.give_conditionalized_cell(src_batch, src_mask, noise_on_prev_word=False,
                                                             demux=True) for model in model_ensemble]
    
        assert mb_size == 1
        # TODO: if mb_size == 1 then src_mask value unnecessary -> remove?
    
        finished_translations = []
    
        # Create the initial Translation state
        #previous_states_ensemble = [None] * len(model_ensemble)

        # Current_translations_states will hold the information for the current beam

        astar_queue = TranslationPriorityQueue()    
        astar_queue.put(Item(), 0)



        #current_translations_states = ATranslationState.make_empty(xp, len(model_ensemble))
        #current_translations_states.previous_states_ensemble = previous_states_ensemble
        #current_translations_states.scores = xp.array([0])

        # current_translations_states = (
        #     [[]],  # translations
        #     xp.array([0]),  # scores
        #     previous_states_ensemble,  # previous states
        #     None,  # previous words
        #     [[]]  # attention
        # )
        
        # Proceed with the search
        for num_step in six.moves.range(nb_steps):
            #breakpoint()
            #print("num_step len", num_step, len(astar_queue.queue))
            still_options_to_explore = astar_update(
                dec_cell_ensemble,
                eos_idx,
                astar_queue,
                beam_width,
                beam_pruning_margin,
                beam_score_length_normalization,
                beam_score_length_normalization_strength,
                beam_score_coverage_penalty,
                beam_score_coverage_penalty_strength,
                finished_translations,
                force_finish=force_finish and num_step == (nb_steps - 1),
                need_attention=need_attention,
                prob_space_combination=prob_space_combination,
                constraints_fn=constraints_fn,
                astar_params=astar_params)
            if not still_options_to_explore:
                break
    
            #if current_translations_states is None:
            #    break
    
    #     print(finished_translations, need_attention)
    
        # Return finished translations
        if len(finished_translations) == 0:
            log.info("no finished translation found")
            if use_unfinished_translation_if_none_found:
                current_translations_states = merge_items_into_TState(astar_queue.get_n(1), xp)
                
                #assert current_translations_states is not None
                if need_attention:
                    finished_translations.append(
                        (current_translations_states.translations[0], 
                        current_translations_states.scores[0], 
                        current_translations_states.attentions[0]))
                else:
                    finished_translations.append(
                        (current_translations_states.translations[0], 
                        current_translations_states.scores[0]))
            else:
                if need_attention:
                    finished_translations.append(([], 0, []))
                else:
                    finished_translations.append(([], 0))
        return finished_translations
