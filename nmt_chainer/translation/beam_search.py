#!/usr/bin/env python
"""beam_search.py: Implementation of beam_search"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import bisect
import dataclasses
import enum
import itertools
import logging
import operator
import queue
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (Callable, Counter, Dict, Iterator, List, Mapping, Optional,
                    Tuple, Union, cast)

import chainer
import chainer.functions as F
import chainer.links as L
import chainerx
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

@dataclass(order=False, frozen=True)
class BeamSearchParams:
    beam_width:int = 20
    beam_pruning_margin:Optional[float] = None
    beam_score_length_normalization:str = "none" # none simple google
    beam_score_length_normalization_strength:float = 0.2
    beam_score_coverage_penalty:str = "none" # none google
    beam_score_coverage_penalty_strength: float=0.2
    force_finish:bool = False
    use_unfinished_translation_if_none_found:bool = False
    always_consider_eos_and_placeholders:bool = False

class TgtIdxConstraint:
    def __init__(self):
        self.dict = {}
        self.placeholders_list: Optional[List[int]] = None

    def __contains__(self, x:int)->bool:
        return x in self.dict

    def set_placeholders_idx_list(self, lst:List[int]):
        self.placeholders_list = lst

    def add(self, x:int)->None:
        if not isinstance(x, int):
            raise Exception("wrong type %s"%x)
        if x not in self.dict:
            self.dict[x] = 1
        else:
            self.dict[x] += 1

    def substract(self, x:int)->None:
        if x not in self.dict:
            raise Exception("trying to remove non existant %r"%x)
        self.dict[x] -= 1
        assert self.dict[x] >= 0
        if self.dict[x] == 0:
            del self.dict[x]

    def __iter__(self):
        return self.dict.__iter__()

    def copy(self)->"TgtIdxConstraint":
        res = TgtIdxConstraint()
        res.dict = self.dict.copy()
        res.placeholders_list = self.placeholders_list
        return res

    def __len__(self)->int:
        return sum(self.dict.values())
    def __repr__(self):
        return repr(self.dict)
    def __str__(self):
        return str(self.dict)  

@dataclass(order=False, frozen=True)
class BeamSearchConstraints:
    constraint_fn: Optional[Callable[[List[int]], float]] = None
    required_tgt_idx: Optional[TgtIdxConstraint] = None

@dataclass(order=False, frozen=True)
class AStarParams:
    astar_batch_size:int = 32
    astar_max_queue_size:int =1000
    astar_prune_margin:float = 10
    astar_prune_ratio:Optional[float] = None
    length_normalization_constant:float = 0
    length_normalization_exponent:float = 1
    astar_priority_eval_string: Optional[str] = None
    max_length_diff:Optional[int]=None


@dataclass(eq = False)
class ATranslationState:
    translations: List[List[int]] = field(default_factory=lambda:[[]])
    scores: np.array = field(default_factory=lambda:np.zeros(1))
    previous_states_ensemble: List[np.array] = field(default_factory=list)
    previous_words: Optional[List[int]] = None
    attentions: List[List[np.array]] = field(default_factory=lambda:[[]])
    required_tgt_idx_list: Optional[List[TgtIdxConstraint]] = None

    @classmethod
    def make_empty(cls, xp, ensemble_size, required_tgt_idx:Optional[Counter[int]], gpu=None):
        required_tgt_idx_list = None if required_tgt_idx is None else [required_tgt_idx]
        if gpu is not None and xp == chainerx:
            scores=xp.zeros(1, device="cuda:%i"%gpu)
        else:
            scores=xp.zeros(1)

        obj = cls(scores=scores, 
                previous_states_ensemble = [None] * ensemble_size,
                required_tgt_idx_list=required_tgt_idx_list)
        return obj

class BSReturn(enum.Enum):
    OK = enum.auto()
    CONSTRAINT_VIOLATED = enum.auto()


@dataclass
class TranslationInfosList:
    next_states_list: Union[List[List[Tuple[Variable]]], List[int]] = field(default_factory = list)  # each item is a list of list
    next_words_list: List[int] = field(default_factory = list)
    next_score_list: List[float] = field(default_factory = list)
    next_normalized_score_list: Optional[List[float]] = None #field(default_factory = list)
    next_translations_list: List[List[int]] = field(default_factory = list)
    next_attentions_list: List[List[np.ndarray]] = field(default_factory = list)
    constraint_values: Optional[List[float]] = None
    required_tgt_idx_list: Optional[List[TgtIdxConstraint]] = None

    def prune_with_margin(self, beam_pruning_margin:float, use_normalized_score:bool):
        if use_normalized_score:
            assert self.next_normalized_score_list is not None
            score_list = self.next_normalized_score_list
        else:
            score_list = self.next_score_list
        best_next_score = max(score_list)
        bad_score_indices = [
            idx for idx,
            elem in enumerate(score_list) if (
                best_next_score -
                elem > beam_pruning_margin)]
        self.prune_from_index_list(bad_score_indices)

    def prune_from_index_list(self, bad_score_indices:List[int]):
        for i in bad_score_indices[::-1]:
            if self.next_normalized_score_list is not None:
                assert len(self.next_normalized_score_list) == len(self.next_score_list)
                del self.next_normalized_score_list[i]
            if self.constraint_values is not None:
                assert len(self.constraint_values) == len(self.next_score_list)
                del self.constraint_values[i]
            if self.required_tgt_idx_list is not None:
                assert len(self.required_tgt_idx_list) == len(self.next_score_list)
                del self.required_tgt_idx_list[i]
            del self.next_states_list[i]
            del self.next_words_list[i]
            del self.next_score_list[i]
            del self.next_translations_list[i]
            del self.next_attentions_list[i]
        assert (len(self.next_states_list) == len(self.next_words_list) == 
                        len(self.next_score_list) == len(self.next_translations_list) == len(self.next_attentions_list)
                        )

def iterate_best_score(new_scores: np.ndarray, beam_width: int, xp=np)->Iterator[Tuple[int, int, float]]:
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

    if xp == chainerx:
        xp, device, new_scores = chainerx._fallback_workarounds._from_chx(new_scores)

    if xp != np:
        new_costs_flattened_non_neg = new_scores.ravel()
        best_idx = xp.argsort(new_costs_flattened_non_neg)[-beam_width:]
        best_scores = -new_costs_flattened_non_neg[best_idx]
        all_num_cases = best_idx / v_size
        all_idx_in_cases = best_idx % v_size
        best_scores = chainer.cuda.to_cpu(best_scores)
        all_num_cases = chainer.cuda.to_cpu(all_num_cases)
        all_idx_in_cases = chainer.cuda.to_cpu(all_idx_in_cases)
    else:
        new_costs_flattened = (- new_scores).ravel()
        
        # TODO replace wit a cupy argpartition when/if implemented
        best_idx = np.argpartition(new_costs_flattened, beam_width)[:beam_width]
        best_scores = new_costs_flattened[best_idx]

        all_num_cases = best_idx / v_size
        all_idx_in_cases = best_idx % v_size

    for num in six.moves.range(len(best_idx)):
        #idx = best_idx[num]
        num_case = all_num_cases[num]
        idx_in_case = all_idx_in_cases[num]
        yield int(num_case), idx_in_case, best_scores[num] #new_costs_flattened[idx]


def iterate_eos_scores(new_scores, eos_idx)->Iterator[Tuple[int, int, float]]:
    nb_cases, v_size = new_scores.shape

    for num_case in six.moves.range(nb_cases):
        idx_in_case = eos_idx
        yield int(num_case), idx_in_case, -cuda.to_cpu(new_scores[num_case, eos_idx])

def iterate_required_word_scores(new_scores, required:List[TgtIdxConstraint])->Iterator[Tuple[int, int, float]]:
    nb_cases, v_size = new_scores.shape
    assert nb_cases == len(required)

    for num_case in six.moves.range(nb_cases):
        for req_idx in required[num_case]:
            yield int(num_case), req_idx, -cuda.to_cpu(new_scores[num_case, req_idx])

def update_next_lists(num_case, idx_in_case, new_cost, eos_idx, get_slice_of_new_state_ensemble, finished_translations, current_translations,
                      current_attentions,
                      t_infos_list: TranslationInfosList,
                      #next_states_list, next_words_list, next_score_list, next_normalized_score_list, next_translations_list,
                      attn_ensemble, 
                      #next_attentions_list, 
                      beam_score_coverage_penalty, beam_score_coverage_penalty_strength, 
                      need_attention=False,
                      constraints_fn=None,
                      required_tgt_idx:Optional[TgtIdxConstraint]=None,
                      xp=np) -> BSReturn:
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

        if required_tgt_idx is not None and len(required_tgt_idx) > 0:
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
        if required_tgt_idx is not None:
            if idx_in_case in required_tgt_idx.placeholders_list:
                if idx_in_case in required_tgt_idx:
                    required_tgt_idx = required_tgt_idx.copy()
                    required_tgt_idx.substract(idx_in_case)
                else:
                    #assert required_tgt_idx[idx_in_case] == 0
                    return BSReturn.CONSTRAINT_VIOLATED


        new_translation = current_translations[num_case]+ [idx_in_case]
        if constraints_fn is not None:
            constraint_val = constraints_fn(new_translation)
            assert t_infos_list.constraint_values is not None
            t_infos_list.constraint_values.append(constraint_val)
        else:
            constraint_val = None

        if constraint_val is not None and constraint_val <0:
            return BSReturn.CONSTRAINT_VIOLATED

        if get_slice_of_new_state_ensemble is not None:
            t_infos_list.next_states_list.append(get_slice_of_new_state_ensemble(num_case))
        else:
            t_infos_list.next_states_list.append(num_case)

        #     [tuple([Variable(substates.data[num_case].reshape(1, -1)) for substates in new_state])
        #      for new_state in new_state_ensemble]
        # )

        t_infos_list.next_words_list.append(idx_in_case)
        t_infos_list.next_score_list.append(-new_cost)

        if required_tgt_idx is not None:
            assert t_infos_list.required_tgt_idx_list is not None
            t_infos_list.required_tgt_idx_list.append(required_tgt_idx)

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
            t_infos_list.next_normalized_score_list.append(normalized_score)

        t_infos_list.next_translations_list.append(new_translation)

        if need_attention:
            #xp = cuda.get_array_module(attn_ensemble[0].data)
            #attn_summed = xp.zeros((attn_ensemble[0].data[0].shape), dtype=xp.float32)
            if len(attn_ensemble) == 1:
                attn_summed = attn_ensemble[0].array[num_case]
            else:
                attn_summed = attn_ensemble[0].array[num_case].copy() 
                for attn in attn_ensemble[1:]:
                    attn_summed += attn.array[num_case]
                attn_summed /= len(attn_ensemble)
            t_infos_list.next_attentions_list.append(current_attentions[num_case] + [attn_summed])
        return BSReturn.OK



def compute_next_lists(new_state_ensemble, new_scores, 
                       beam_search_params: BeamSearchParams,
                       #beam_width, beam_pruning_margin,
                       #beam_score_length_normalization, beam_score_length_normalization_strength,
                       #beam_score_coverage_penalty, beam_score_coverage_penalty_strength,
                       eos_idx,
                       current_translations,
                       finished_translations,
                       current_attentions,
                       attn_ensemble,
                       force_finish=False,
                       need_attention=False,
                       constraints_fn=None,
                       required_tgt_idx_list:Optional[List[Counter[int]]]=None,
                       xp=np) -> TranslationInfosList:
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
    # next_states_list = []  # each item is a list of list
    # next_words_list = []
    # next_score_list = []
    # next_normalized_score_list = []
    # next_translations_list = []
    # next_attentions_list = []

    
    if beam_search_params.beam_score_coverage_penalty == "google":
        t_infos_list = TranslationInfosList(next_normalized_score_list = [])
    else:
        t_infos_list = TranslationInfosList()

    if constraints_fn is not None:
        t_infos_list.constraint_values = []

    if required_tgt_idx_list is not None:
        t_infos_list.required_tgt_idx_list = []

    if force_finish:
        score_iterator = iterate_eos_scores(new_scores, eos_idx)
    elif beam_search_params.always_consider_eos_and_placeholders:
        score_iterator_list = [iterate_best_score(new_scores, beam_search_params.beam_width, xp=xp),
                                   iterate_eos_scores(new_scores, eos_idx) ]
        if required_tgt_idx_list is not None:
            score_iterator_list.append(iterate_required_word_scores(new_scores, required_tgt_idx_list))

        def chained_score_iterator():
            already_seen = set()
            for num_case, idx_in_case, new_cost in itertools.chain(*score_iterator_list):
                if (num_case, idx_in_case) in already_seen:
                    continue
                already_seen.add((num_case, idx_in_case))
                yield num_case, idx_in_case, new_cost
        score_iterator = chained_score_iterator()
    else:
        score_iterator = iterate_best_score(new_scores, beam_search_params.beam_width, xp=xp)

    if new_state_ensemble is not None:
        memoized_state_ensemble_slices = {}
        def get_slice_of_new_state_ensemble(num_case):
            if num_case not in memoized_state_ensemble_slices:
                memoized_state_ensemble_slices[num_case] = [
                    tuple([substates.data[num_case].reshape(1, -1) for substates in new_state])
                                    for new_state in new_state_ensemble]
            return memoized_state_ensemble_slices[num_case]
    else:
        get_slice_of_new_state_ensemble = None

    for num_case, idx_in_case, new_cost in score_iterator:
        if len(current_translations[num_case]) > 0:
            if beam_search_params.beam_score_length_normalization == 'simple':
                new_cost /= len(current_translations[num_case])
            elif beam_search_params.beam_score_length_normalization == 'google':
                new_cost /= (pow((len(current_translations[num_case]) + 5), 
                    beam_search_params.beam_score_length_normalization_strength) / 
                    pow(6, beam_search_params.beam_score_length_normalization_strength))
        
        required_tgt_idx=required_tgt_idx_list[num_case] if required_tgt_idx_list is not None else None

        

        update_next_lists(num_case, idx_in_case, new_cost, eos_idx, get_slice_of_new_state_ensemble,
                          finished_translations, current_translations, current_attentions,
                          t_infos_list,
                          #next_states_list, next_words_list, next_score_list, next_normalized_score_list, next_translations_list,
                          attn_ensemble, 
                          #next_attentions_list, 
                          beam_search_params.beam_score_coverage_penalty, 
                          beam_search_params.beam_score_coverage_penalty_strength, 
                          need_attention=need_attention, constraints_fn=constraints_fn,
                          required_tgt_idx=required_tgt_idx,
                          xp=xp)
        assert len(t_infos_list.next_states_list) <= beam_search_params.beam_width or beam_search_params.always_consider_eos_and_placeholders
#             if len(next_states_list) >= beam_width:
#                 break


    # Prune items that have a score worse than beam_pruning_margin below the
    # best score.
    if (beam_search_params.beam_pruning_margin is not None and t_infos_list.next_score_list):
        t_infos_list.prune_with_margin(beam_search_params.beam_pruning_margin, use_normalized_score=False)


    # Prune items that have a normalized score worse than beam_pruning_margin
    # below the best normalized score.
    if (beam_search_params.beam_score_coverage_penalty ==
            "google" and beam_search_params.beam_pruning_margin is not None and t_infos_list.next_normalized_score_list is not None):

        t_infos_list.prune_with_margin(beam_search_params.beam_pruning_margin, use_normalized_score=True)



    return t_infos_list #next_states_list, next_words_list, next_score_list, next_translations_list, next_attentions_list


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
    #combined_scores = xp.zeros((logits_ensemble[0].data.shape), dtype=xp.float32)
    combined_scores = None
    if not prob_space_combination:
        for logits in logits_ensemble:
            if combined_scores is None:
                combined_scores = F.log_softmax(logits).data
            else:
                combined_scores += F.log_softmax(logits).data #xp.log(F.softmax(logits).data)
        combined_scores /= len(dec_cell_ensemble)
    else:
        for logits in logits_ensemble:
            if combined_scores is None:
                combined_scores = F.softmax(logits).data
            else:
                combined_scores += F.softmax(logits).data
        combined_scores /= len(dec_cell_ensemble)
        combined_scores = xp.log(combined_scores)


    #print(combined_scores[0,0]) #force sync

    return combined_scores, new_state_ensemble, attn_ensemble



def advance_one_step(dec_cell_ensemble, eos_idx, 
                     current_translations_states: ATranslationState,
                     beam_search_params:BeamSearchParams,
                     #beam_width, beam_pruning_margin,
                     #beam_score_length_normalization,
                     #beam_score_length_normalization_strength,
                     #beam_score_coverage_penalty,
                     #beam_score_coverage_penalty_strength,
                     finished_translations,
                     force_finish=False, need_attention=False,
                     prob_space_combination=False,
                     constraints_fn=None,
                     gpu=None) -> Optional[ATranslationState]:
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
    #current_translations, current_scores, current_states_ensemble, current_words, current_attentions, required_tgt_idx_list = dataclasses.astuple(
    #                                                                                current_translations_states)

    current_translations = current_translations_states.translations
    current_scores = current_translations_states.scores
    current_states_ensemble = current_translations_states.previous_states_ensemble
    current_words = current_translations_states.previous_words
    current_attentions = current_translations_states.attentions
    required_tgt_idx_list = current_translations_states.required_tgt_idx_list

    # Compute the next states and associated next word scores
    combined_scores, new_state_ensemble, attn_ensemble = compute_next_states_and_scores(
        dec_cell_ensemble, current_states_ensemble, current_words,
        prob_space_combination=prob_space_combination)

    nb_cases, v_size = combined_scores.shape
    assert nb_cases <= beam_search_params.beam_width

    # Add the new scores to the previous ones for each states in the beam
    new_scores = current_scores[:, xp.newaxis] + combined_scores

    # Compute the list of new translation states after pruning
    #next_states_list, next_words_list, next_score_list, next_translations_list, next_attentions_list 
    t_infos_list = compute_next_lists(
        None, new_scores, 
        beam_search_params,
        #beam_width, beam_pruning_margin,
        #beam_score_length_normalization, beam_score_length_normalization_strength,
        #beam_score_coverage_penalty, beam_score_coverage_penalty_strength,
        eos_idx,
        current_translations, finished_translations,
        current_attentions, attn_ensemble, force_finish=force_finish, need_attention=need_attention,
        constraints_fn=constraints_fn,
        required_tgt_idx_list=required_tgt_idx_list,
        xp=xp)

    if len(t_infos_list.next_states_list) == 0:

        return None  # We only found finished translations

    # Create the new translation states
    if xp == np:
        next_words_array = np.array(t_infos_list.next_words_list, dtype=np.int32)
        next_score_array = np.array(t_infos_list.next_score_list, dtype=np.float32)
    elif xp == chainerx:
        if gpu is None:
            next_words_array = xp.array(t_infos_list.next_words_list, dtype=xp.int32)
            next_score_array = xp.array(t_infos_list.next_score_list, dtype=xp.float32)
        else:
            next_words_array = xp.array(t_infos_list.next_words_list, dtype=xp.int32, device="cuda:%i"%gpu)
            next_score_array = xp.array(t_infos_list.next_score_list, dtype=np.float32, device="cuda:%i"%gpu)
    else:
        next_words_array = np.array(t_infos_list.next_words_list, dtype=np.int32)
        next_words_array = cuda.to_gpu(next_words_array)

        next_score_array = np.array(t_infos_list.next_score_list, dtype=np.float32)
        next_score_array = cuda.to_gpu(next_score_array)


    # concatenated_next_states_list = []
    # for next_states_list_one_model in six.moves.zip(*t_infos_list.next_states_list):
    #     concatenated_next_states_list.append(
    #         tuple([Variable(xp.concatenate(substates, axis=0)) for substates in six.moves.zip(*next_states_list_one_model)])
    #     )

    concatenated_next_states_list = []
    for next_states_list_one_model in new_state_ensemble:
        concatenated_next_states_list.append([])
        for substates in next_states_list_one_model:
            new_substate = xp.take(substates.data, t_infos_list.next_states_list, axis=0)
            concatenated_next_states_list[-1].append(Variable(new_substate))

    next_translations_states = ATranslationState(t_infos_list.next_translations_list,
                                next_score_array,
                                concatenated_next_states_list,
                                Variable(next_words_array, requires_grad=False),
                                t_infos_list.next_attentions_list,
                                required_tgt_idx_list=t_infos_list.required_tgt_idx_list
                                )

    return next_translations_states



def ensemble_beam_search(model_ensemble, src_batch, src_mask, nb_steps, eos_idx,
                         beam_search_params:BeamSearchParams,
                         
                         #beam_width=20, beam_pruning_margin=None,
                         #beam_score_length_normalization=None,
                         #beam_score_length_normalization_strength=0.2,
                         #beam_score_coverage_penalty=None,
                         #beam_score_coverage_penalty_strength=0.2,
                         need_attention=False,
                         #force_finish=False,
                         prob_space_combination=False, 
                         #use_unfinished_translation_if_none_found=False,
                         constraints:Optional[BeamSearchConstraints] = None,
                         #constraints_fn=None,
                         use_astar: bool = False,
                         astar_params:AStarParams = AStarParams(),
                         gpu=None
                         #required_tgt_idx:Optional[List[int]] = None
                         ):
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
                         beam_search_params=beam_search_params,
                         #beam_width=beam_width, beam_pruning_margin=beam_pruning_margin,
                         #beam_score_length_normalization=beam_score_length_normalization,
                         #beam_score_length_normalization_strength=beam_score_length_normalization_strength,
                         #beam_score_coverage_penalty=beam_score_coverage_penalty,
                         #beam_score_coverage_penalty_strength=beam_score_coverage_penalty_strength,
                         need_attention=need_attention,
                         #force_finish=force_finish,
                         prob_space_combination=prob_space_combination, 
                         #use_unfinished_translation_if_none_found=use_unfinished_translation_if_none_found,
                         constraints=constraints,
                         #constraints_fn=constraints_fn,
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
        required_tgt_idx = constraints.required_tgt_idx if constraints is not None else None
        constraints_fn = constraints.constraint_fn if constraints is not None else None

        current_translations_states = ATranslationState.make_empty(xp, len(model_ensemble), 
                        required_tgt_idx=required_tgt_idx,
                        gpu=gpu)
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
                beam_search_params,
                finished_translations,
                force_finish=beam_search_params.force_finish and num_step == (nb_steps - 1),
                need_attention=need_attention,
                prob_space_combination=prob_space_combination,
                constraints_fn=constraints_fn,
                gpu=gpu)
    
            if current_translations_states is None:
                break
    
    #     print(finished_translations, need_attention)
    
        # Return finished translations
        if len(finished_translations) == 0:
            log.info(f"no finished translation found  {nb_steps}" )
            if beam_search_params.use_unfinished_translation_if_none_found:
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
    state: List[Tuple[np.ndarray, ...]] = field(default_factory=list)
    last_word: Optional[int] = None
    current_translation: List[int] = field(default_factory=list)
    attention: Optional[List[np.array]] = field(default_factory=list)
    constraint_val: Optional[float] = None
    required_tgt_idx: Optional[TgtIdxConstraint] = None

    def __repr__(self):
        state_repr = []
        for x in self.state:
            state_repr.append("<")
            for y in x:
                state_repr.append(str(y.shape))
            state_repr.append(">")
        cv_repr = "" if self.constraint_val is None else " CV:%f"%self.constraint_val
        required_str = "" if self.required_tgt_idx is None else " RT:%r"%self.required_tgt_idx
        return f"Item[SC:{self.score:2f}, T:{self.current_translation}, LW:{self.last_word}, ST:{''.join(state_repr)}{cv_repr}{required_str}]"

    def compute_priority_from_string(self, eval_string):
        return eval(eval_string)

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

    def stats(self):
        dd = defaultdict(int)
        for pitem in self.queue:
            dd[len(pitem.item.current_translation)] +=1
        return " ".join([f"{key}:{val}" for key, val in sorted(dd.items(), reverse=True)])

    def prune_queue(self, ratio = None, margin = None, top_n = None, max_length_diff = None):
        if len(self.queue) == 0:
            return

        if max_length_diff is not None:
            max_length = max(len(pitem.item.current_translation) for pitem in self.queue)
            self.queue = [pitem for pitem in self.queue if len(pitem.item.current_translation) >= max_length - max_length_diff]

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


def make_item_list(t_info_list: TranslationInfosList #next_states_list, next_words_list, 
                #next_score_list, next_translations_list, next_attentions_list
                )->List[Item]:
    res = []
    for num_item in range(len(t_info_list.next_states_list)):
        if t_info_list.constraint_values is None:
            constraint_val = None
        else:
            constraint_val = t_info_list.constraint_values[num_item]

        required_tgt_idx = None if t_info_list.required_tgt_idx_list is None else t_info_list.required_tgt_idx_list[num_item]

        new_item = Item(score = t_info_list.next_score_list[num_item], state = t_info_list.next_states_list[num_item],
                        last_word=t_info_list.next_words_list[num_item], current_translation = t_info_list.next_translations_list[num_item],
                        attention=t_info_list.next_attentions_list[num_item],
                        constraint_val=constraint_val,
                        required_tgt_idx=required_tgt_idx)
        res.append(new_item)
    return res

def merge_items_into_TState(items_list: List[Item], xp) -> ATranslationState:
    if len(items_list) > 1:
        assert all(item.last_word is not None for item in items_list)
        assert all(item.state is not None for item in items_list)

    # for item in items_list:
    #     assert item.last_word is not None #except first item...

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

    if len(items_list) > 0 and items_list[0].required_tgt_idx is None:
        assert all(item.required_tgt_idx is None for item in  items_list)
        required_tgt_idx_list = None
    else:
        required_tgt_idx_list = [item.required_tgt_idx for item in items_list]

    next_states_list = cast(List[Tuple[np.ndarray, ...]], [item.state for item in items_list])
    concatenated_next_states_list: List[Tuple[np.ndarray,...]]= []
    for next_states_list_one_model in zip(*next_states_list):
        concatenated_next_states_list.append(
            tuple([Variable(xp.concatenate(substates, axis=0)) for substates in zip(*next_states_list_one_model)])
        )

    return ATranslationState(translations=translations, scores=scores, 
            previous_states_ensemble= concatenated_next_states_list, 
            previous_words = next_words_array, attentions = attentions,
            required_tgt_idx_list=required_tgt_idx_list)



def astar_update(num_step:int, nb_steps:int, dec_cell_ensemble, eos_idx, 
                     translations_priority_queue: TranslationPriorityQueue,
                     beam_search_params:BeamSearchParams,
                     #beam_width, beam_pruning_margin,
                     #beam_score_length_normalization,
                     #beam_score_length_normalization_strength,
                     #beam_score_coverage_penalty,
                     #beam_score_coverage_penalty_strength,
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
                                top_n = astar_params.astar_max_queue_size,
                                max_length_diff = astar_params.max_length_diff)
    items = translations_priority_queue.get_n(astar_params.astar_batch_size)

    if len(items) == 0:
        log.info("astar queue got empty early")
        return False

    xp = dec_cell_ensemble[0].xp

    current_translations_states = merge_items_into_TState(items, xp)


#     xp = cuda.get_array_module(dec_ensemble[0].initial_state.data)
    
    current_translations = current_translations_states.translations
    current_scores = current_translations_states.scores
    current_states_ensemble = current_translations_states.previous_states_ensemble
    current_words = current_translations_states.previous_words
    current_attentions = current_translations_states.attentions
    required_tgt_idx_list = current_translations_states.required_tgt_idx_list

    #, current_scores, current_states_ensemble, current_words, current_attentions, required_tgt_idx_list = dataclasses.astuple(
    #                                                                                current_translations_states)

    # Compute the next states and associated next word scores
    combined_scores, new_state_ensemble, attn_ensemble = compute_next_states_and_scores(
        dec_cell_ensemble, current_states_ensemble, current_words,
        prob_space_combination=prob_space_combination)

    

    nb_cases, v_size = combined_scores.shape
    assert nb_cases <= beam_search_params.beam_width

    # Add the new scores to the previous ones for each states in the beam
    new_scores = current_scores[:, xp.newaxis] + combined_scores

    # Compute the list of new translation states after pruning
    #next_states_list, next_words_list, next_score_list, next_translations_list, next_attentions_list 
    t_infos_list = compute_next_lists(
        new_state_ensemble, new_scores, 
        beam_search_params,
        #beam_width, beam_pruning_margin,
        #beam_score_length_normalization, beam_score_length_normalization_strength,
        #beam_score_coverage_penalty, beam_score_coverage_penalty_strength,
        eos_idx,
        current_translations, finished_translations,
        current_attentions, attn_ensemble, force_finish=force_finish, need_attention=need_attention,
        constraints_fn=constraints_fn,
        required_tgt_idx_list=required_tgt_idx_list,
        xp=xp)

    if len(t_infos_list.next_states_list) == 0:
        return False  # We only found finished translations

    # Create the new translation states

    #next_words_array = np.array(t_infos_list.next_words_list, dtype=np.int32)
    #if xp is not np:
    #    next_words_array = cuda.to_gpu(next_words_array)

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
    item_list = make_item_list(t_infos_list) #.next_states_list, t_infos_list.next_words_list, 
               # t_infos_list.next_score_list, t_infos_list.next_translations_list, t_infos_list.next_attentions_list)
    #print("adding", len(item_list), "items")
    for item in item_list:
        if astar_params.astar_priority_eval_string is not None:
            priority = eval(astar_params.astar_priority_eval_string)
        else:
            length_normalization = astar_params.length_normalization_constant + len(item.current_translation)
            if astar_params.length_normalization_exponent != 1:
                length_normalization = xp.power(length_normalization, astar_params.length_normalization_exponent)
            priority = item.score/length_normalization
        translations_priority_queue.put(item, priority)

    return True


def ensemble_astar_search(model_ensemble, src_batch, src_mask, nb_steps, eos_idx,
                         beam_search_params:BeamSearchParams,
                         #beam_width=20, beam_pruning_margin=None,
                         #beam_score_length_normalization=None,
                         #beam_score_length_normalization_strength=0.2,
                         #beam_score_coverage_penalty=None,
                         #beam_score_coverage_penalty_strength=0.2,
                         need_attention=False,
                         #force_finish=False,
                         prob_space_combination=False, 
                         #use_unfinished_translation_if_none_found=False,
                         constraints: Optional[BeamSearchConstraints] = None,
                         #constraints_fn=None,
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
    
    required_tgt_idx = constraints.required_tgt_idx if constraints is not None else None
    constraints_fn = constraints.constraint_fn if constraints is not None else None

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

        required_tgt_idx = None if constraints is None else constraints.required_tgt_idx

        astar_queue = TranslationPriorityQueue()    
        astar_queue.put(Item(required_tgt_idx=required_tgt_idx), 0)



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
                num_step, nb_steps,
                dec_cell_ensemble,
                eos_idx,
                astar_queue,
                beam_search_params,
                #beam_width,
                #beam_pruning_margin,
                #beam_score_length_normalization,
                #beam_score_length_normalization_strength,
                #beam_score_coverage_penalty,
                #beam_score_coverage_penalty_strength,
                finished_translations,
                force_finish=beam_search_params.force_finish and num_step == (nb_steps - 1),
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
            log.info(f"no finished translation found  {nb_steps} {len(astar_queue.queue)} {astar_queue.stats()}")
            if beam_search_params.use_unfinished_translation_if_none_found:
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
