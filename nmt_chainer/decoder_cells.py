#!/usr/bin/env python
"""decoder_cells.py: Implementation of RNNSearch in Chainer"""
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
import math, random

import rnn_cells
from utils import ortho_init, minibatch_sampling

from constant_batch_mul import batch_matmul_constant, matmul_constant

from models import AttentionModule

import logging
logging.basicConfig()
log = logging.getLogger("rnns:dec")
log.setLevel(logging.INFO)

class ConditionalizedDecoderCell(object):
    def __init__(self, decoder_chain, compute_ctxt, mb_size, noise_on_prev_word = False,
                    mode = "test", lexicon_probability_matrix = None, lex_epsilon = 1e-3, demux = False):
        self.decoder_chain = decoder_chain
        self.compute_ctxt = compute_ctxt
        self.noise_on_prev_word = noise_on_prev_word
        self.mode = mode
        self.lexicon_probability_matrix = lexicon_probability_matrix
        self.lex_epsilon = lex_epsilon
        
        self.mb_size = mb_size
        self.demux = demux
        
        self.xp = decoder_chain.xp
        
        if noise_on_prev_word:
            self.noise_mean = self.xp.ones((mb_size, self.decoder_chain.Eo), dtype = self.xp.float32)
            self.noise_lnvar = self.xp.zeros((mb_size, self.decoder_chain.Eo), dtype = self.xp.float32)
    
    def advance_state(self, previous_states, prev_y):
        current_mb_size = prev_y.data.shape[0]
        assert self.mb_size is None or current_mb_size <= self.mb_size
        
        if current_mb_size < len(previous_states[0].data):
            truncated_states = [None] * len(previous_states)
            for num_state in xrange(len(previous_states)):
                truncated_states[num_state], _ = F.split_axis(previous_states[num_state], (current_mb_size,), 0)
            previous_states = tuple(truncated_states)
                    
        output_state = previous_states[-1]
        ci, attn = self.compute_ctxt(output_state)
        concatenated = F.concat( (prev_y, ci) )
            
        new_states = self.decoder_chain.gru(previous_states, concatenated, mode = self.mode)
        return  new_states, concatenated, attn
    
    def compute_logits(self, new_states, concatenated, attn):
        new_output_state = new_states[-1]
            
        all_concatenated = F.concat((concatenated, new_output_state))
        logits = self.decoder_chain.lin_o(self.decoder_chain.maxo(all_concatenated))
        
        if self.lexicon_probability_matrix is not None:
            current_mb_size = new_output_state.data.shape[0]
            assert self.mb_size is None or current_mb_size <= self.mb_size
            lexicon_probability_matrix = self.lexicon_probability_matrix[:current_mb_size]
            
            # Just making sure data shape is as expected
            attn_mb_size, max_source_length_attn = attn.data.shape 
            assert attn_mb_size == current_mb_size
            lex_mb_size, max_source_length_lexicon, v_size_lexicon = lexicon_probability_matrix.shape
            assert max_source_length_lexicon == max_source_length_attn
            assert logits.data.shape == (current_mb_size, v_size_lexicon)
            
            if self.demux:
                assert lex_mb_size == 1
                weighted_lex_probs = F.reshape(
                        matmul_constant(attn, lexicon_probability_matrix.reshape(lexicon_probability_matrix.shape[1],
                                                            lexicon_probability_matrix.shape[2])), 
                                               logits.data.shape)
            else:
                assert lex_mb_size == current_mb_size
                
    #                 weighted_lex_probs = F.reshape(
    #                         F.batch_matmul(attn, ConstantFunction(lexicon_probability_matrix)(), transa = True), 
    #                                                logits.data.shape)
                
                weighted_lex_probs = F.reshape(
                        batch_matmul_constant(attn, lexicon_probability_matrix, transa = True), 
                                               logits.data.shape)
            
            logits += F.log(weighted_lex_probs + self.lex_epsilon)
        return logits
    
    def advance_one_step(self, previous_states, prev_y):
        
        if self.noise_on_prev_word:
            current_mb_size = prev_y.data.shape[0]
            assert self.mb_size is None or current_mb_size <= self.mb_size
            prev_y = prev_y * F.gaussian(Variable(self.noise_mean[:current_mb_size], volatile = "auto"), 
                                         Variable(self.noise_lnvar[:current_mb_size], volatile = "auto"))
        
        new_states, concatenated, attn = self.advance_state(previous_states, prev_y)
        
        logits = self.compute_logits(new_states, concatenated, attn)
        
        return new_states, logits, attn
    
    def get_initial_logits(self, mb_size = None):
        if mb_size is None:
            mb_size = self.mb_size
        assert mb_size is not None
        
        previous_states = self.decoder_chain.gru.get_initial_states(mb_size)

        prev_y = F.broadcast_to(self.decoder_chain.bos_embeding, (mb_size, self.decoder_chain.Eo))
        
        new_states, logits, attn = self.advance_one_step(previous_states, prev_y)
        
        return new_states, logits, attn
    
    def __call__(self, prev_states, inpt):
        prev_y = self.decoder_chain.emb(inpt)
        
        new_states, logits, attn = self.advance_one_step(prev_states, prev_y)

        return new_states, logits, attn


def compute_loss_from_decoder_cell(cell, targets, use_previous_prediction = 0, 
                                   raw_loss_info = False, per_sentence = False, keep_attn = False):
    loss = None
    attn_list = []
    
    mb_size = targets[0].data.shape[0]
    assert cell.mb_size is None or cell.mb_size == mb_size
    states, logits, attn = cell.get_initial_logits(mb_size)
    
    total_nb_predictions = 0
    
    
    for i in xrange(len(targets)):
        if keep_attn:
            attn_list.append(attn)
        
        if per_sentence:
            normalized_logits = F.log_softmax(logits)
            total_local_loss = F.select_item(normalized_logits, targets[i])
            if loss is not None and total_local_loss.data.shape[0] != loss.data.shape[0]:
                assert total_local_loss.data.shape[0] < loss.data.shape[0]
                total_local_loss = F.concat(
                            (total_local_loss, 
                             Variable(cell.xp.zeros(loss.data.shape[0] - total_local_loss.data.shape[0], 
                                                    dtype = cell.xp.float32), volatile = "auto")),
                            axis = 0)
        else:
            local_loss = F.softmax_cross_entropy(logits, targets[i]) 
            nb_predictions = targets[i].data.shape[0]
            total_local_loss = local_loss * nb_predictions
            total_nb_predictions += nb_predictions
        
        loss = total_local_loss if loss is None else loss + total_local_loss
        
        if i >= len(targets) -1: # skipping generation of last states as unneccessary
            break
        
        current_mb_size = targets[i].data.shape[0]
        required_next_mb_size = targets[i+1].data.shape[0]
        
        if use_previous_prediction > 0 and random.random() < use_previous_prediction:
            previous_word = Variable(cell.xp.argmax(logits.data[:required_next_mb_size], axis = 1).astype(cell.xp.int32), volatile = "auto")
        else:
            if required_next_mb_size < current_mb_size:
                previous_word, _ = F.split_axis(targets[i], (required_next_mb_size,), 0 )
                current_mb_size = required_next_mb_size
            else:
                previous_word = targets[i]
                               
        states, logits, attn = cell(states, previous_word)
        
        
    if raw_loss_info:
        return (loss, total_nb_predictions), attn_list
    else:
        loss = loss / total_nb_predictions
        return loss, attn_list

def sample_from_decoder_cell(cell, nb_steps, best = False, keep_attn_values = False,
           need_score = False):
    
    states, logits, attn = cell.get_initial_logits()
    
    score = 0
    sequences = []
    attn_list = []
    
    for _ in xrange(nb_steps):
        if keep_attn_values:
            attn_list.append(attn)
            
        probs = F.softmax(logits)
        if best:
            curr_idx = cell.xp.argmax(probs.data, 1).astype(np.int32)
        else:
#                 curr_idx = self.xp.empty((mb_size,), dtype = np.int32)
            if cell.xp != np:
                probs_data = cuda.to_cpu(probs.data)
            else:
                probs_data = probs.data
            curr_idx = minibatch_sampling(probs_data)
            if cell.xp != np:
                curr_idx = cuda.to_gpu(curr_idx.astype(np.int32))
            else:
                curr_idx = curr_idx.astype(np.int32)
#                 for i in xrange(mb_size):
#                     sampler = chainer.utils.WalkerAlias(probs_data[i])
#                     curr_idx[i] =  sampler.sample(1)[0]
        if need_score:
            score = score + np.log(cuda.to_cpu(probs.data)[np.arange(cell.mb_size),cuda.to_cpu(curr_idx)])
        sequences.append(curr_idx)
        
        previous_word = Variable(curr_idx, volatile = "auto")
        
        states, logits, attn = cell(states, previous_word)
        
        
    return sequences, score, attn_list

class Decoder(Chain):
    """ Decoder for RNNSearch. 
        The __call_ takes 3 required parameters: fb_concat, targets, mask.
        
        fb_concat should be the result of a call to Encoder.
        
        targets is a python list of chainer variables of type int32 and of variable shape (n,)
            the values n should be decreasing:
                i < j => targets[i].data.shape[0] >= targets[j].data.shape[0]
            targets[i].data[j] is the jth elements of the ith sequence in the minibatch
            all this imply that the sequences of the minibatch should be sorted from longest to shortest
            
        mask is as in the description of Encoder.
        
        * it is up to the user to add an EOS token to the data.
               
        Return a loss and the attention model values
    """
    def __init__(self, Vo, Eo, Ho, Ha, Hi, Hl, attn_cls = AttentionModule, init_orth = False,
                 cell_type = rnn_cells.LSTMCell):
#         assert cell_type in "gru dgru lstm slow_gru".split()
#         self.cell_type = cell_type
#         if cell_type == "gru":
#             gru = faster_gru.GRU(Ho, Eo + Hi)
#         elif cell_type == "dgru":
#             gru = DoubleGRU(Ho, Eo + Hi)
#         elif cell_type == "lstm":
#             gru = L.StatelessLSTM(Eo + Hi, Ho) #, forget_bias_init = 3)
#         elif cell_type == "slow_gru":
#             gru = L.GRU(Ho, Eo + Hi)
        
        
        if isinstance(cell_type, (str,unicode)):
            cell_type = rnn_cells.create_cell_model_from_string(cell_type)
        
        gru = cell_type(Eo + Hi, Ho)
        
        log.info("constructing decoder [%r]"%(cell_type,))
        
        super(Decoder, self).__init__(
            emb = L.EmbedID(Vo, Eo),
#             gru = L.GRU(Ho, Eo + Hi),
            
            gru = gru,
            
            maxo = L.Maxout(Eo + Hi + Ho, Hl, 2),
            lin_o = L.Linear(Hl, Vo, nobias = False),
            
            attn_module = attn_cls(Hi, Ha, Ho, init_orth = init_orth)
        )
#         self.add_param("initial_state", (1, Ho))
        self.add_param("bos_embeding", (1, Eo))
        
        self.Hi = Hi
        self.Ho = Ho
        self.Eo = Eo
#         self.initial_state.data[...] = np.random.randn(Ho)
        self.bos_embeding.data[...] = np.random.randn(Eo)
        
        if init_orth:
            ortho_init(self.gru)
            ortho_init(self.lin_o)
            ortho_init(self.maxo)
        
    def give_conditionalized_cell(self, fb_concat, src_mask, noise_on_prev_word = False,
                    mode = "test", lexicon_probability_matrix = None, lex_epsilon = 1e-3, demux = False):
        assert mode in "test train".split()
        mb_size, nb_elems, Hi = fb_concat.data.shape
        assert Hi == self.Hi, "%i != %i"%(Hi, self.Hi)
    
        compute_ctxt = self.attn_module(fb_concat, src_mask)
        
        if not demux:
            return ConditionalizedDecoderCell(self, compute_ctxt, mb_size, noise_on_prev_word = noise_on_prev_word,
                    mode = mode, lexicon_probability_matrix = lexicon_probability_matrix, lex_epsilon = lex_epsilon)
        else:
            assert mb_size == 1
            assert demux >= 1
            compute_ctxt = self.attn_module.compute_ctxt_demux(fb_concat, src_mask)
            return ConditionalizedDecoderCell(self, compute_ctxt, None, noise_on_prev_word = noise_on_prev_word,
                    mode = mode, lexicon_probability_matrix = lexicon_probability_matrix, lex_epsilon = lex_epsilon,
                    demux = True)    
    
    def compute_loss(self, fb_concat, src_mask, targets, raw_loss_info = False, keep_attn_values = False, 
                     noise_on_prev_word = False, use_previous_prediction = 0, mode = "test", per_sentence = False,
                     lexicon_probability_matrix = None, lex_epsilon = 1e-3
                     ):
        decoding_cell = self.give_conditionalized_cell(fb_concat, src_mask, noise_on_prev_word = noise_on_prev_word,
                    mode = mode, lexicon_probability_matrix = lexicon_probability_matrix, lex_epsilon = lex_epsilon)
        loss, attn_list = compute_loss_from_decoder_cell(decoding_cell, targets, 
                                                         use_previous_prediction = use_previous_prediction,
                                                         raw_loss_info = raw_loss_info,
                                                         per_sentence = per_sentence,
                                                         keep_attn = keep_attn_values)
        return loss, attn_list
    
    def sample(self, fb_concat, src_mask, nb_steps, mb_size, lexicon_probability_matrix = None, 
               lex_epsilon = 1e-3, best = False, keep_attn_values = False, need_score = False):
        decoding_cell = self.give_conditionalized_cell(fb_concat, src_mask, noise_on_prev_word = False,
                    mode = "test", lexicon_probability_matrix = lexicon_probability_matrix,
                     lex_epsilon = lex_epsilon)
        sequences, score, attn_list = sample_from_decoder_cell(decoding_cell, nb_steps, best = best, 
                                                keep_attn_values = keep_attn_values,
                                                need_score = need_score)
         
        return sequences, score, attn_list
        
