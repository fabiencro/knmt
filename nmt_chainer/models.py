#!/usr/bin/env python
"""models.py: Implementation of RNNSearch in Chainer"""
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

import rnn_cells
from utils import ortho_init, compute_lexicon_matrix, minibatch_sampling

import logging
logging.basicConfig()
log = logging.getLogger("rnns:models")
log.setLevel(logging.INFO)


class CopyMechanism(Chain):
    def __init__(self, Hi, Ho):
        super(CopyMechanism, self).__init__(
            lin = L.Linear(Hi, Ho)
            )
        self.Hi = Hi
        self.Ho = Ho
        
    def __call__(self, inpt, mask):
        mb_size = inpt.data.shape[0]
        max_length = inpt.data.shape[1]
        
        precomp = F.reshape(F.tanh(self.lin(F.reshape(inpt, (-1, self.Hi)))),(mb_size, -1, self.Ho))
        
        mask_offset = max_length - len(mask)

        precomp_mask_penalties = self.xp.concatenate(
                                    [
                self.xp.zeros((mb_size, mask_offset), dtype = self.xp.float32),
                        -10000 * (1-self.xp.concatenate([
                self.xp.reshape(mask_elem, (mb_size, 1)).astype(self.xp.float32) for mask_elem in mask], 1))
                ], 1
                )

        
        def compute_copy_coefficients(state):
            betas = F.reshape(F.batch_matmul(precomp, state),(mb_size, -1) )
            masked_betas = betas + precomp_mask_penalties
            return masked_betas
        
        return compute_copy_coefficients
        
class BNList(ChainList):
    def __init__(self, size, max_length):
        super(BNList, self).__init__()
        for _ in xrange(max_length):
            bn = L.BatchNormalization(size)
            bn.gamma.data.fill(0.1)
            self.add_link(bn)
        self.max_length = max_length
        
    def __call__(self, x, i, test = False):
        if i < self.max_length:
            return self[i](x, test = test)
        else:
            return self[self.max_length - 1](x, test = test)

class Encoder(Chain):
    """ Chain that encode a sequence. 
        The __call_ takes 2 parameters: sequence and mask.
        mask and length should be 2 python lists of same length #length.
        
        sequence should be a python list of Chainer Variables wrapping a numpy/cupy array of shape (mb_size,) and type int32 each.
            -- where mb_size is the minibatch size
        sequence[i].data[j] should be the jth element of source sequence number i, or a padding value if the sequence number i is
            shorter than j.
        
        mask should be a python list of Chainer Variables wrapping a numpy/cupy array of shape (mb_size,) and type bool each.
        mask[i].data[j] should be True if and only if sequence[i].data[j] is not a padding value.
        
        Return a chainer variable of shape (mb_size, #length, 2*Hi) and type float32
    """
    def __init__(self, Vi, Ei, Hi, init_orth = False, use_bn_length = 0, cell_type = rnn_cells.LSTMCell):
        gru_f = cell_type(Ei, Hi)
        gru_b = cell_type(Ei, Hi)

        log.info("constructing encoder [%s]"%(cell_type,))
        super(Encoder, self).__init__(
            emb = L.EmbedID(Vi, Ei),
#             gru_f = L.GRU(Hi, Ei),
#             gru_b = L.GRU(Hi, Ei)
            
            gru_f = gru_f,
            gru_b = gru_b
        )
        self.Hi = Hi
        
        if use_bn_length > 0:
            self.add_link("bn_f", BNList(Hi, use_bn_length))
#             self.add_link("bn_b", BNList(Hi, use_bn_length)) #TODO
        self.use_bn_length = use_bn_length
        
        if init_orth:
            ortho_init(self.gru_f)
            ortho_init(self.gru_b)
        
    def __call__(self, sequence, mask, mode = "test"):
        assert mode in "test train".split()
        
        mb_size = sequence[0].data.shape[0]
        
        mb_initial_states_f = self.gru_f.get_initial_states(mb_size)
        mb_initial_states_b = self.gru_b.get_initial_states(mb_size)
        
        embedded_seq = []
        for elem in sequence:
            embedded_seq.append(self.emb(elem))
            
        prev_states = mb_initial_states_f
        forward_seq = []
        for i, x in enumerate(embedded_seq):
            prev_states = self.gru_f(prev_states, x, mode = mode)
            output = prev_states[-1]
            forward_seq.append(output)
            
        mask_length = len(mask)
        seq_length = len(sequence)
        assert mask_length <= seq_length
        mask_offset = seq_length - mask_length
        
        prev_states = mb_initial_states_b
            
        backward_seq = []
        for pos, x in reversed(list(enumerate(embedded_seq))):
            if pos < mask_offset:
                prev_states = self.gru_b(prev_states, x, mode = mode)
                output = prev_states[-1]
            else:
                reshaped_mask = F.broadcast_to(
                                Variable(self.xp.reshape(mask[pos - mask_offset], 
                                    (mb_size, 1)), volatile = "auto"), (mb_size, self.Hi))
                
                prev_states = self.gru_b(prev_states, x, mode = mode)
                output = prev_states[-1]
                
                masked_prev_states = [None] * len(prev_states)
                for num_state in xrange(len(prev_states)):
                    masked_prev_states[num_state] = F.where(reshaped_mask,
                                    prev_states[num_state], mb_initial_states_b[num_state]) #TODO: optimize?
                prev_states = tuple(masked_prev_states)
                output = prev_states[-1]

            
            backward_seq.append(output)
        
        assert len(backward_seq) == len(forward_seq)
        res = []
        for xf, xb in zip(forward_seq, reversed(backward_seq)):
            res.append(F.reshape(F.concat((xf, xb), 1), (-1, 1, 2 * self.Hi)))
        
        return F.concat(res, 1)
    
class AttentionModule(Chain):
    """ Attention Module for computing the current context during decoding. 
        The __call_ takes 2 parameters: fb_concat and mask.
        
        fb_concat should be the result of a call to Encoder.
        mask is as in the description of Encoder
               
        Return a chainer variable of shape (mb_size, Hi) and type float32
    """
    def __init__(self, Hi, Ha, Ho, init_orth = False):
        super(AttentionModule, self).__init__(
            al_lin_h = L.Linear(Hi, Ha, nobias = False),
            al_lin_s = L.Linear(Ho, Ha, nobias = True),
            al_lin_o = L.Linear(Ha, 1, nobias = True)                                     
        )
        self.Hi = Hi
        self.Ha = Ha
        
        if init_orth:
            ortho_init(self.al_lin_h)
            ortho_init(self.al_lin_s)
            ortho_init(self.al_lin_o)
        
    def __call__(self, fb_concat, mask):
        mb_size, nb_elems, Hi = fb_concat.data.shape
        assert Hi == self.Hi
        precomputed_al_factor = F.reshape(self.al_lin_h(
                        F.reshape(fb_concat, (mb_size * nb_elems, self.Hi)))
                                          , (mb_size, nb_elems, self.Ha))
        
        mask_length = len(mask)
        seq_length = nb_elems
        assert mask_length <= seq_length
        mask_offset = seq_length - mask_length
        
#         concatenated_mask = F.concat([F.reshape(mask_elem, (mb_size, 1)) for mask_elem in mask], 1)
        
        if mask_length > 0:
            with cuda.get_device(mask[0]):
                if mask_offset > 0:
                    concatenated_penalties = self.xp.concatenate(
                                    [
                                    self.xp.zeros((mb_size, mask_offset), dtype = self.xp.float32),
                                    -10000 * (1-self.xp.concatenate([
                            self.xp.reshape(mask_elem, (mb_size, 1)).astype(self.xp.float32) for mask_elem in mask], 1))
                                    ], 1
                                    )
                else:
                    concatenated_penalties =  -10000 * (1-self.xp.concatenate([
                            self.xp.reshape(mask_elem, (mb_size, 1)).astype(self.xp.float32) for mask_elem in mask], 1))
                                    
        
        
        def compute_ctxt(previous_state):
            current_mb_size = previous_state.data.shape[0]
            if current_mb_size < mb_size:
                al_factor, _ = F.split_axis(precomputed_al_factor, (current_mb_size,), 0)
                used_fb_concat, _ = F.split_axis(fb_concat, (current_mb_size,), 0)
                if mask_length > 0:
                    used_concatenated_penalties = concatenated_penalties[:current_mb_size]
            else:
                al_factor = precomputed_al_factor
                used_fb_concat = fb_concat
                if mask_length > 0:
                    used_concatenated_penalties = concatenated_penalties
                
            state_al_factor = self.al_lin_s(previous_state)
            state_al_factor_bc = F.broadcast_to(F.reshape(state_al_factor, (current_mb_size, 1, self.Ha)), (current_mb_size, nb_elems, self.Ha) )
            a_coeffs = F.reshape(self.al_lin_o(F.reshape(F.tanh(state_al_factor_bc + al_factor), 
                            (current_mb_size* nb_elems, self.Ha))), (current_mb_size, nb_elems))
            
            if mask_length > 0:
                with cuda.get_device(used_concatenated_penalties):
                    a_coeffs = a_coeffs + used_concatenated_penalties# - 10000 * (1-used_concatenated_mask.data) 
            
            attn = F.softmax(a_coeffs)
            
            ci = F.reshape(F.batch_matmul(attn, used_fb_concat, transa = True), (current_mb_size, self.Hi))
            
            return ci, attn
        
        return compute_ctxt
    
    def compute_ctxt_demux(self, fb_concat, mask):
        mb_size, nb_elems, Hi = fb_concat.data.shape
        assert Hi == self.Hi
        assert mb_size == 1
        assert len(mask) == 0
        
        precomputed_al_factor = F.reshape(self.al_lin_h(
                        F.reshape(fb_concat, (mb_size * nb_elems, self.Hi)))
                                          , (mb_size, nb_elems, self.Ha))
        
#         concatenated_mask = F.concat([F.reshape(mask_elem, (mb_size, 1)) for mask_elem in mask], 1)  
        
        def compute_ctxt(previous_state):
            current_mb_size = previous_state.data.shape[0]
                
            al_factor = F.broadcast_to(precomputed_al_factor, (current_mb_size, nb_elems, self.Ha))
#             used_fb_concat = F.broadcast_to(fb_concat, (current_mb_size, nb_elems, Hi))
#             used_concatenated_mask = F.broadcast_to(concatenated_mask, (current_mb_size, nb_elems))
                
            state_al_factor = self.al_lin_s(previous_state)
            state_al_factor_bc = F.broadcast_to(F.reshape(state_al_factor, (current_mb_size, 1, self.Ha)), (current_mb_size, nb_elems, self.Ha) )
            a_coeffs = F.reshape(self.al_lin_o(F.reshape(F.tanh(state_al_factor_bc + al_factor), 
                            (current_mb_size* nb_elems, self.Ha))), (current_mb_size, nb_elems))
            
            
#             with cuda.get_device(used_concatenated_mask.data):
#                 a_coeffs = a_coeffs - 10000 * (1-used_concatenated_mask.data) 
            
            attn = F.softmax(a_coeffs)
            
#             ci = F.reshape(F.batch_matmul(attn, used_fb_concat, transa = True), (current_mb_size, self.Hi))
            
            ci = F.reshape(F.matmul(attn, F.reshape(fb_concat, (nb_elems, Hi))), (current_mb_size, self.Hi))
            
            return ci, attn
        
        return compute_ctxt
    
class DeepAttentionModule(Chain):
    """ DeepAttention Module for computing the current context during decoding. 
        The __call_ takes 2 parameters: fb_concat and mask.
        
        fb_concat should be the result of a call to Encoder.
        mask is as in the description of Encoder
               
        Return a chainer variable of shape (mb_size, Hi) and type float32
    """
    def __init__(self, Hi, Ha, Ho, init_orth = False):
        log.info("using deep attention")
        super(DeepAttentionModule, self).__init__(
            attn1 = AttentionModule(Hi, Ha, Ho, init_orth = init_orth),
            attn2 = AttentionModule(Hi, Ha, Ho + Hi, init_orth = init_orth)                                
        )
        
    def __call__(self, fb_concat, mask):
        compute_ctxt1 = self.attn1(fb_concat, mask)
        compute_ctxt2 = self.attn2(fb_concat, mask)
        
        def compute_ctxt(previous_state):
            ci1, attn1 = compute_ctxt1(previous_state)
            intermediate_state = F.concat((previous_state, ci1), axis = 1)
            ci2, attn2 = compute_ctxt2(intermediate_state)
            
            return ci2, attn2
        
        return compute_ctxt
    
    def compute_ctxt_demux(self, fb_concat, mask):
        raise NotImplemented
    

class GradKeeper(chainer.Function):
    def __init__(self, grad):
        super(GradKeeper, self).__init__()
        self.grad = grad
        
    def forward(self, inputs):
        return (np.array([0], dtype = np.float32),)
    
    def backward(self, inputs, grad_output):
        return (self.grad,)

class ConstantFunction(chainer.Function):
    def __init__(self, val):
        super(ConstantFunction, self).__init__()
        self.val = val
        
    def forward(self, inputs):
        return (self.val,)
    
    def backward(self, inputs, grad_output):
        return ()


#     
#     def get_sampler_reinf(self, fb_concat, mask, eos_idx, nb_steps = 50, use_best_for_sample = False,
#                     temperature = None,
#                     lexicon_probability_matrix = None,  
#                     lex_epsilon = 1e-3,
#                     mode = "test"):
#         mb_size, nb_elems, Hi = fb_concat.data.shape
#         assert Hi == self.Hi, "%i != %i"%(Hi, self.Hi)
#     
#         compute_ctxt = self.attn_module(fb_concat, mask)
#         while 1:
#             yield self.sample_reinf(self, nb_steps, compute_ctxt, mb_size, eos_idx, best = use_best_for_sample, 
#                      lexicon_probability_matrix = lexicon_probability_matrix, lex_epsilon = lex_epsilon, 
#                      temperature = temperature,
#                      mode = mode)
#             
# 
#     def compute_loss(self, targets, compute_ctxt, raw_loss_info = False, keep_attn_values = False, 
#                      noise_on_prev_word = False, use_previous_prediction = 0, mode = "test", per_sentence = False,
#                      lexicon_probability_matrix = None, lex_epsilon = 1e-3
#                      ):
#         assert mode in "test train".split()
#         
#         loss = None
#         current_mb_size = targets[0].data.shape[0]
# 
#         previous_states = self.gru.get_initial_states(current_mb_size)
# 
#         previous_word = None
#         with cuda.get_device(previous_states[0].data):
#             prev_y = F.broadcast_to(self.bos_embeding, (current_mb_size, self.Eo))
#         attn_list = []
#         total_nb_predictions = 0
#         
#         if noise_on_prev_word:
#             noise_mean = Variable(self.xp.ones_like(prev_y.data, dtype = self.xp.float32))
#             noise_lnvar = Variable(self.xp.zeros_like(prev_y.data, dtype = self.xp.float32))
#         
#         for i in xrange(len(targets)):
#             assert i == 0 or previous_states[0].data.shape[0] == previous_word.data.shape[0]
#             current_mb_size = targets[i].data.shape[0]
#             if current_mb_size < len(previous_states[0].data):
#                 truncated_states = [None] * len(previous_states)
#                 for num_state in xrange(len(previous_states)):
#                     truncated_states[num_state], _ = F.split_axis(previous_states[num_state], (current_mb_size,), 0)
#                 previous_states = tuple(truncated_states)
#                     
#                 if previous_word is not None:
#                     previous_word, _ = F.split_axis(previous_word, (current_mb_size,), 0 )
#                     
#                 if noise_on_prev_word:
#                     noise_mean, _ = F.split_axis(noise_mean, (current_mb_size,), 0)
#                     noise_lnvar, _ = F.split_axis(noise_lnvar, (current_mb_size,), 0)
#                     
#                 if lexicon_probability_matrix is not None:
#                     lexicon_probability_matrix = lexicon_probability_matrix[:current_mb_size]
#                     
#             if previous_word is not None: #else we are using the initial prev_y
#                 prev_y = self.emb(previous_word)
#             assert previous_states[0].data.shape[0] == current_mb_size
#             
#             if noise_on_prev_word:
#                 prev_y = prev_y * F.gaussian(noise_mean, noise_lnvar)
#             
#             
#             new_states, logits, attn = self.advance_one_step(previous_states, prev_y, 
#                                                     compute_ctxt, i, mode = mode)
# 
# 
#             if lexicon_probability_matrix is not None:
#                 # Just making sure data shape is as expected
#                 attn_mb_size, max_source_length_attn = attn.data.shape 

#                 assert attn_mb_size == current_mb_size
#                 lex_mb_size, max_source_length_lexicon, v_size_lexicon = lexicon_probability_matrix.shape
#                 assert max_source_length_lexicon == max_source_length_attn
#                 assert logits.data.shape == (current_mb_size, v_size_lexicon)
#                 assert lex_mb_size == current_mb_size
#                 
# #                 weighted_lex_probs = F.reshape(
# #                         F.batch_matmul(attn, ConstantFunction(lexicon_probability_matrix)(), transa = True), 
# #                                                logits.data.shape)
#                 
#                 weighted_lex_probs = F.reshape(
#                         batch_matmul_constant(attn, lexicon_probability_matrix, transa = True), 
#                                                logits.data.shape)
#                 
#                 logits += F.log(weighted_lex_probs + lex_epsilon)
#                 
#                 
# 
#             if keep_attn_values:
#                 attn_list.append(attn)
#                 
#             if per_sentence:
#                 normalized_logits = F.log_softmax(logits)
#                 total_local_loss = F.select_item(normalized_logits, targets[i])
#                 if loss is not None and total_local_loss.data.shape[0] != loss.data.shape[0]:
#                     assert total_local_loss.data.shape[0] < loss.data.shape[0]
#                     total_local_loss = F.concat(
#                                 (total_local_loss, self.xp.zeros(loss.data.shape[0] - total_local_loss.data.shape[0], dtype = self.xp.float32)),
#                                 axis = 0)
#             else:
#                 local_loss = F.softmax_cross_entropy(logits, targets[i])   
#             
#                 total_nb_predictions += current_mb_size
#                 total_local_loss = local_loss * current_mb_size
#             
# #             loss = local_loss if loss is None else loss + local_loss#         assert loss_format in "average raw per_sentence".split(" ")
#             loss = total_local_loss if loss is None else loss + total_local_loss
#             if use_previous_prediction > 0 and random.random() < use_previous_prediction:
#                 previous_word = Variable(self.xp.argmax(logits.data, axis = 1).astype(self.xp.int32), volatile = "auto")
#             else:
#                 previous_word = targets[i]
# #             prev_y = self.emb(previous_word)
#             previous_states = new_states
# #             attn_list.append(attn)
#         if raw_loss_info:
#             return (loss, total_nb_predictions), attn_list
#         else:
#             loss = loss / total_nb_predictions
#             return loss, attn_list
#     
#     def compute_loss_and_backward(self, fb_concat, targets, mask, 
#                      noise_on_prev_word = False, use_previous_prediction = 0):
#         
#         compute_ctxt = self.attn_module(fb_concat, mask)
#         
#         loss = 0
#         pseudo_loss = 0
#         
#         current_mb_size = targets[0].data.shape[0]
# 
#         previous_states = self.gru.get_initial_states(current_mb_size)
# 
#         previous_word = None
#         with cuda.get_device(previous_states[0].data):
#             prev_y = F.broadcast_to(self.bos_embeding, (current_mb_size, self.Eo))
#         
#         if noise_on_prev_word:
#             noise_mean = Variable(self.xp.ones_like(prev_y.data, dtype = self.xp.float32))
#             noise_lnvar = Variable(self.xp.zeros_like(prev_y.data, dtype = self.xp.float32))
#         
#         total_nb_predictions = sum(t.data.shape[0] for t in targets)
#         
#         for i in xrange(len(targets)):
#             assert i == 0 or previous_states[0].data.shape[0] == previous_word.data.shape[0]
#             current_mb_size = targets[i].data.shape[0]
#             if current_mb_size < len(previous_states[0].data):
#                 truncated_states = [None] * len(previous_states)
#                 for num_state in xrange(len(previous_states)):
#                     truncated_states[num_state], _ = F.split_axis(previous_states[num_state], (current_mb_size,), 0)
#                 previous_states = tuple(truncated_states)
#                     
#                 if previous_word is not None:
#                     previous_word, _ = F.split_axis(previous_word, (current_mb_size,), 0 )
#                     
#                 if noise_on_prev_word:
#                     noise_mean, _ = F.split_axis(noise_mean, (current_mb_size,), 0)
#                     noise_lnvar, _ = F.split_axis(noise_lnvar, (current_mb_size,), 0)
#             if previous_word is not None: #else we are using the initial prev_y
#                 prev_y = self.emb(previous_word)
#             assert previous_states[0].data.shape[0] == current_mb_size
#             
#             if noise_on_prev_word:
#                 prev_y = prev_y * F.gaussian(noise_mean, noise_lnvar)
#             
# 
#             output_state = previous_states[-1]
#             ci, attn = compute_ctxt(output_state)
#             concatenated = F.concat( (prev_y, ci) )
#                 
#             new_states = self.gru(previous_states, concatenated, mode = "train")
#             new_output_state = new_states[-1]
# 
#             all_concatenated = F.concat((concatenated, new_output_state))
#             
#             all_concatenated_bis = Variable(all_concatenated.data)
#             logits = self.lin_o(self.maxo(all_concatenated_bis))
#             
#             partial_loss = F.softmax_cross_entropy(logits, targets[i]) * current_mb_size / total_nb_predictions
#             loss += partial_loss.data
#             partial_loss.backward()
#             gr = GradKeeper(all_concatenated_bis.grad)
#             del logits
#             del all_concatenated_bis
#             del partial_loss
#             
#             pseudo_loss += gr(all_concatenated)
#             
#             if use_previous_prediction > 0 and random.random() < use_previous_prediction:
#                 previous_word = Variable(self.xp.argmax(logits.data, axis = 1).astype(self.xp.int32), volatile = "auto")
#             else:
#                 previous_word = targets[i]
# 
#             previous_states = new_states
# 
#         pseudo_loss.backward()
#         return loss, total_nb_predictions
# 
#     
# 
# #     
#     def get_predictor(self, fb_concat, mask):
#         mb_size, nb_elems, Hi = fb_concat.data.shape
#         assert Hi == self.Hi, "%i != %i"%(Hi, self.Hi)
#         xp = cuda.get_array_module(self.initial_state.data)
#         
#         compute_ctxt = self.attn_module.compute_ctxt_demux(fb_concat, mask)
#         
#         assert mb_size == 1
#         current_mb_size = mb_size
# #         previous_state = F.concat( [self.initial_state] * current_mb_size, 0)
#         previous_state = [F.broadcast_to(self.initial_state, (current_mb_size, self.Ho))]
# #         previous_word = Variable(np.array([self.bos_idx] * mb_size, dtype = np.int32))
#         previous_word = [None]
#         with cuda.get_device(self.initial_state.data):
# #             previous_word = Variable(xp.array([self.bos_idx] * current_mb_size, dtype = np.int32))
#             prev_y_initial = F.broadcast_to(self.bos_embeding, (current_mb_size, self.Eo))
#             
#             
#         def choose(voc_list, i):
#             if previous_word[0] is not None: #else we are using the initial prev_y
#                 prev_y = self.emb(previous_word[0])
#             else:
#                 prev_y = prev_y_initial
#             assert previous_state[0].data.shape[0] == current_mb_size
#             
#             new_state, logits, attn = self.advance_one_step(previous_state[0], prev_y, 
#                                                       compute_ctxt, i, mode = "test")
#             
#             best_w = None
#             best_score = None
#             for w in voc_list:
#                 score = logits.data[0][w]
#                 if best_score is None or score > best_score:
#                     best_score = score
#                     best_w = w
#                             
#                         
#             previous_word[0] = Variable(self.xp.array([best_w], dtype = self.xp.int32), volatile = "auto")
#             previous_state[0] = new_state
#             return best_w
#         return choose
# 
#     
#     def __call__(self, fb_concat, targets, mask, use_best_for_sample = False, raw_loss_info = False,
#                     keep_attn_values = False, need_score = False, noise_on_prev_word = False,
#                     use_previous_prediction = 0, mode = "test", lexicon_probability_matrix = None, lex_epsilon = 1e-3):
#         assert mode in "test train".split()
#         mb_size, nb_elems, Hi = fb_concat.data.shape
#         assert Hi == self.Hi, "%i != %i"%(Hi, self.Hi)
#     
#         compute_ctxt = self.attn_module(fb_concat, mask)
# 
#         if isinstance(targets, int):
#             return self.sample(targets, compute_ctxt, mb_size, best = use_best_for_sample,
#                                keep_attn_values = keep_attn_values, need_score = need_score,
#                                lexicon_probability_matrix = lexicon_probability_matrix, lex_epsilon = lex_epsilon)
#         else:
#             return self.compute_loss(targets, compute_ctxt, raw_loss_info = raw_loss_info,
#                                      keep_attn_values = keep_attn_values, noise_on_prev_word = noise_on_prev_word,
#                                      use_previous_prediction = use_previous_prediction, mode = mode,
#                                      lexicon_probability_matrix = lexicon_probability_matrix, lex_epsilon = lex_epsilon)     
#         
# class EncoderDecoder(Chain):
#     """ Do RNNSearch Encoding/Decoding
#         The __call__ takes 3 required parameters: src_batch, tgt_batch, src_mask
#         src_batch is as in the sequence parameter of Encoder
#         tgt_batch is as in the targets parameter of Decoder
#         src_mask is as in the mask parameter of Encoder
#         
#         return loss and attention values
#     """
#     def __init__(self, Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl, attn_cls = AttentionModule, init_orth = False, use_bn_length = 0,
#                 encoder_cell_type = "gru", decoder_cell_type = "gru",
#                 lexical_probability_dictionary = None, lex_epsilon = 1e-3
#                 ):
#         log.info("constructing encoder decoder with Vi:%i Ei:%i Hi:%i Vo:%i Eo:%i Ho:%i Ha:%i Hl:%i" % 
#                                         (Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl))
#         super(EncoderDecoder, self).__init__(
#             enc = Encoder(Vi, Ei, Hi, init_orth = init_orth, use_bn_length = use_bn_length,
#                           cell_type = encoder_cell_type),
#             dec = Decoder(Vo, Eo, Ho, Ha, 2 * Hi, Hl, attn_cls = attn_cls, init_orth = init_orth, 
#                           use_bn_length = use_bn_length, cell_type = decoder_cell_type)
#         )
#         self.Vo = Vo
#         self.lexical_probability_dictionary = lexical_probability_dictionary
#         self.lex_epsilon = lex_epsilon
#         
#     def __call__(self, src_batch, tgt_batch, src_mask, use_best_for_sample = False, display_attn = False,
#                  raw_loss_info = False, keep_attn_values = False, need_score = False, noise_on_prev_word = False,
#                  use_previous_prediction = 0, mode = "test"
#                  ):
#         assert mode in "test train".split()
#         
#         if self.lexical_probability_dictionary is not None:
#             lexicon_probability_matrix = compute_lexicon_matrix(src_batch, self.lexical_probability_dictionary, self.Vo)
#             if self.xp != np:
#                 lexicon_probability_matrix = cuda.to_gpu(lexicon_probability_matrix, cuda.get_device(self.dec.lin_o.W.data))
#         else:
#             lexicon_probability_matrix = None
#             
#         fb_src = self.enc(src_batch, src_mask, mode = mode)
#         loss = self.dec(fb_src, tgt_batch, src_mask, use_best_for_sample = use_best_for_sample, raw_loss_info = raw_loss_info,
#                         keep_attn_values = keep_attn_values, need_score = need_score, noise_on_prev_word = noise_on_prev_word,
#                         mode = mode, use_previous_prediction = use_previous_prediction,
#                         lexicon_probability_matrix = lexicon_probability_matrix, lex_epsilon = self.lex_epsilon)
#         return loss
#     
# #     def compute_loss_and_backward(self, src_batch, tgt_batch, src_mask, raw_loss_info = False):
# #         fb_src = self.enc(src_batch, src_mask)
# #         loss = self.dec.compute_loss_and_backward(fb_src, tgt_batch, src_mask, raw_loss_info = raw_loss_info)
# #         return loss
#     
#     def compute_loss_and_backward(self, src_batch, tgt_batch, src_mask, 
#                      noise_on_prev_word = False, use_previous_prediction = 0):
#         
#         fb_concat = self.enc(src_batch, src_mask, mode = "train")
#         return self.dec.compute_loss_and_backward(fb_concat, tgt_batch, src_mask, 
#                      noise_on_prev_word = False, use_previous_prediction = 0)
#     
# #     def sample(self, src_batch, src_mask, nb_steps, use_best_for_sample, keep_attn_values = False, need_score = False):
# #         fb_src = self.enc(src_batch, src_mask)
# #         samp = self.dec.sample(self, fb_src, nb_steps, src_mask, use_best_for_sample = use_best_for_sample,
# #                         keep_attn_values = keep_attn_values, need_score = need_score)
# #         return samp
# #         return self.dec.beam_search(fb_src, src_mask, nb_steps, eos_idx = eos_idx, beam_width = beam_width)
#         
#         
#     def get_predictor(self, src_batch, src_mask):
#         fb_src = self.enc(src_batch, src_mask)
#         return self.dec.get_predictor(fb_src, src_mask)
#        
#     def nbest_scorer(self, src_batch, src_mask):
#         assert len(src_batch[0].data) == 1
#         fb_concat = self.enc(src_batch, src_mask)
#         compute_ctxt = self.dec.attn_module.compute_ctxt_demux(fb_concat, src_mask)
#         
#         def scorer(tgt_batch):
#             return self.dec.compute_loss(tgt_batch, compute_ctxt, raw_loss_info = True, keep_attn_values = False, 
#                  noise_on_prev_word = False, use_previous_prediction = 0, mode = "test", per_sentence = True)
#     
#         return scorer
#       
#     def get_sampler_reinf(self, src_batch, src_mask, eos_idx, nb_steps = 50, use_best_for_sample = False,
#                     temperature = None,
#                     mode = "test"):
#         fb_concat = self.enc(src_batch, src_mask)
#         return self.dec.get_sampler_reinf(fb_concat, src_mask, eos_idx, nb_steps = nb_steps, 
#                     use_best_for_sample = use_best_for_sample,
#                     temperature = temperature,
#                     lexicon_probability_matrix = self.lexicon_probability_matrix,  
#                     lex_epsilon = self.lex_epsilon,
#                     mode = mode)
#         
#     def get_reinf_loss(self, src_batch, src_mask, eos_idx, test_references, nb_steps = 50, nb_samples = 5, use_best_for_sample = False,
#                     temperature = None,
#                     mode = "test"):
#         
#         mb_size = len(src_batch[0])
#         sampler = self.get_sampler_reinf(self, src_batch, src_mask, eos_idx, nb_steps = nb_steps, 
#                     use_best_for_sample = use_best_for_sample,
#                     temperature = None,
#                     mode = mode)
#         
#         from utils import de_batch
#         from bleu_computer import BleuComputer
#         total_score = 0
#         for i in xrange(nb_samples):
#             sentences, score = sampler.next()
#             
#             deb = de_batch(sentences, mask = None, eos_idx = eos_idx, is_variable = False)
#             
#             assert len(deb) == len(test_references)
#             assert len(deb) == mb_size
#             
#             bleu_vec = np.zeros((mb_size, 1), dtype = np.float32)
#             for num_t in range(len(deb)):
#                 t = deb[num_t]
#                 if t[-1] == eos_idx:
#                     t = t[:-1]
#                 ref = test_references[num_t]
#                 bc = BleuComputer()
#                 bc.update(ref, t)
#                 bleu = bc.bleu_plus_alpha(0.5)
#                 bleu_vec[num_t] = bleu
#             
#             if self.xp is not np:
#                 bleu_vec = cuda.to_gpu(bleu_vec)
#                 
#             weighted_score = F.matmul(score, Variable(bleu_vec, volatile = "auto"))
#             total_score += weighted_score
#                 
#         total_score /= nb_samples
#         return total_score

import decoder_cells
class EncoderDecoder(Chain):
    """ Do RNNSearch Encoding/Decoding
    
        Args:
            Vi: Source vocabulary size
            Ei: Size of Source word embeddings
            Hi: Size of source encoder hidden layer
            Vo: Target Vocabulary Size
            Eo: Size of Target word embeddings
            Ho: Size of Decoder hidden state
            Ha: Size of Attention mechanism hidden layer size
            Hl: Size of maxout output layer
            attn_cls: class of the Attention to be used
            
    
        The __call__ takes 3 required parameters: src_batch, tgt_batch, src_mask
        src_batch is as in the sequence parameter of Encoder
        tgt_batch is as in the targets parameter of Decoder
        src_mask is as in the mask parameter of Encoder
        
        return loss and attention values
    """
    
    
    def __init__(self, Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl, attn_cls = AttentionModule, init_orth = False, use_bn_length = 0,
                encoder_cell_type = rnn_cells.LSTMCell,
                decoder_cell_type = rnn_cells.LSTMCell,
                lexical_probability_dictionary = None, lex_epsilon = 1e-3
                ):
        log.info("constructing encoder decoder with Vi:%i Ei:%i Hi:%i Vo:%i Eo:%i Ho:%i Ha:%i Hl:%i" % 
                                        (Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl))
        super(EncoderDecoder, self).__init__(
            enc = Encoder(Vi, Ei, Hi, init_orth = init_orth, use_bn_length = use_bn_length,
                          cell_type = encoder_cell_type),
            dec = decoder_cells.Decoder(Vo, Eo, Ho, Ha, 2 * Hi, Hl, attn_cls = attn_cls, init_orth = init_orth, 
                          cell_type = decoder_cell_type)
        )
        self.Vo = Vo
        self.lexical_probability_dictionary = lexical_probability_dictionary
        self.lex_epsilon = lex_epsilon
        
    def compute_lexicon_probability_matrix(self, src_batch):
        if self.lexical_probability_dictionary is not None:
            lexicon_probability_matrix = compute_lexicon_matrix(src_batch, self.lexical_probability_dictionary, self.Vo)
            if self.xp != np:
                lexicon_probability_matrix = cuda.to_gpu(lexicon_probability_matrix, cuda.get_device(self.dec.lin_o.W.data))
        else:
            lexicon_probability_matrix = None
        return lexicon_probability_matrix
        
    def __call__(self, src_batch, tgt_batch, src_mask, use_best_for_sample = False, display_attn = False,
                 raw_loss_info = False, keep_attn_values = False, need_score = False, noise_on_prev_word = False,
                 use_previous_prediction = 0, mode = "test"
                 ):
        assert mode in "test train".split()

            
        lexicon_probability_matrix = self.compute_lexicon_probability_matrix(src_batch)
            
        fb_concat = self.enc(src_batch, src_mask, mode = mode)
        
        
        mb_size, nb_elems, Hi = fb_concat.data.shape

        if isinstance(tgt_batch, int):
            return self.dec.sample(fb_concat, src_mask, tgt_batch, mb_size, 
                                   lexicon_probability_matrix = lexicon_probability_matrix, 
                                   lex_epsilon = self.lex_epsilon, best = use_best_for_sample, 
                                   keep_attn_values = keep_attn_values, need_score = need_score)

        else:
            return self.dec.compute_loss(fb_concat, src_mask, tgt_batch, raw_loss_info = raw_loss_info,
                                         keep_attn_values = keep_attn_values, noise_on_prev_word = noise_on_prev_word,
                                         use_previous_prediction = use_previous_prediction, mode = "test", 
                                         per_sentence = False, lexicon_probability_matrix = lexicon_probability_matrix, 
                                         lex_epsilon = self.lex_epsilon)
                                         
    def give_conditionalized_cell(self, src_batch, src_mask, noise_on_prev_word = False,
                    mode = "test", demux = False):
             
        if self.lexical_probability_dictionary is not None:
            lexicon_probability_matrix = compute_lexicon_matrix(src_batch, self.lexical_probability_dictionary, self.Vo)
            if self.xp != np:
                lexicon_probability_matrix = cuda.to_gpu(lexicon_probability_matrix, cuda.get_device(self.dec.lin_o.W.data))
        else:
            lexicon_probability_matrix = None
            
        fb_concat = self.enc(src_batch, src_mask, mode = mode)
        
        
        mb_size, nb_elems, Hi = fb_concat.data.shape
        
        return self.dec.give_conditionalized_cell(fb_concat, src_mask, 
                    noise_on_prev_word = noise_on_prev_word,
                    mode = mode, lexicon_probability_matrix = lexicon_probability_matrix, 
                    lex_epsilon = self.lex_epsilon, demux = demux)                     

    def nbest_scorer(self, src_batch, src_mask, keep_attn = False):
        assert len(src_batch[0].data) == 1
        
        lexicon_probability_matrix = self.compute_lexicon_probability_matrix(src_batch)
        fb_concat = self.enc(src_batch, src_mask)
         
        decoding_cell = self.dec.give_conditionalized_cell(fb_concat, src_mask, noise_on_prev_word = False,
                    mode = "test", lexicon_probability_matrix = lexicon_probability_matrix, lex_epsilon = self.lex_epsilon,
                    demux = True)
        
        def scorer(tgt_batch):
            
            loss, attn_list = decoder_cells.compute_loss_from_decoder_cell(decoding_cell, tgt_batch, 
                                                         use_previous_prediction = 0,
                                                         raw_loss_info = True,
                                                         per_sentence = True,
                                                         keep_attn = keep_attn)
            
            return loss, attn_list
            
        return scorer

#     def compute_loss_and_backward(self, src_batch, tgt_batch, src_mask, raw_loss_info = False):
#         fb_src = self.enc(src_batch, src_mask)
#         loss = self.dec.compute_loss_and_backward(fb_src, tgt_batch, src_mask, raw_loss_info = raw_loss_info)
#         return loss
    

# class AttentionModuleAcumulated(Chain):
#     """ Attention Module for computing the current context during decoding. 
#         The __call_ takes 2 parameters: fb_concat and mask.
#         
#         fb_concat should be the result of a call to Encoder.
#         mask is as in the description of Encoder
#                
#         Return a chainer variable of shape (mb_size, Hi) and type float32
#     """
#     def __init__(self, Hi, Ha, Ho):
#         super(AttentionModuleAcumulated, self).__init__(
#             al_lin_h = L.Linear(Hi, Ha, nobias = False),
#             al_lin_s = L.Linear(Ho, Ha, nobias = True),
#             al_lin_o = L.Linear(Ha, 1, nobias = True)                                     
#         )
#         self.Hi = Hi
#         self.Ha = Ha
#         
#     def __call__(self, fb_concat, mask):
#         mb_size, nb_elems, Hi = fb_concat.data.shape
#         assert Hi == self.Hi
#         precomputed_al_factor = F.reshape(self.al_lin_h(
#                         F.reshape(fb_concat, (mb_size * nb_elems, self.Hi)))
#                                           , (mb_size, nb_elems, self.Ha))
#         
#         concatenated_mask = F.concat([F.reshape(mask_elem, (mb_size, 1)) for mask_elem in mask], 1)
#         
#         accumulation = [Variable(self.xp.zeros((mb_size, nb_elems), dtype = self.xp.float32), volatile = "auto")]
#         
#         def compute_ctxt(previous_state):
#             current_mb_size = previous_state.data.shape[0]
#             if current_mb_size < mb_size:
#                 al_factor, _ = F.split_axis(precomputed_al_factor, (current_mb_size,), 0)
#                 used_fb_concat, _ = F.split_axis(fb_concat, (current_mb_size,), 0)
#                 used_concatenated_mask, _ = F.split_axis(concatenated_mask, (current_mb_size,), 0)
#                 
#                 # Warning: here we are actually changing the global accumulation
#                 if current_mb_size < accumulation[0].data.shape[0]:
#                     accumulation[0], _ = F.split_axis(accumulation[0], (current_mb_size,), 0)
#             else:
#                 al_factor = precomputed_al_factor
#                 used_fb_concat = fb_concat
#                 used_concatenated_mask = concatenated_mask
#                 
#             state_al_factor = self.al_lin_s(previous_state)
#             state_al_factor_bc = F.broadcast_to(F.reshape(state_al_factor, (current_mb_size, 1, self.Ha)), (current_mb_size, nb_elems, self.Ha) )
#             a_coeffs = F.reshape(self.al_lin_o(F.reshape(F.tanh(state_al_factor_bc + al_factor), 
#                             (current_mb_size* nb_elems, self.Ha))), (current_mb_size, nb_elems))
#             
#             a_coeffs_mods = a_coeffs -  accumulation[0]
#             accumulation[0] += a_coeffs
#             
#             with cuda.get_device(used_concatenated_mask.data):
#                 a_coeffs_mods = a_coeffs_mods - 50000 * (1-used_concatenated_mask.data) 
#             
#             attn = F.softmax(a_coeffs_mods)
#             
#             ci = F.reshape(F.batch_matmul(attn, used_fb_concat, transa = True), (current_mb_size, self.Hi))
#             
#             return ci, attn
#         
#         return compute_ctxt
#     
#     def compute_ctxt_demux(self, fb_concat, mask):
#         raise NotImplemented
#         mb_size, nb_elems, Hi = fb_concat.data.shape
#         assert Hi == self.Hi
#         assert mb_size == 1
#         
#         precomputed_al_factor = F.reshape(self.al_lin_h(
#                         F.reshape(fb_concat, (mb_size * nb_elems, self.Hi)))
#                                           , (mb_size, nb_elems, self.Ha))
#         
#         concatenated_mask = F.concat([F.reshape(mask_elem, (mb_size, 1)) for mask_elem in mask], 1)  
#         
#         
#         def compute_ctxt(previous_state):
#             current_mb_size = previous_state.data.shape[0]
#                 
#             al_factor = F.broadcast_to(precomputed_al_factor, (current_mb_size, nb_elems, self.Ha))
# #             used_fb_concat = F.broadcast_to(fb_concat, (current_mb_size, nb_elems, Hi))
#             used_concatenated_mask = F.broadcast_to(concatenated_mask, (current_mb_size, nb_elems))
#                 
#             state_al_factor = self.al_lin_s(previous_state)
#             state_al_factor_bc = F.broadcast_to(F.reshape(state_al_factor, (current_mb_size, 1, self.Ha)), (current_mb_size, nb_elems, self.Ha) )
#             a_coeffs = F.reshape(self.al_lin_o(F.reshape(F.tanh(state_al_factor_bc + al_factor), 
#                             (current_mb_size* nb_elems, self.Ha))), (current_mb_size, nb_elems))
#             
#             
#             with cuda.get_device(used_concatenated_mask.data):
#                 a_coeffs = a_coeffs - 10000 * (1-used_concatenated_mask.data) 
#             
#             attn = F.softmax(a_coeffs)
#             
# #             ci = F.reshape(F.batch_matmul(attn, used_fb_concat, transa = True), (current_mb_size, self.Hi))
#             
#             ci = F.reshape(F.matmul(attn, F.reshape(fb_concat, (nb_elems, Hi))), (current_mb_size, self.Hi))
#             
#             return ci, attn
#         
#         return compute_ctxt
# 
# class DecoderWithAlign(Chain):
#     """ Decoder for RNNSearch. 
#         The __call_ takes 3 required parameters: fb_concat, targets, mask.
#         
#         fb_concat should be the result of a call to Encoder.
#         
#         targets is a python list of chainer variables of type int32 and of variable shape (n,)
#             the values n should be decreasing:
#                 i < j => targets[i].data.shape[0] >= targets[j].data.shape[0]
#             targets[i].data[j] is the jth elements of the ith sequence in the minibatch
#             all this imply that the sequences of the minibatch should be sorted from longest to shortest
#             
#         mask is as in the description of Encoder.
#         
#         * it is up to the user to add an EOS token to the data.
#                
#         Return a loss and the attention model values
#     """
#     def __init__(self, Vo, Eo, Ho, Ha, Hi, Hl, attn_cls = AttentionModule, init_orth = False):
#         super(DecoderWithAlign, self).__init__(
#             emb = L.EmbedID(Vo, Eo),
#             gru = L.GRU(Ho, Eo + Hi),
#             
#             maxo = L.Maxout(Eo + Hi + Ho, Hl, 2),
#             lin_o = L.Linear(Hl, Vo, nobias = False),
#             
#             attn_module = attn_cls(Hi, Ha, Ho, init_orth = init_orth),
#             
#             align_pred_mo = L.Maxout(Ho, 1,4),
#             size_pred_mo = L.Maxout(Hi, 2, 2)
#         )
#         self.add_param("initial_state", (1, Ho))
#         self.add_param("bos_embeding", (1, Eo))
#         self.Hi = Hi
#         self.Ho = Ho
#         self.Eo = Eo
#         self.initial_state.data[...] = np.random.randn(Ho)
#         self.bos_embeding.data[...] = np.random.randn(Eo)
#         
#         if init_orth:
#             ortho_init(self.gru)
#             ortho_init(self.lin_o)
#             ortho_init(self.maxo)
#         
#     def advance_one_step(self, previous_state, prev_y, compute_ctxt):
# 
#         ci, attn = compute_ctxt(previous_state)
#         concatenated = F.concat( (prev_y, ci) )
# #             print concatenated.data.shape
#         new_state = self.gru(previous_state, concatenated)
# 
#         all_concatenated = F.concat((concatenated, new_state))
#         logits = self.lin_o(self.maxo(all_concatenated))
#         
#         al_pred =  self.align_pred_mo(new_state)
#         
#         return new_state, logits, attn, al_pred
#           
#     def compute_loss(self, targets, fb_concat, mask, sizes, raw_loss_info = False, keep_attn_values = False, noise_on_prev_word = False):
#         compute_ctxt = self.attn_module(fb_concat, mask)
#         loss = None
#         current_mb_size = targets[0].data.shape[0]
# #         previous_state = F.concat( [self.initial_state] * current_mb_size, 0)
#         previous_state = F.broadcast_to(self.initial_state, (current_mb_size, self.Ho))
# #         previous_word = Variable(np.array([self.bos_idx] * mb_size, dtype = np.int32))
#         xp = cuda.get_array_module(self.initial_state.data)
#         previous_word = None
#         with cuda.get_device(self.initial_state.data):
# #             previous_word = Variable(xp.array([self.bos_idx] * current_mb_size, dtype = np.int32))
#             prev_y = F.broadcast_to(self.bos_embeding, (current_mb_size, self.Eo))
#         attn_list = []
#         total_nb_predictions = 0
#         
#         if noise_on_prev_word:
#             noise_mean = Variable(self.xp.ones_like(prev_y.data, dtype = self.xp.float32))
#             noise_lnvar = Variable(self.xp.zeros_like(prev_y.data, dtype = self.xp.float32))
#         
#         first_fb, _ = F.split_axis(fb_concat, (1,), 1)
#         size_pred = F.softplus(self.size_pred_mo(first_fb))
#         size_mean, size_precision = F.split_axis(size_pred, (1,), 1)
#         loss_size = 0.5 * (size_precision* (sizes - size_mean) ) **2 - F.log(size_precision)
#         loss = F.sum(loss_size)/ current_mb_size
#         
#         for i in xrange(len(targets)):
#             assert i == 0 or previous_state.data.shape[0] == previous_word.data.shape[0]
#             current_mb_size = targets[i].data.shape[0]
#             if current_mb_size < len(previous_state.data):
#                 previous_state, _ = F.split_axis(previous_state, (current_mb_size,), 0)
#                 if previous_word is not None:
#                     previous_word, _ = F.split_axis(previous_word, (current_mb_size,), 0 )
#                     
#                 if noise_on_prev_word:
#                     noise_mean, _ = F.split_axis(noise_mean, (current_mb_size,), 0)
#                     noise_lnvar, _ = F.split_axis(noise_lnvar, (current_mb_size,), 0)
#             if previous_word is not None: #else we are using the initial prev_y
#                 prev_y = self.emb(previous_word)
#             assert previous_state.data.shape[0] == current_mb_size
#             
#             if noise_on_prev_word:
#                 prev_y = prev_y * F.gaussian(noise_mean, noise_lnvar)
#             
#             new_state, logits, attn, al_pred = self.advance_one_step(previous_state, prev_y, 
#                                                       compute_ctxt)
# 
#             if keep_attn_values:
#                 attn_list.append(attn)
#                 
#             local_loss = F.softmax_cross_entropy(logits, targets[i][0])   
#             
#             local_loss += F.softmax_cross_entropy(al_pred, targets[i][1])
#             
#             total_nb_predictions += current_mb_size
#             total_local_loss = local_loss * current_mb_size
#             
# #             loss = local_loss if loss is None else loss + local_loss
#             loss = total_local_loss if loss is None else loss + total_local_loss
#             
#             previous_word = targets[i]
# #             prev_y = self.emb(previous_word)
#             previous_state = new_state
# #             attn_list.append(attn)
#         if raw_loss_info:
#             return (loss, total_nb_predictions), attn_list
#         else:
#             loss = loss / total_nb_predictions
#             return loss, attn_list
               
# class EncoderDecoderPredictAlign(Chain):
#     """ Do RNNSearch Encoding/Decoding
#         The __call__ takes 3 required parameters: src_batch, tgt_batch, src_mask
#         src_batch is as in the sequence parameter of Encoder
#         tgt_batch is as in the targets parameter of Decoder
#         src_mask is as in the mask parameter of Encoder
#         
#         return loss and attention values
#     """
#     def __init__(self, Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl, attn_cls = AttentionModule, init_orth = False):
#         log.info("constructing encoder decoder with Vi:%i Ei:%i Hi:%i Vo:%i Eo:%i Ho:%i Ha:%i Hl:%i" % 
#                                         (Vi, Ei, Hi, Vo, Eo, Ho, Ha, Hl))
#         super(EncoderDecoderPredictAlign, self).__init__(
#             enc = Encoder(Vi, Ei, Hi, init_orth = init_orth),
#             dec = DecoderWithAlign(Vo, Eo, Ho, Ha, 2 * Hi, Hl, attn_cls = attn_cls, init_orth = init_orth)
#         )
#         
#     def __call__(self, src_batch, tgt_batch, src_mask, sizes, use_best_for_sample = False, display_attn = False,
#                  raw_loss_info = False, keep_attn_values = False, need_score = False, noise_on_prev_word = False):
#         fb_src = self.enc(src_batch, src_mask)
#         loss = self.dec.compute_loss(tgt_batch, fb_src, src_mask, sizes, use_best_for_sample = use_best_for_sample, raw_loss_info = raw_loss_info,
#                         keep_attn_values = keep_attn_values, need_score = need_score, noise_on_prev_word = noise_on_prev_word)
#         return loss 
        
