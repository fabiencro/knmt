#!/usr/bin/env python
"""models_tests.py: Some correctness tests"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import models

import logging
logging.basicConfig()
log = logging.getLogger("rnns:models_test")
log.setLevel(logging.INFO)



class EncoderNaive(models.Encoder):
    def __init__(self, Vi, Ei, Hi):
        super(EncoderNaive, self).__init__(Vi, Ei, Hi)
        
    def naive_call(self, sequence, mask):
        
        mb_size = sequence[0].data.shape[0]
        
        mb_initial_state_f = F.broadcast_to(F.reshape(self.initial_state_f, (1, self.Hi)), (mb_size, self.Hi))
        mb_initial_state_b = F.broadcast_to(F.reshape(self.initial_state_b, (1, self.Hi)), (mb_size, self.Hi))
        
        embedded_seq = []
        for elem in sequence:
            embedded_seq.append(self.emb(elem))
            
#         self.gru_f.reset_state()
        prev_state = mb_initial_state_f
        forward_seq = []
        for x in embedded_seq:
            prev_state = self.gru_f(prev_state, x)
            forward_seq.append(prev_state)
            
#         self.gru_b.reset_state()
        prev_state = mb_initial_state_b
        backward_seq = []
        for pos, x in reversed(list(enumerate(embedded_seq))):
            prev_state = self.gru_b(prev_state, x)
            backward_seq.append(prev_state)
        
        assert len(backward_seq) == len(forward_seq)
        res = []
        for xf, xb in zip(forward_seq, reversed(backward_seq)):
            res.append(F.concat((xf, xb), 1))
        
        return res
    
    
class AttentionModuleNaive(models.AttentionModule):
    def __init__(self, Hi, Ha, Ho):
        super(AttentionModuleNaive, self).__init__(Hi, Ha, Ho)
        
    def naive_call(self, fb_concat, mask):
#         mb_size, nb_elems, Hi = fb_concat.data.shape
#         
        nb_elems = len(fb_concat)
        mb_size, Hi = fb_concat[0].data.shape
        assert Hi == self.Hi
        
        precomputed_al_factor = []
        for x in fb_concat:
            precomputed_al_factor.append(self.al_lin_h(x))
        
        def compute_ctxt(previous_state):
            current_mb_size = previous_state.data.shape[0]
            if current_mb_size < mb_size:
                assert False
                al_factor, _ = F.split_axis(precomputed_al_factor, (current_mb_size,), 0)
                used_fb_concat, _ = F.split_axis(fb_concat, (current_mb_size,), 0)
#                 used_concatenated_mask, _ = F.split_axis(concatenated_mask, (current_mb_size,), 0)
            else:
                al_factor = precomputed_al_factor
                used_fb_concat = fb_concat
#                 used_concatenated_mask = concatenated_mask
                
            state_al_factor = self.al_lin_s(previous_state)
            
            a_coeffs = []
            for x in al_factor:
                a_coeffs.append(self.al_lin_o(F.tanh(x + state_al_factor)))
#             state_al_factor_bc = F.broadcast_to(F.reshape(state_al_factor, (current_mb_size, 1, self.Ha)), (current_mb_size, nb_elems, self.Ha) )
#             a_coeffs = F.reshape(self.al_lin_o(F.reshape(F.tanh(state_al_factor_bc + al_factor), 
#                             (current_mb_size* nb_elems, self.Ha))), (current_mb_size, nb_elems))
            
#             a_coeffs = a_coeffs - 10000 * (1-used_concatenated_mask.data) 
            a_coeffs_concat = F.concat(a_coeffs, 1)
            assert a_coeffs_concat.data.shape == (mb_size, nb_elems)
            attn = F.softmax(a_coeffs_concat)
            
            splitted_attn = F.split_axis(attn, len(fb_concat), 1)
            ci = None
            for i in xrange(nb_elems):
                contrib = F.broadcast_to(splitted_attn[i], (mb_size, Hi)) * used_fb_concat[i]
                ci = ci + contrib if ci is not None else contrib
#             ci = F.reshape(F.batch_matmul(attn, used_fb_concat, transa = True), (current_mb_size, self.Hi))
            
            return ci, attn
        
        return compute_ctxt
    
class DecoderNaive(models.Decoder):
    def __init__(self, Vo, Eo, Ho, Ha, Hi, Hl):
        super(DecoderNaive, self).__init__(Vo, Eo, Ho, Ha, Hi, Hl)
        
    def naive_call(self, fb_concat, targets, mask, use_best_for_sample = False):
        compute_ctxt = self.attn_module.naive_call(fb_concat, mask)
        loss = None
        current_mb_size = targets[0].data.shape[0]
        previous_state = F.concat( [self.initial_state] * current_mb_size, 0)
#         previous_word = Variable(np.array([self.bos_idx] * mb_size, dtype = np.int32))
        xp = cuda.get_array_module(self.initial_state.data)
        with cuda.get_device(self.initial_state.data):
            previous_word = Variable(xp.array([self.bos_idx] * current_mb_size, dtype = np.int32))
        
        attn_list = []
        for i in xrange(len(targets)):
#             assert previous_state.data.shape[0] == previous_word.data.shape[0]
#             current_mb_size = targets[i].data.shape[0]
#             if current_mb_size < len(previous_state.data):
#                 previous_state, _ = F.split_axis(previous_state, (current_mb_size,), 0)
#                 previous_word, _ = F.split_axis(previous_word, (current_mb_size,), 0 )
#             assert previous_state.data.shape[0] == current_mb_size
#             print "i", i
#             new_state, logits, attn = self.advance_one_step(fb_concat, previous_state, previous_word, 
#                                                       compute_ctxt)

            ci, attn = compute_ctxt(previous_state)
            prev_y = self.emb(previous_word)
            concatenated = F.concat( (prev_y, ci) )
    #             print concatenated.data.shape
            new_state = self.gru(previous_state, concatenated)
    
            all_concatenated = F.concat((concatenated, new_state))
            logits = self.lin_o(self.maxo(all_concatenated))

#             print type(logits.data), type(targets[i].data)
            local_loss = F.softmax_cross_entropy(logits, targets[i])
                                    
            loss = local_loss if loss is None else loss + local_loss
            
            previous_word = targets[i]
            previous_state = new_state
            attn_list.append(attn)
        return loss, attn_list
        
        
class EncoderDecoderNaive(models.EncoderDecoder):
    def __init__(self, Vo, Eo, Ho, Ha, Hi, Hl):
        super(EncoderDecoderNaive, self).__init__(Vo, Eo, Ho, Ha, Hi, Hl)
        
    def naive_call(self, src_batch, tgt_batch, src_mask, use_best_for_sample = False, display_attn = False):
        fb_src = self.enc.naive_call(src_batch, src_mask)
        loss = self.dec.naive_call(fb_src, tgt_batch, src_mask, use_best_for_sample = use_best_for_sample)
        return loss 