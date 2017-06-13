#!/usr/bin/env python
"""attention.py: Implementation of Attention mechanisms"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

from chainer import cuda, Variable
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer
from nmt_chainer.utilities.utils import ortho_init
import numpy as np
import logging
logging.basicConfig()
log = logging.getLogger("rnns:attn")
log.setLevel(logging.INFO)


def softmax_mask_maker(mask, seq_length, mb_size):
    mask_length = len(mask)
    assert mask_length <= seq_length
    mask_offset = seq_length - mask_length
    
    if mask_length > 0:
        xp = chainer.cuda.get_array_module(mask[0])
        device = cuda.get_device(mask[0])
        with device:
            if mask_offset > 0:
                concatenated_penalties = xp.concatenate(
                    [
                        xp.zeros((mb_size, mask_offset), dtype=xp.float32),
                        -10000 * (1 - xp.concatenate([
                            xp.reshape(mask_elem, (mb_size, 1)).astype(xp.float32) for mask_elem in mask], 1))
                    ], 1
                )
            else:
                concatenated_penalties = -10000 * (1 - xp.concatenate([
                    xp.reshape(mask_elem, (mb_size, 1)).astype(xp.float32) for mask_elem in mask], 1))    
    
    def apply_mask(logits):
        current_mb_size = logits.data.shape[0]
        assert current_mb_size <= mb_size
        if mask_length == 0:
            return logits
        elif current_mb_size == mb_size:
            with device:
                result = logits + concatenated_penalties
            return result
        else:
            with device:
                result = logits + concatenated_penalties[:current_mb_size]
            return result
        
    return apply_mask
  
class PointerMechanismComputer(object):
    def __init__(self, pointer_mechanism_chain, fb_concat, mask, demux=False, represented_data=None):
        mb_size, nb_elems, Hi = fb_concat.data.shape
        assert Hi == pointer_mechanism_chain.Hi
        assert mb_size == 1 or not demux
        
        self.pointer_mechanism_chain = pointer_mechanism_chain
        
        if self.pointer_mechanism_chain.nb_sentinels > 0:
            broadcasted_sentinels = F.broadcast_to(self.pointer_mechanism_chain.sentinels, 
                                                (mb_size, self.pointer_mechanism_chain.nb_sentinels, Hi))
            fb_concat = F.concat((broadcasted_sentinels, fb_concat))
            nb_elems += self.pointer_mechanism_chain.nb_sentinels
        
        self.nb_elems = nb_elems
        self.mb_size = mb_size
        self.fb_concat = fb_concat
        
        if represented_data is None:
            self.represented_data = fb_concat
        else:
            
            broadcasted_represented_data_sentinels = F.broadcast_to(self.pointer_mechanism_chain.represented_data_sentinel, 
                                                (mb_size, self.pointer_mechanism_chain.nb_sentinels, self.pointer_mechanism_chain.represented_data_sentinel.shape[-1]))
            self.represented_data = F.concat((broadcasted_represented_data_sentinels, represented_data))
            assert self.represented_data.shape[:2] == fb_concat.shape[:2]
        
        self.demux = demux
        self.precomputed_al_factor = F.reshape(self.pointer_mechanism_chain.lin_i(
                F.reshape(fb_concat, (mb_size * nb_elems, self.pointer_mechanism_chain.Hi))), 
                                               (mb_size, nb_elems, self.pointer_mechanism_chain.Ha))

        self.xp = chainer.cuda.get_array_module(fb_concat.data)
        seq_length = nb_elems

        if not demux:
            self.apply_softmax_mask = softmax_mask_maker(mask, seq_length, mb_size)      
    
    def compute_pointer(self, previous_state, additional_input=None):
        current_mb_size = previous_state.data.shape[0]
        
        if self.demux:
            al_factor = F.broadcast_to(self.precomputed_al_factor, (current_mb_size, self.nb_elems, self.pointer_mechanism_chain.Ha))
        else:
            if current_mb_size < self.mb_size:
                al_factor, _ = F.split_axis(
                    self.precomputed_al_factor, (current_mb_size,), 0)
            else:
                al_factor = self.precomputed_al_factor

        

        state_al_factor = self.pointer_mechanism_chain.lin_s(previous_state)
        
        #As suggested by Isao Goto
        if additional_input is not None:
            state_al_factor = state_al_factor + self.pointer_mechanism_chain.lin_add(additional_input)
            
        state_al_factor_bc = F.broadcast_to(F.reshape(state_al_factor, (current_mb_size, 1, self.pointer_mechanism_chain.Ha)), 
                                            (current_mb_size, self.nb_elems, self.pointer_mechanism_chain.Ha))
        a_coeffs = F.reshape(self.pointer_mechanism_chain.lin_o(F.reshape(F.tanh(state_al_factor_bc + al_factor),
                                                     (current_mb_size * self.nb_elems, self.pointer_mechanism_chain.Ha))),
                                                      (current_mb_size, self.nb_elems))

        if not self.demux:
            a_coeffs = self.apply_softmax_mask(a_coeffs)

        pointer = F.softmax(a_coeffs)
        
        return pointer         

    def compute_ctxt(self, previous_state, prev_word_embedding=None):
        current_mb_size = previous_state.data.shape[0]
        if not self.demux:
            if current_mb_size < self.mb_size:
                used_fb_concat, _ = F.split_axis(
                    self.fb_concat, (current_mb_size,), 0)
            else:
                used_fb_concat = self.fb_concat

        attn = self.compute_pointer(previous_state, prev_word_embedding=prev_word_embedding)
        
        if self.demux:
            ci = F.reshape(F.matmul(attn, F.reshape(self.fb_concat, (self.nb_elems, self.pointer_mechanism_chain.Hi))), 
                           (current_mb_size, self.pointer_mechanism_chain.Hi))
        else:
            ci = F.reshape(F.batch_matmul(attn, used_fb_concat, transa=True), (current_mb_size, self.pointer_mechanism_chain.Hi))
            
        return ci, attn
    
    def get_repr_ptr(self, idx):
        mb_size = idx.shape[0]
        assert mb_size <= self.mb_size
        idx_range = self.xp.array(range(mb_size), dtype=self.xp.int32)
        return self.represented_data[idx_range, idx]

            
class PointerMechanism(Chain):
    def __init__(self, Hi, Hs, Ha, additional_input_size = None, nb_sentinels=0, size_represented_data_sentinel=None):
        super(PointerMechanism, self).__init__(
            lin_i = L.Linear(Hi, Ha, nobias=False),
            lin_s = L.Linear(Hs, Ha, nobias=True),
            lin_o = L.Linear(Ha, 1, nobias=True)
            )
        
        self.Hi = Hi
        self.Hs = Hs
        self.Ha = Ha
        
        if additional_input_size is not None:
            self.add_link("lin_add", L.Linear(additional_input_size, Ha, nobias=True))
            
        self.nb_sentinels = nb_sentinels
        if self.nb_sentinels > 0:
            self.add_param("sentinels", (1, self.nb_sentinels, Hi))
            self.sentinels.data[...] = np.random.randn(1, self.nb_sentinels, Hi)
            
        if size_represented_data_sentinel is not None:
            self.add_param("represented_data_sentinel", (1, self.nb_sentinels, size_represented_data_sentinel))
            self.represented_data_sentinel.data[...] = np.random.randn(1, self.nb_sentinels, size_represented_data_sentinel)
            
        
    def __call__(self, fb_concat, mask, demux=False, represented_data=None):
        return PointerMechanismComputer(self, fb_concat, mask, demux=demux, represented_data=represented_data)
  


class AttentionModule(Chain):
    """ Attention Module for computing the current context during decoding.
        The __call_ takes 2 parameters: fb_concat and mask.

        fb_concat should be the result of a call to Encoder.
        mask is as in the description of Encoder

        Return a chainer variable of shape (mb_size, Hi) and type float32
    """

    def __init__(self, Hi, Ha, Ho, init_orth=False, prev_word_embedding_size = None):
        super(AttentionModule, self).__init__(
            al_lin_h=L.Linear(Hi, Ha, nobias=False),
            al_lin_s=L.Linear(Ho, Ha, nobias=True),
            al_lin_o=L.Linear(Ha, 1, nobias=True)
        )
        self.Hi = Hi
        self.Ha = Ha

        if prev_word_embedding_size is not None:
            self.add_link("al_lin_y", L.Linear(prev_word_embedding_size, Ha))

        if init_orth:
            ortho_init(self.al_lin_h)
            ortho_init(self.al_lin_s)
            ortho_init(self.al_lin_o)

    def __call__(self, fb_concat, mask):
        mb_size, nb_elems, Hi = fb_concat.data.shape
        assert Hi == self.Hi
        precomputed_al_factor = F.reshape(self.al_lin_h(
            F.reshape(fb_concat, (mb_size * nb_elems, self.Hi))), (mb_size, nb_elems, self.Ha))

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
                            self.xp.zeros((mb_size, mask_offset), dtype=self.xp.float32),
                            -10000 * (1 - self.xp.concatenate([
                                self.xp.reshape(mask_elem, (mb_size, 1)).astype(self.xp.float32) for mask_elem in mask], 1))
                        ], 1
                    )
                else:
                    concatenated_penalties = -10000 * (1 - self.xp.concatenate([
                        self.xp.reshape(mask_elem, (mb_size, 1)).astype(self.xp.float32) for mask_elem in mask], 1))

        def compute_ctxt(previous_state, prev_word_embedding=None):
            current_mb_size = previous_state.data.shape[0]
            if current_mb_size < mb_size:
                al_factor, _ = F.split_axis(
                    precomputed_al_factor, (current_mb_size,), 0)
                used_fb_concat, _ = F.split_axis(
                    fb_concat, (current_mb_size,), 0)
                if mask_length > 0:
                    used_concatenated_penalties = concatenated_penalties[:current_mb_size]
            else:
                al_factor = precomputed_al_factor
                used_fb_concat = fb_concat
                if mask_length > 0:
                    used_concatenated_penalties = concatenated_penalties

            state_al_factor = self.al_lin_s(previous_state)
            
            #As suggested by Isao Goto
            if prev_word_embedding is not None:
                state_al_factor = state_al_factor + self.al_lin_y(prev_word_embedding)
                
            state_al_factor_bc = F.broadcast_to(F.reshape(state_al_factor, (current_mb_size, 1, self.Ha)), (current_mb_size, nb_elems, self.Ha))
            a_coeffs = F.reshape(self.al_lin_o(F.reshape(F.tanh(state_al_factor_bc + al_factor),
                                                         (current_mb_size * nb_elems, self.Ha))), (current_mb_size, nb_elems))

            if mask_length > 0:
                with cuda.get_device(used_concatenated_penalties):
                    a_coeffs = a_coeffs + used_concatenated_penalties  # - 10000 * (1-used_concatenated_mask.data)

            attn = F.softmax(a_coeffs)

            ci = F.reshape(F.batch_matmul(attn, used_fb_concat, transa=True), (current_mb_size, self.Hi))

            return ci, attn

        return compute_ctxt

    def compute_ctxt_demux(self, fb_concat, mask):
        mb_size, nb_elems, Hi = fb_concat.data.shape
        assert Hi == self.Hi
        assert mb_size == 1
        assert len(mask) == 0

        precomputed_al_factor = F.reshape(self.al_lin_h(
            F.reshape(fb_concat, (mb_size * nb_elems, self.Hi))), (mb_size, nb_elems, self.Ha))

#         concatenated_mask = F.concat([F.reshape(mask_elem, (mb_size, 1)) for mask_elem in mask], 1)

        def compute_ctxt(previous_state, prev_word_embedding=None):
            current_mb_size = previous_state.data.shape[0]

            al_factor = F.broadcast_to(precomputed_al_factor, (current_mb_size, nb_elems, self.Ha))
#             used_fb_concat = F.broadcast_to(fb_concat, (current_mb_size, nb_elems, Hi))
#             used_concatenated_mask = F.broadcast_to(concatenated_mask, (current_mb_size, nb_elems))

            state_al_factor = self.al_lin_s(previous_state)
            
            #As suggested by Isao Goto
            if prev_word_embedding is not None:
                state_al_factor = state_al_factor + self.al_lin_y(prev_word_embedding)
            
            state_al_factor_bc = F.broadcast_to(F.reshape(state_al_factor, (current_mb_size, 1, self.Ha)), (current_mb_size, nb_elems, self.Ha))
            a_coeffs = F.reshape(self.al_lin_o(F.reshape(F.tanh(state_al_factor_bc + al_factor),
                                                         (current_mb_size * nb_elems, self.Ha))), (current_mb_size, nb_elems))


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

    def __init__(self, Hi, Ha, Ho, init_orth=False, prev_word_embedding_size = None):
        if prev_word_embedding_size is not None:
            raise NotImplemented
        log.info("using deep attention")
        super(DeepAttentionModule, self).__init__(
            attn1=AttentionModule(Hi, Ha, Ho, init_orth=init_orth),
            attn2=AttentionModule(Hi, Ha, Ho + Hi, init_orth=init_orth)
        )

    def __call__(self, fb_concat, mask):
        compute_ctxt1 = self.attn1(fb_concat, mask)
        compute_ctxt2 = self.attn2(fb_concat, mask)

        def compute_ctxt(previous_state):
            ci1, attn1 = compute_ctxt1(previous_state)
            intermediate_state = F.concat((previous_state, ci1), axis=1)
            ci2, attn2 = compute_ctxt2(intermediate_state)

            return ci2, attn2

        return compute_ctxt

    def compute_ctxt_demux(self, fb_concat, mask):
        raise NotImplemented



class CopyMechanism(Chain):
    def __init__(self, Hi, Ho):
        super(CopyMechanism, self).__init__(
            lin=L.Linear(Hi, Ho)
        )
        self.Hi = Hi
        self.Ho = Ho

    def __call__(self, inpt, mask):
        mb_size = inpt.data.shape[0]
        max_length = inpt.data.shape[1]

        precomp = F.reshape(F.tanh(self.lin(F.reshape(inpt, (-1, self.Hi)))), (mb_size, -1, self.Ho))

        mask_offset = max_length - len(mask)

        precomp_mask_penalties = self.xp.concatenate(
            [
                self.xp.zeros((mb_size, mask_offset), dtype=self.xp.float32),
                -10000 * (1 - self.xp.concatenate([
                    self.xp.reshape(mask_elem, (mb_size, 1)).astype(self.xp.float32) for mask_elem in mask], 1))
            ], 1
        )

        def compute_copy_coefficients(state):
            betas = F.reshape(F.batch_matmul(precomp, state), (mb_size, -1))
            masked_betas = betas + precomp_mask_penalties
            return masked_betas

        return compute_copy_coefficients
