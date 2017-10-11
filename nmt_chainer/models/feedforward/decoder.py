import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Chain, ChainList

from nmt_chainer.models.feedforward.utils import (
    generate_pos_vectors, make_batch_mask, pad_data, FeedForward, apply_linear_layer_to_last_dims, cut_minibatch)
from nmt_chainer.models.feedforward.multi_attention import AddAndNormalizedSelfAttentionLayer, AddAndNormalizedCrossAttentionLayer

import logging
logging.basicConfig()
log = logging.getLogger("ff:dec")
log.setLevel(logging.INFO)

class DecoderLayer(Chain):
    def __init__(self, d_model, n_heads, d_ff=2048, experimental_relu=False, dropout=None,
                 residual_mode="normal", no_normalize=False):
        super(DecoderLayer, self).__init__(
            ff_layer = FeedForward(d_model, d_ff=d_ff, dropout=dropout, residual_mode=residual_mode, no_normalize=no_normalize),
            self_attention_layer = AddAndNormalizedSelfAttentionLayer(d_model=d_model, n_heads=n_heads,
                                                             experimental_relu=experimental_relu,
                                                          dropout=dropout, residual_mode=residual_mode, no_normalize=no_normalize),
            
            cross_attention_layer = AddAndNormalizedCrossAttentionLayer(d_model=d_model, n_heads=n_heads,
                                                             experimental_relu=experimental_relu,
                                                          dropout=dropout, 
                                        residual_mode=residual_mode if residual_mode is not "none" else "normal", no_normalize=no_normalize) # Does not seem good to not let the cross attention be bypassed
        )
        
        self.n_heads = n_heads
        self.d_model = d_model
    
    def __call__(self, tgt, src, mask, mask_input):
        y1 = self.self_attention_layer(tgt, mask)
        y2 = self.cross_attention_layer(y1, src, mask_input)
        y3 = self.ff_layer(y2)
        return y3
    
    def one_step(self, new_inpt, prev_states, src, mask_input):
        mb_size_inpt, nQ_inpt, d_model_inpt = new_inpt.data.shape
        assert nQ_inpt == 1
        assert d_model_inpt == self.d_model
        
        mb_size_mask_input, n_heads_mask_input, nQ_mask_input, nV_mask_input = mask_input.shape
        assert mb_size_mask_input == mb_size_inpt
        assert nQ_mask_input == 1
        assert n_heads_mask_input == self.n_heads
        
        mb_size_src, max_length_src, d_model_src = src.data.shape
        assert max_length_src == nV_mask_input
        assert mb_size_src == mb_size_inpt
        
#         assert mask.shape == (mb_size_inpt, self.n_heads, 1, )
        if prev_states is not None:
            prev_self_attn, prev_cross_attn = prev_states
            full_tgt = F.concat((prev_self_attn, new_inpt) , axis=1)
        else:
            full_tgt = new_inpt
            
        y1_last = self.self_attention_layer(full_tgt, mask = None, only_last=True)
        
        if prev_states is not None:
            full_y1 = F.concat((prev_cross_attn, y1_last) , axis=1)
        else:
            full_y1 = y1_last
            
        y2_last = self.cross_attention_layer(full_y1, src, mask_input, only_last=True)
        y3_last = self.ff_layer(y2_last)
        return y3_last, (full_tgt, full_y1)
    
class DecoderMultiLayer(ChainList):
    def __init__(self, d_model, n_heads, d_ff=2048, experimental_relu=False, dropout=None, nb_layers=6,
                 residual_mode="normal", no_normalize=False):
        super(DecoderMultiLayer, self).__init__()
        for _ in range(nb_layers):
            self.add_link(DecoderLayer(d_model, n_heads, d_ff=d_ff, experimental_relu=experimental_relu, dropout=dropout,
                                       residual_mode=residual_mode, no_normalize=no_normalize))
        
    def __call__(self, tgt, src, mask, mask_input):
        for link in self:
            tgt = link(tgt, src, mask, mask_input)
        return tgt
    
    def one_step(self, new_inpt, prev_states, src, mask_input):
        assert prev_states is None or len(prev_states) == len(self)
        new_prev_tgt = []
        tgt_last = new_inpt
        for num_link, link in enumerate(self):
            tgt_last, this_prev_tgt = link.one_step(tgt_last, prev_states[num_link] if prev_states is not None else None, src, mask_input)
            new_prev_tgt.append(this_prev_tgt)
        return tgt_last, tuple(new_prev_tgt)      
    
# class FakeDecoderMultiLayer(Chain):
#     def __init__(self, d_model, n_heads, d_ff=2048, experimental_relu=False, dropout=None, nb_layers=6,
#                  no_add=False, no_normalize=False):
#         super(DecoderMultiLayer, self).__init__(    
#             ff_layer1 = FeedForward(d_model, d_ff=d_ff, dropout=dropout, no_add=no_add, no_normalize=no_normalize),
#             self_attention_layer1 = AddAndNormalizedSelfAttentionLayer(d_model=d_model, n_heads=n_heads,
#                                                              experimental_relu=experimental_relu,
#                                                           dropout=dropout, no_add=no_add, no_normalize=no_normalize),
#             ff_layer2 = FeedForward(d_model, d_ff=d_ff, dropout=dropout, no_add=no_add, no_normalize=no_normalize),
#             self_attention_layer2 = AddAndNormalizedSelfAttentionLayer(d_model=d_model, n_heads=n_heads,
#                                                              experimental_relu=experimental_relu,
#                                                           dropout=dropout, no_add=no_add, no_normalize=no_normalize),
#             ff_layer3 = FeedForward(d_model, d_ff=d_ff, dropout=dropout, no_add=no_add, no_normalize=no_normalize),
#             self_attention_layer3 = AddAndNormalizedSelfAttentionLayer(d_model=d_model, n_heads=n_heads,
#                                                              experimental_relu=experimental_relu,
#                                                           dropout=dropout, no_add=no_add, no_normalize=no_normalize)
#             )
#         
#     def __call__(self, tgt, src, mask, mask_input, train=True):
#         y1 = self.ff_layer1(self.self_attention_layer1(tgt, mask))
#         y2 = self.ff_layer2(self.self_attention_layer2(y1, mask))
#         y3 = self.ff_layer3(self.self_attention_layer3(y2, mask))
#         return y3
#     
#     def one_step(self, new_inpt, prev_states, src, mask_input, train=True):
#         assert prev_states is None or len(prev_states) == len(self)
#         new_prev_tgt = []
#         tgt_last = new_inpt
#         for num_link, link in enumerate(self):
#             tgt_last, this_prev_tgt = link.one_step(tgt_last, prev_states[num_link] if prev_states is not None else None, src, mask_input, train=train)
#             new_prev_tgt.append(this_prev_tgt)
#         return tgt_last, tuple(new_prev_tgt)    
    
class DecoderState(object):
    def __init__(self, pos, prev_states):
        self.pos = pos
        self.prev_states = prev_states
        self.mb_size = self.prev_states[0][0].data.shape[0]
        assert [st.data.shape[0] == self.mb_size for state_group in prev_states for st in state_group]
        assert isinstance(pos, int) and pos >= -1
        
    def get_mb_size(self):
        return self.mb_size
    
    def reduce_to_minibatch_size(self, new_minibatch_size):
        assert new_minibatch_size <= self.mb_size
        if new_minibatch_size == self.mb_size:
            return self
        else:
            splitted_states = []
            for state_group in self.prev_states:
                splitted_states.append(
                    tuple(cut_minibatch(st, new_minibatch_size) for st in state_group)
                    )
            return DecoderState(self.pos, tuple(splitted_states))
    
    def get_states(self):
        return self.prev_states
    
    def get_pos(self):
        return self.pos
    
class ConditionalizedDecoderCell(object):
    def __init__(self, decoder_chain, src_encoding, mask_input):
        self.decoder_chain = decoder_chain
        self.src_encoding = src_encoding
        src_mb_size, src_max_length, src_d_model = src_encoding.data.shape
        self.src_mb_size = src_mb_size
        self.mask_input = mask_input
        
    def get_initial_logits(self, mb_size = None):
        if mb_size is None:
            mb_size = self.src_mb_size
        else:
            assert self.src_mb_size == 1
        assert mb_size is not None
    
        bos_encoding = F.broadcast_to(self.decoder_chain.bos_encoding, (mb_size, 1, self.decoder_chain.d_model))
        
        cross_mask = self.decoder_chain.xp.broadcast_to(self.mask_input[:,0:1,0:1,:], (self.mask_input.shape[0], self.decoder_chain.n_heads, 1, self.mask_input.shape[3]))
        
        final_layer, prev_states =  self.decoder_chain.encoding_layers.one_step(bos_encoding, None,
                                                               self.src_encoding, cross_mask)
        
        logits = self.decoder_chain.logits_layer(F.reshape(final_layer, (mb_size, self.decoder_chain.d_model)))
        return logits, DecoderState(pos=-1, prev_states=prev_states)
    
    def __call__(self, prev_decoder_state, inpt):
        current_mb_size = inpt.shape[0]
#         mask = np.zeros((current_mb_size, ), dtype = np.float32)
#         padded = np.zeros((current_mb_size, ), dtype = np.float32)
#         for num_batch, idx in enumerate(inpt):
#             padded[num_batch] = idx if idx is not None else 0
#             mask[num_batch] = 0 if idx is not None else -10000
        
        prev_decoder_state = prev_decoder_state.reduce_to_minibatch_size(current_mb_size)
        current_pos = prev_decoder_state.get_pos() + 1
        
        encoded = self.decoder_chain.emb(inpt)
        pos_vect = self.decoder_chain.get_one_pos_vect(current_mb_size, current_pos)
        
        encoded = encoded + pos_vect
        
        if self.decoder_chain.dropout is not None:
            encoded = F.dropout(encoded, self.decoder_chain.dropout)
            
        cross_mask = self.decoder_chain.xp.broadcast_to(
            self.mask_input[:,0:1,0:1,:], 
            (self.mask_input.shape[0], self.decoder_chain.n_heads, 1, self.mask_input.shape[3]))
       
        final_layer, prev_states =  self.decoder_chain.encoding_layers.one_step(encoded, prev_decoder_state.get_states(),
                                                               self.src_encoding, cross_mask)
       
#         logits = apply_linear_layer_to_last_dims(final_layer, self.decoder_chain.logits_layer)
        logits = self.decoder_chain.logits_layer(F.reshape(final_layer, (current_mb_size, self.decoder_chain.d_model)))
        return logits, DecoderState(pos=current_pos, prev_states=prev_states)
    
class Decoder(Chain):
    def __init__(self, V, d_model=512, n_heads=8, d_ff=2048, experimental_relu=False, dropout=None, nb_layers=6,
                 residual_mode="normal", no_normalize=False):
        super(Decoder, self).__init__(
            emb = L.EmbedID(V, d_model),
            encoding_layers = DecoderMultiLayer(d_model, n_heads, d_ff=d_ff,
                                                experimental_relu=experimental_relu, 
                                                dropout=dropout, nb_layers=nb_layers,
                                                residual_mode=residual_mode, no_normalize=no_normalize),
            logits_layer = L.Linear(d_model, V + 1)
        )
        
        self.dropout = dropout
        self.n_heads = n_heads
        self.d_model = d_model
        self.cached_pos_vect = None
        
        self.add_param("bos_encoding", (1, 1, d_model))
        self.bos_encoding.data[...] = np.random.randn(d_model)
        
        self.V = V
        self.eos_idx = V
        
    def get_device(self):
        if self.xp is np:
            return None
        else:
            return self.emb.W.data.device
        
    def move_np_array_to_correct_device(self, np_array):
        device = self.get_device()
        if device is None:
            return np_array
        else:
            return chainer.cuda.to_gpu(np_array, device=device)
        
    def get_conditionalized_cell(self, encoded_input, mask_input):
        return ConditionalizedDecoderCell(self, encoded_input, mask_input)
        
        
    def get_cached_pos_vect(self, length):    
        if self.cached_pos_vect is None or self.cached_pos_vect.shape[0] < length:
            self.cached_pos_vect = generate_pos_vectors(self.d_model, length)
            self.cached_pos_vect = self.move_np_array_to_correct_device(self.cached_pos_vect)
        return self.cached_pos_vect
    
    def get_pos_vect(self, mb_size, length):
        cached_pos_vect = self.get_cached_pos_vect(length)
#         print self.cached_pos_vect[None, :length, :].shape, mb_size, length, self.d_model
        return self.xp.broadcast_to(cached_pos_vect[None, :length, :], (mb_size, length, self.d_model))
        
    def get_one_pos_vect(self, mb_size, pos):
        cached_pos_vect = self.get_cached_pos_vect(pos+1)
        return self.xp.broadcast_to(cached_pos_vect[None, pos:pos+1, :], (mb_size, 1, self.d_model))
    
    def make_batch(self, seq_list):
        padded_data = pad_data(seq_list, pad_value=0)
        seq_length = [len(x) + 1 for x in seq_list] #BOS
        max_length_1 = max(seq_length)
        max_length_2 = max_length_1
        mb_size = len(seq_list)
        mask = make_batch_mask(mb_size, self.n_heads, max_length_1, max_length_2, 
#                     key_seq_lengths=seq_length, #actually not needed
                    future_mask=True,
                    mask_value=-10000)
        
        padded_data = self.move_np_array_to_correct_device(padded_data)
        mask = self.move_np_array_to_correct_device(mask)
                
        return padded_data, mask
    
    def compute_logits(self, seq_list, encoded_input, mask_input):
        mb_size = len(seq_list)
        max_length_1 = max(len(x) for x in seq_list)
        x, mask = self.make_batch(seq_list)
        
#         print "padded_data", x
#         print "mask", mask
        
        assert self.xp.all(mask_input == self.xp.broadcast_to(mask_input[:,0:1,0:1,:], mask_input.shape))
        
        encoded = self.emb(x)
        encoded += self.get_pos_vect(mb_size, max_length_1)
        
        if self.dropout is not None:
            encoded = F.dropout(encoded, self.dropout)
        
        bos_plus_encoded = F.concat((F.broadcast_to(self.bos_encoding, (mb_size, 1, self.d_model)), encoded), axis=1)
        
        cross_mask = self.xp.broadcast_to(mask_input[:,0:1,0:1,:], (mask_input.shape[0], self.n_heads, bos_plus_encoded.data.shape[1], mask_input.shape[3]))
        
        final_layer =  self.encoding_layers(bos_plus_encoded, encoded_input, mask, cross_mask)
        logits = apply_linear_layer_to_last_dims(final_layer, self.logits_layer)
        return logits
    
    def compute_loss(self, seq_list, encoded_input, mask_input, reduce="mean"):
        logits = self.compute_logits(seq_list, encoded_input, mask_input)
        padded_target_with_eos = pad_data(seq_list, pad_value=-1, add_eos=self.eos_idx)
        padded_target_with_eos = self.move_np_array_to_correct_device(padded_target_with_eos)
        loss = F.softmax_cross_entropy(F.reshape(logits, (-1, self.V+1)), padded_target_with_eos.reshape(-1,), reduce=reduce)
        return loss
    
    