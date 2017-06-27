import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Chain, ChainList

from multi_attention import make_batch_mask, pad_data, AddAndNormalizedSelfAttentionLayer, AddAndNormalizedCrossAttentionLayer, apply_linear_layer_to_last_dims
    
def generate_pos_vectors(d_model, max_length):
    pos_component = np.arange(max_length, dtype = np.float32)
    dim_component = np.arange(d_model, dtype = np.float32)
    dim_component_even = np.floor_divide(dim_component, 2) * 2
    dim_factor = np.power(1e-4, dim_component_even / d_model)
    pos_dim = pos_component[:, None] * dim_factor[None, :]
    pos_dim[:, ::2] = np.sin(pos_dim[:, ::2])
    pos_dim[:, 1::2] = np.cos(pos_dim[:, 1::2])
    return pos_dim
    

class FeedForward(Chain):
    def __init__(self, d_model=512, d_ff=2048, dropout=None):
        super(FeedForward, self).__init__(
            lin1 = L.Linear(d_model, d_ff),
            lin2 = L.Linear(d_ff, d_model),
            normalization_layer = L.LayerNormalization()
        )
        
        self.dropout = dropout
        
    def __call__(self, x_input, train=True):
        if len(x_input.data.shape) > 2:
            x = F.reshape(x_input, (-1, x_input.shape[-1]))
        else:
            x = x_input
            
        ff_output = self.lin2(F.relu(self.lin1(x)))
        
        if self.dropout is not None:
            ff_output = F.dropout(ff_output, self.dropout, train=train)
            
        norm_ff_output = self.normalization_layer(ff_output + x)
        
        if len(x_input.data.shape) > 2:
            norm_ff_output = F.reshape(norm_ff_output, x_input.data.shape)
            
        return norm_ff_output

class EncoderLayer(Chain):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None):
        super(EncoderLayer, self).__init__(
            ff_layer = FeedForward(d_model, n_heads, dropout=dropout),
            self_attention_layer = AddAndNormalizedSelfAttentionLayer(d_model=d_model, n_heads=n_heads,
                                                             experimental_relu=experimental_relu,
                                                          dropout=dropout)
        )
        
        
    def __call__(self, x, mask, train=True):
        y1 = self.self_attention_layer(x, mask, train)
        y2 = self.ff_layer(y1, train=train)
        return y2
    
    
class EncoderMultiLayer(ChainList):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None, nb_layers=6):
        super(EncoderMultiLayer, self).__init__()
        for _ in range(nb_layers):
            self.add_link(EncoderLayer(d_model, n_heads, experimental_relu=experimental_relu, dropout=dropout))
        
    def __call__(self, x, mask, train=True):
        for link in self:
            x = link(x, mask, train=train)
        return x
        
        
class Encoder(Chain):
    def __init__(self, V, d_model=512, n_heads=8, experimental_relu=False, dropout=None, nb_layers=6):
        super(Encoder, self).__init__(
            emb = L.EmbedID(V, d_model),
            encoding_layers = EncoderMultiLayer(d_model, n_heads, 
                                                experimental_relu=experimental_relu, 
                                                dropout=dropout, nb_layers=nb_layers)
        )
        
        self.dropout = dropout
        self.n_heads = n_heads
        self.d_model = d_model
        self.cached_pos_vect = None
        
    def get_pos_vect(self, mb_size, length):
        if self.cached_pos_vect is None or self.cached_pos_vect.shape[0] < length:
            self.cached_pos_vect = generate_pos_vectors(self.d_model, length)
            if self.xp != np:
                self.cached_pos_vect = chainer.cuda.to_gpu(self.cached_pos_vect, device=self.device)
#         print self.cached_pos_vect[None, :length, :].shape, mb_size, length, self.d_model
        return self.xp.broadcast_to(self.cached_pos_vect[None, :length, :], (mb_size, length, self.d_model))
        
    
    def make_batch(self, seq_list):
        padded_data = pad_data(seq_list, pad_value=0)
        seq_length = [len(x) for x in seq_list]
        max_length_1 = max(seq_length)
        max_length_2 = max_length_1
        mb_size = len(seq_list)
        mask = make_batch_mask(mb_size, self.n_heads, max_length_1, max_length_2, 
                    key_seq_lengths=seq_length,
                    future_mask=False,
                    mask_value=-10000)
        return padded_data, mask
    
    def __call__(self, seq_list, train=True):
        mb_size = len(seq_list)
        max_length_1 = max(len(x) for x in seq_list)
        x, mask = self.make_batch(seq_list)
    
        
        encoded = self.emb(x)
        encoded += self.get_pos_vect(mb_size, max_length_1)
        
        if self.dropout is not None:
            encoded = F.dropout(encoded, self.dropout, train=train)
        
        return self.encoding_layers(encoded, mask, train=train), mask
        
    
class DecoderLayer(Chain):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None):
        super(DecoderLayer, self).__init__(
            ff_layer = FeedForward(d_model, n_heads, dropout=dropout),
            self_attention_layer = AddAndNormalizedSelfAttentionLayer(d_model=d_model, n_heads=n_heads,
                                                             experimental_relu=experimental_relu,
                                                          dropout=dropout),
            
            cross_attention_layer = AddAndNormalizedCrossAttentionLayer(d_model=d_model, n_heads=n_heads,
                                                             experimental_relu=experimental_relu,
                                                          dropout=dropout)
        )
        
        self.n_heads = n_heads
        self.d_model = d_model
    
    def __call__(self, tgt, src, mask, mask_input, train=True):
        y1 = self.self_attention_layer(tgt, mask, train=train)
        y2 = self.cross_attention_layer(y1, src, mask_input, train=train)
        y3 = self.ff_layer(y2, train=train)
        return y3
    
    def one_step(self, new_inpt, prev_states, src, mask_input, train=True):
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
            
        y1_last = self.self_attention_layer(full_tgt, mask = None, train=train, only_last=True)
        
        if prev_states is not None:
            full_y1 = F.concat((prev_cross_attn, y1_last) , axis=1)
        else:
            full_y1 = y1_last
            
        y2_last = self.cross_attention_layer(full_y1, src, mask_input, train=train, only_last=True)
        y3_last = self.ff_layer(y2_last, train=train)
        return y3_last, (full_tgt, full_y1)
    
class DecoderMultiLayer(ChainList):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None, nb_layers=6):
        super(DecoderMultiLayer, self).__init__()
        for _ in range(nb_layers):
            self.add_link(DecoderLayer(d_model, n_heads, experimental_relu=experimental_relu, dropout=dropout))
        
    def __call__(self, tgt, src, mask, mask_input, train=True):
        for link in self:
            tgt = link(tgt, src, mask, mask_input, train=train)
        return tgt
    
    def one_step(self, new_inpt, prev_states, src, mask_input, train=True):
        assert prev_states is None or len(prev_states) == len(self)
        new_prev_tgt = []
        for num_link, link in enumerate(self):
            tgt_last, this_prev_tgt = link.one_step(new_inpt, prev_states[num_link] if prev_states is not None else None, src, mask_input, train=train)
            new_prev_tgt.append(this_prev_tgt)
        return tgt_last, tuple(new_prev_tgt)      
    
def cut_minibatch(minibatch, new_mb_size):
    current_mb_size = minibatch.data.shape[0]
    assert new_mb_size <= current_mb_size
    if current_mb_size == new_mb_size:
        return minibatch
    new_minibatch, _ = F.split_axis(minibatch, (new_mb_size,), axis=0, force_tuple=True)
    return new_minibatch
    
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
        
    def get_initial_logits(self, mb_size = None, train=True):
        if mb_size is None:
            mb_size = self.src_mb_size
        else:
            assert self.src_mb_size == 1
        assert mb_size is not None
    
        bos_encoding = F.broadcast_to(self.decoder_chain.bos_encoding, (mb_size, 1, self.decoder_chain.d_model))
        
        cross_mask = F.broadcast_to(self.mask_input[:,1:2,1:2,:], (self.mask_input.shape[0], self.decoder_chain.n_heads, 1, self.mask_input.shape[3]))
        
        final_layer, prev_states =  self.decoder_chain.encoding_layers.one_step(bos_encoding, None,
                                                               self.src_encoding, cross_mask, train=train)
        
        logits = self.decoder_chain.logits_layer(F.reshape(final_layer, (mb_size, self.decoder_chain.d_model)))
        return logits, DecoderState(pos=-1, prev_states=prev_states)
    
    def __call__(self, prev_decoder_state, inpt, train=True):
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
            encoded = F.dropout(encoded, self.decoder_chain.dropout, train=train)
            
        cross_mask = F.broadcast_to(
            self.mask_input[:,0:1,0:1,:], 
            (self.mask_input.shape[0], self.decoder_chain.n_heads, 1, self.mask_input.shape[3]))
       
        final_layer, prev_states =  self.decoder_chain.encoding_layers.one_step(encoded, prev_decoder_state.get_states(),
                                                               self.src_encoding, cross_mask, train=train)
       
#         logits = apply_linear_layer_to_last_dims(final_layer, self.decoder_chain.logits_layer)
        logits = self.decoder_chain.logits_layer(F.reshape(final_layer, (current_mb_size, self.decoder_chain.d_model)))
        return logits, DecoderState(pos=current_pos, prev_states=prev_states)
    
class Decoder(Chain):
    def __init__(self, V, d_model=512, n_heads=8, experimental_relu=False, dropout=None, nb_layers=6):
        super(Decoder, self).__init__(
            emb = L.EmbedID(V, d_model),
            encoding_layers = DecoderMultiLayer(d_model, n_heads, 
                                                experimental_relu=experimental_relu, 
                                                dropout=dropout, nb_layers=nb_layers),
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
        
    def get_conditionalized_cell(self, encoded_input, mask_input):
        return ConditionalizedDecoderCell(self, encoded_input, mask_input)
        
        
    def get_cached_pos_vect(self, length):    
        if self.cached_pos_vect is None or self.cached_pos_vect.shape[0] < length:
            self.cached_pos_vect = generate_pos_vectors(self.d_model, length)
            if self.xp != np:
                self.cached_pos_vect = chainer.cuda.to_gpu(self.cached_pos_vect, device=self.device)
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
                    key_seq_lengths=seq_length,
                    future_mask=True,
                    mask_value=-10000)
        return padded_data, mask
    
    def compute_loss(self, seq_list, encoded_input, mask_input, train=True):
        mb_size = len(seq_list)
        max_length_1 = max(len(x) for x in seq_list)
        x, mask = self.make_batch(seq_list)
        
        assert self.xp.all(mask_input == self.xp.broadcast_to(mask_input[:,0:1,0:1,:], mask_input.shape))
        
        encoded = self.emb(x)
        encoded += self.get_pos_vect(mb_size, max_length_1)
        
        if self.dropout is not None:
            encoded = F.dropout(encoded, self.dropout, train=train)
        
        bos_plus_encoded = F.concat((F.broadcast_to(self.bos_encoding, (mb_size, 1, self.d_model)), encoded), axis=1)
        
        cross_mask = F.broadcast_to(mask_input[:,0:1,0:1,:], (mask_input.shape[0], self.n_heads, bos_plus_encoded.data.shape[1], mask_input.shape[3]))
        
        final_layer =  self.encoding_layers(bos_plus_encoded, encoded_input, mask, cross_mask, train=train)
        logits = apply_linear_layer_to_last_dims(final_layer, self.logits_layer)
        padded_target_with_eos = pad_data(seq_list, pad_value=-1, add_eos=self.eos_idx)
        
        loss = F.softmax_cross_entropy(F.reshape(logits, (-1, self.V+1)), padded_target_with_eos.reshape(-1,))
        
        return loss
    
    
class EncoderDecoder(Chain):
    def __init__(self, V_src, V_tgt, d_model=512, n_heads=8, experimental_relu=False, dropout=None, 
                 nb_layers_src=6, nb_layers_tgt=6):
        super(EncoderDecoder, self).__init__(
            encoder = Encoder(V_src, d_model=d_model, n_heads=n_heads, 
                              experimental_relu=experimental_relu, dropout=dropout, nb_layers=nb_layers_src),
            decoder = Decoder(V_tgt, d_model=d_model, n_heads=n_heads, 
                              experimental_relu=experimental_relu, dropout=dropout, nb_layers=nb_layers_tgt),
        )
        
        self.V_tgt = V_tgt
        
    def compute_loss(self, src_seq, tgt_seq, train=True):
        encoded_source, src_mask = self.encoder(src_seq, train=train)
        loss = self.decoder.compute_loss(tgt_seq, encoded_source, src_mask, train=train)
        return loss
        
    def greedy_translate(self, src_seq, nb_steps):
        encoded_source, src_mask = self.encoder(src_seq, train=False)
        decoding_cell = self.decoder.get_conditionalized_cell(encoded_source, src_mask)
        
        logits, decoder_state = decoding_cell.get_initial_logits(train=False)
        prev_word = None
        
        mb_size = len(src_seq)
        result = [[] for _ in xrange(mb_size)]
        
        num_step = 0
        while 1:
#             print "logits shape", logits.shape
            prev_word = self.xp.argmax(logits.data, axis = 1).reshape(-1, 1).astype(self.xp.int32)
#             print "prev w shape", prev_word.shape
            assert prev_word.shape == (mb_size, 1)
            for i in xrange(mb_size):
                result[i].append(prev_word[i, 0])
            
            prev_word = self.xp.where(prev_word == self.decoder.eos_idx, 0, prev_word)
            num_step += 1
            if num_step > nb_steps:
                break
            logits, decoder_state = decoding_cell(decoder_state, prev_word, train=False)
        return result
            
            
    
        