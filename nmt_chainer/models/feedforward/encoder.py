import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Chain, ChainList

from nmt_chainer.models.feedforward.utils import generate_pos_vectors, make_batch_mask, pad_data, FeedForward
from nmt_chainer.models.feedforward.multi_attention import AddAndNormalizedSelfAttentionLayer

class EncoderLayer(Chain):
    def __init__(self, d_model, n_heads, d_ff=2048, experimental_relu=False, dropout=None, residual_mode="normal", no_normalize=False):
        super(EncoderLayer, self).__init__(
            ff_layer = FeedForward(d_model, d_ff=d_ff, dropout=dropout, residual_mode=residual_mode, no_normalize=no_normalize),
            self_attention_layer = AddAndNormalizedSelfAttentionLayer(d_model=d_model, n_heads=n_heads,
                                                             experimental_relu=experimental_relu,
                                                          dropout=dropout, residual_mode=residual_mode, no_normalize=no_normalize)
        )
        
        
    def __call__(self, x, mask):
        y1 = self.self_attention_layer(x, mask)
        y2 = self.ff_layer(y1)
        return y2
   
class EncoderMultiLayer(ChainList):
    def __init__(self, d_model, n_heads, d_ff=2048, experimental_relu=False, dropout=None, nb_layers=6,
                 residual_mode="normal", no_normalize=False):
        super(EncoderMultiLayer, self).__init__()
        for _ in range(nb_layers):
            self.add_link(EncoderLayer(d_model, n_heads, d_ff=d_ff, experimental_relu=experimental_relu, dropout=dropout,
                                residual_mode=residual_mode, no_normalize=no_normalize))
        
    def __call__(self, x, mask):
        for link in self:
            x = link(x, mask)
        return x
        
        
class Encoder(Chain):
    def __init__(self, V, d_model=512, n_heads=8, d_ff=2048, experimental_relu=False, dropout=None, nb_layers=6,
                 residual_mode="normal", no_normalize=False):
        super(Encoder, self).__init__(
            emb = L.EmbedID(V, d_model),
            encoding_layers = EncoderMultiLayer(d_model, n_heads, d_ff=d_ff,
                                                experimental_relu=experimental_relu, 
                                                dropout=dropout, nb_layers=nb_layers,
                                                residual_mode=residual_mode, no_normalize=no_normalize)
        )
        
        self.dropout = dropout
        self.n_heads = n_heads
        self.d_model = d_model
        self.cached_pos_vect = None
        
    def get_pos_vect(self, mb_size, length):
        if self.cached_pos_vect is None or self.cached_pos_vect.shape[0] < length:
            self.cached_pos_vect = generate_pos_vectors(self.d_model, length)
            device = self.get_device()
            if device is not None:
                self.cached_pos_vect = chainer.cuda.to_gpu(self.cached_pos_vect, device=device)
#         print self.cached_pos_vect[None, :length, :].shape, mb_size, length, self.d_model
        return self.xp.broadcast_to(self.cached_pos_vect[None, :length, :], (mb_size, length, self.d_model))
        
    def get_device(self):
        if self.xp is np:
            return None
        else:
            return self.emb.W.data.device
    
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
        
        device = self.get_device()
        if device is not None:
            with device:
                padded_data = self.xp.array(padded_data)
                mask = self.xp.array(mask)
            
        return padded_data, mask
    
    def __call__(self, seq_list):
        mb_size = len(seq_list)
        max_length_1 = max(len(x) for x in seq_list)
        x, mask = self.make_batch(seq_list)
    
        encoded = self.emb(x)
        encoded += self.get_pos_vect(mb_size, max_length_1)
        
        if self.dropout is not None:
            encoded = F.dropout(encoded, self.dropout)
        
        return self.encoding_layers(encoded, mask), mask