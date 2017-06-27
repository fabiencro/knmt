import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Chain, ChainList

########################################################################
# batch handling
#

def make_batch_mask(mb_size, n_head, max_length_1, max_length_2, 
                    key_seq_lengths=None,
                    future_mask=False,
                    mask_value=-10000):
    
    if future_mask:
        assert max_length_1 == max_length_2
        mask = np.array(
                np.broadcast_to(( (-mask_value) * (np.tri(max_length_1, dtype = np.float32)-1))[None,None,:,:], 
                                (mb_size, n_head, max_length_1, max_length_2))
                )
    else:
        mask = np.zeros((mb_size, n_head, max_length_1, max_length_2), dtype = np.float32)
        
    if key_seq_lengths is not None:
        assert mb_size == len(key_seq_lengths)
        assert min(key_seq_lengths) > 0
        assert max(key_seq_lengths) <= max_length_2
        for num_batch, length in enumerate(key_seq_lengths):
            mask[num_batch, :, :, length:] = mask_value

    return mask

def pad_data(data, pad_value=0, add_eos=None):
    mb_size = len(data)
    padded_length = max(len(x) for x in data)
    if add_eos is not None:
        padded_length += 1
    padded_array = np.zeros((mb_size, padded_length), dtype=np.int32)
    if pad_value != 0:
        padded_array += pad_value
    for num_batch, seq in enumerate(data):
        padded_array[num_batch, :len(seq)] = seq
        if add_eos is not None:
            padded_array[num_batch, len(seq)] = add_eos
    return padded_array

########################################################################
# helper functions
#

def reorganize_by_head(Q, n_heads):
    mb_size, n_Q, d_model = Q.data.shape
    assert d_model%n_heads == 0
    head_size = d_model / n_heads
    reshaped_Q = F.reshape(Q, (mb_size, n_Q, n_heads, head_size))
    return F.swapaxes(reshaped_Q, 1, 2)

def undo_reorganize_by_head(Q):
    mb_size, n_heads, n_Q, head_size = Q.data.shape
    swapped_Q = F.swapaxes(Q, 1, 2)
    return F.reshape(swapped_Q, (mb_size, n_Q, -1))

def test_reorganize_by_head():
    Q = Variable(np.arange(2*3*5*7).reshape(5, 7, 2*3).astype(np.float32))
    Qr = reorganize_by_head(Q, 2)
    Qrr = undo_reorganize_by_head(Qr)
    
    assert np.all(Qrr.data == Q.data)
    assert Qr.data.base is Q.data
    assert Qrr.data.base is Q.data
    assert np.all(Qr.data[:, 0, :, :]%6 < 3)
    assert np.all(Qr.data[:, 1, :, :]%6 >= 3)

def batch_matmul_last_dims(A, B, transa=False, transb=False):
    assert A.data.shape[:-2] == B.data.shape[:-2]
    reshaped_A = F.reshape(A, (-1,) + A.data.shape[-2:])
    reshaped_B = F.reshape(B, (-1,) + B.data.shape[-2:])
    reshaped_result = F.batch_matmul(reshaped_A, reshaped_B, transa=transa, transb=transb)
    result = F.reshape(reshaped_result, A.data.shape[:-2] + reshaped_result.data.shape[-2:])
    return result

def apply_linear_layer_to_last_dims(Q, w_Q):
    mb_size_Q, n_Q, d_model_Q = Q.data.shape
    return F.reshape(w_Q(F.reshape(Q, (mb_size_Q * n_Q, d_model_Q))), (mb_size_Q, n_Q, -1))

########################################################################
# Multihead Attention
#

class ConstantSizeMultiBatchMultiHeadAttention(Chain):
    """
        Assume all layers have same size d_model
    """
    def __init__(self, d_model = 512, n_heads = 8, experimental_relu=False, dropout=None):
        if d_model%n_heads != 0:
            raise ValueError("d_model(%i) should be divisible by n_head(%i)"%(d_model, n_heads))
        
        super(ConstantSizeMultiBatchMultiHeadAttention, self).__init__(
            w_Q = L.Linear(d_model, d_model, nobias=False),
            w_K = L.Linear(d_model, d_model, nobias=True),
            w_V = L.Linear(d_model, d_model, nobias=False),
            )
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model / n_heads
        
        scaling_factor = 1.0 / self.xp.sqrt(self.xp.array([[[[self.head_size]]]], dtype=self.xp.float32))
        self.add_persistent("scaling_factor", scaling_factor) #added as persistent so that it works with to_gpu/to_cpu
        
        self.experimental_relu = experimental_relu
        
        self.dropout = dropout
                                                 
                                                 
    def __call__(self, Q, K, V, batch_mask = None, train=True):
        mb_size_Q, n_Q, d_model_Q = Q.data.shape
        mb_size_K, seq_length_K, d_model_K = K.data.shape
        mb_size_V, seq_length_V, d_model_V = V.data.shape
                                            
        assert mb_size_Q == mb_size_K == mb_size_V
        assert d_model_Q == d_model_K == d_model_V == self.d_model
        assert seq_length_K == seq_length_V                
                                            
        mb_size = mb_size_Q
                                            
        if batch_mask is not None:                            
            mb_size_batch_mask, mask_n_heads, mask_n_Q, mask_seq_length_K = batch_mask.shape
            assert  mb_size_batch_mask == mb_size
            assert mask_n_heads == self.n_heads
            assert mask_n_Q == n_Q, "%i != %i"%(mask_n_Q, n_Q)
            assert mask_seq_length_K == mask_seq_length_K               
                                            
        
        proj_Q = apply_linear_layer_to_last_dims(Q, self.w_Q)
        proj_K = apply_linear_layer_to_last_dims(K, self.w_K)
        proj_V = apply_linear_layer_to_last_dims(V, self.w_V)
        
        reorganized_Q = reorganize_by_head(proj_Q, self.n_heads)
        reorganized_K = reorganize_by_head(proj_K, self.n_heads)
        
        scalar_product = batch_matmul_last_dims(reorganized_Q, reorganized_K, transb=True)
                                            
        scaling_factor = self.xp.broadcast_to(self.scaling_factor, (mb_size, self.n_heads, n_Q, seq_length_K))
        scaled_scalar_product = scalar_product * scaling_factor
        
        if batch_mask is not None:
            scaled_scalar_product = scaled_scalar_product + batch_mask
        
        if self.experimental_relu:
            addressing_weights = F.relu(scaled_scalar_product)
        else:
            addressing_weights = F.softmax(scaled_scalar_product)
        
        if self.dropout is not None:
            addressing_weights = F.dropout(addressing_weights, ratio=self.dropout, train=train)
        
        reorganized_V = reorganize_by_head(proj_V, self.n_heads)
        reorganized_result = batch_matmul_last_dims(addressing_weights, reorganized_V)
        return undo_reorganize_by_head(reorganized_result)
    
class AddAndNormalizedAttentionBase(Chain):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None):
        super(AddAndNormalizedAttentionBase, self).__init__(
            multi_attention= ConstantSizeMultiBatchMultiHeadAttention(d_model = d_model, n_heads=n_heads,
                                                             experimental_relu=experimental_relu,
                                                                     dropout=dropout),
            normalizing_layer = L.LayerNormalization()
        )
        
        self.dropout = dropout
        self.d_model = d_model
        
    def dropout_and_add_and_normalize(self, sub_output, inpt, train=True):
        if self.dropout is not None:
            sub_output = F.dropout(sub_output, ratio=self.dropout, train=train)
        added_output = sub_output + inpt
        
        mb, length, d_model = added_output.shape
        return F.reshape(
                self.normalizing_layer(
                    F.reshape(added_output, (mb * length, d_model))
                    ), (mb, length, d_model))
      
    def extract_last(self, x):
        mb_size, nQ, dm = x.data.shape
        if nQ == 1:
            return x
        _, x_last = F.split_axis(x, (nQ-1,), axis=1, force_tuple=True)
        assert x_last.data.shape == (mb_size, 1, dm)
        return x_last
        
        
    
class AddAndNormalizedSelfAttentionLayer(AddAndNormalizedAttentionBase):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None):
        super(AddAndNormalizedSelfAttentionLayer, self).__init__(
            d_model, n_heads, experimental_relu=experimental_relu, dropout=dropout
        )
        
    def __call__(self, x, mask, train=True, only_last=False):
        if only_last:
            x_in = self.extract_last(x)
        else:
            x_in = x
        sub_output = self.multi_attention(x_in, x, x, mask, train=train)
            
        return self.dropout_and_add_and_normalize(sub_output, x_in)
        
    
class AddAndNormalizedCrossAttentionLayer(AddAndNormalizedAttentionBase):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None):
        super(AddAndNormalizedCrossAttentionLayer, self).__init__(
            d_model, n_heads, experimental_relu=experimental_relu, dropout=dropout
        )
        
    def __call__(self, tgt_x, src_x, mask, train=True, only_last=False):
        if only_last:
            x_in = self.extract_last(tgt_x)
        else:
            x_in = tgt_x
        sub_output = self.multi_attention(x_in, src_x, src_x, mask, train=train)
            
        return self.dropout_and_add_and_normalize(sub_output, x_in)
        
