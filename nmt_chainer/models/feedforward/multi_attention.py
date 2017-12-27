import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Chain, ChainList

from nmt_chainer.models.feedforward.utils import apply_linear_layer_to_last_dims, DropoutAndAddAndNormalize

#from nmt_chainer.additional_links.layer_normalization import LayerNormalizationLink as LayerNormalization

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

########################################################################
# Multihead Attention
#

disable_cudnn_softmax=False

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
        
        if n_heads >= 2:
            self.add_link("w_O", L.Linear(d_model, d_model)) #if n_heads == 1, it is redundant with w_V
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model / n_heads
        
        scaling_factor = 1.0 / self.xp.sqrt(self.xp.array([[[[self.head_size]]]], dtype=self.xp.float32))
        self.add_persistent("scaling_factor", scaling_factor) #added as persistent so that it works with to_gpu/to_cpu
        
        self.experimental_relu = experimental_relu
        
        self.dropout = dropout
                                                 
                                                 
    def __call__(self, Q, K, V, batch_mask = None):
#         print "Q",
#         print Q.data
#         print "K",
#         print K.data
#         print "V",
#         print V.data
#         print "M", batch_mask
        
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
#         print "S", scalar_product.data                       
        scaling_factor = self.xp.broadcast_to(self.scaling_factor, (mb_size, self.n_heads, n_Q, seq_length_K))
        scaled_scalar_product = scalar_product * scaling_factor
        
        if batch_mask is not None:
            scaled_scalar_product = scaled_scalar_product + batch_mask
#         print "B", scaled_scalar_product.data  
        if self.experimental_relu:
            addressing_weights = F.relu(scaled_scalar_product)
        else:
            addressing_weights = F.reshape(F.softmax(F.reshape(scaled_scalar_product, (mb_size * n_Q * self.n_heads, seq_length_K)),
                                                     #use_cudnn=disable_cudnn_softmax
                                                     ),
                                           (mb_size, self.n_heads, n_Q, seq_length_K) )
        
        if self.dropout is not None:
            addressing_weights = F.dropout(addressing_weights, ratio=self.dropout)
        
#         print "A", addressing_weights.data
        reorganized_V = reorganize_by_head(proj_V, self.n_heads)
        reorganized_result = batch_matmul_last_dims(addressing_weights, reorganized_V)
        result = undo_reorganize_by_head(reorganized_result)
        
        if self.n_heads >= 2:
            result = apply_linear_layer_to_last_dims(result, self.w_O)
        
#         print "R", result.data
        return result
    
    
class AddAndNormalizedAttentionBase(Chain):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None, residual_mode="normal", no_normalize=False):
        super(AddAndNormalizedAttentionBase, self).__init__(
            multi_attention= ConstantSizeMultiBatchMultiHeadAttention(d_model = d_model, n_heads=n_heads,
                                                             experimental_relu=experimental_relu,
                                                                     dropout=dropout),
                                                            
            residual_layer = DropoutAndAddAndNormalize(dropout=dropout, residual_mode=residual_mode, no_normalize=no_normalize)
        )
        
        self.d_model = d_model
        
#         self.dropout = dropout
#         
#         if not no_normalize:
#             self.add_link("normalizing_layer", LayerNormalization())
#         
#         self.no_add = no_add
#         self.no_normalize = no_normalize
        
#     def dropout_and_add_and_normalize(self, sub_output, inpt, train=True):
#         if self.dropout is not None:
#             sub_output = F.dropout(sub_output, ratio=self.dropout, train=train)
#             
#         if self.no_add:
#             added_output = sub_output
#         else:
#             added_output = sub_output + inpt
#         
#         if self.no_normalize:
#             final_layer = added_output
#         else:
#             mb, length, d_model = added_output.shape
#             final_layer = F.reshape(
#                 self.normalizing_layer(
#                     F.reshape(added_output, (mb * length, d_model))
#                     ), (mb, length, d_model))
#         
#         return final_layer
      
    def extract_last(self, x):
        mb_size, nQ, dm = x.data.shape
        if nQ == 1:
            return x
        _, x_last = F.split_axis(x, (nQ-1,), axis=1, force_tuple=True)
        assert x_last.data.shape == (mb_size, 1, dm)
        return x_last
        
        
    
class AddAndNormalizedSelfAttentionLayer(AddAndNormalizedAttentionBase):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None, residual_mode="normal", no_normalize=False):
        super(AddAndNormalizedSelfAttentionLayer, self).__init__(
            d_model, n_heads, experimental_relu=experimental_relu, dropout=dropout,
            residual_mode=residual_mode, no_normalize=no_normalize
        )
        
    def __call__(self, x, mask, only_last=False):
#         print "SELF"
        if only_last:
            x_in = self.extract_last(x)
        else:
            x_in = x
        sub_output = self.multi_attention(x_in, x, x, mask)
            
        return self.residual_layer(sub_output, x_in)
        
    
class AddAndNormalizedCrossAttentionLayer(AddAndNormalizedAttentionBase):
    def __init__(self, d_model, n_heads, experimental_relu=False, dropout=None, residual_mode="normal", no_normalize=False):
        super(AddAndNormalizedCrossAttentionLayer, self).__init__(
            d_model, n_heads, experimental_relu=experimental_relu, dropout=dropout,
            residual_mode=residual_mode, no_normalize=no_normalize
        )
        
    def __call__(self, tgt_x, src_x, mask, only_last=False):
#         print "CROSS"
        if only_last:
            x_in = self.extract_last(tgt_x)
        else:
            x_in = tgt_x
        sub_output = self.multi_attention(x_in, src_x, src_x, mask)
            
        return self.residual_layer(sub_output, x_in)
        
