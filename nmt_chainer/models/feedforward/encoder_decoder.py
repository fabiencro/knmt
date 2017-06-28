import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Chain, ChainList

from nmt_chainer.models.feedforward.encoder import Encoder
from nmt_chainer.models.feedforward.decoder import Decoder
    
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
            
    def compute_logits(self, src_seq, tgt_seq, train=True):
        encoded_source, src_mask = self.encoder(src_seq, train=train)
        logits = self.decoder.compute_logits(tgt_seq, encoded_source, src_mask, train=train)
        return logits
      
    def compute_logits_step_by_step(self, src_seq, tgt_seq, train=True):
        encoded_source, src_mask = self.encoder(src_seq, train=train)
        decoding_cell = self.decoder.get_conditionalized_cell(encoded_source, src_mask)
        
        logits, decoder_state = decoding_cell.get_initial_logits(train=train)            
        
        from nmt_chainer.models.feedforward.utils import pad_data
        padded_tgt = pad_data(tgt_seq, pad_value=0)#, add_eos=self.decoder.eos_idx)
        
        decoder_device = self.decoder.get_device()
        if decoder_device is not None:
            padded_tgt = self.xp.array(padded_tgt)

        max_tgt_length = padded_tgt.shape[1]
        seq_padded_tgt = [padded_tgt[:, i:i+1] for i in range(max_tgt_length)]
        

#         loss = F.softmax_cross_entropy(F.reshape(logits, (-1, self.decoder.V+1)), padded_target_with_eos.reshape(-1,))
        
        mb_size = len(src_seq)
        result = [logits]
        
        
        for num_step in range(max_tgt_length):
#             print "num_step", num_step
#             print "logits shape", logits.shape
            prev_word = seq_padded_tgt[num_step]
#             print "prev w shape", prev_word.shape
            assert prev_word.shape == (mb_size, 1)
            
            logits, decoder_state = decoding_cell(decoder_state, prev_word, train=train)
            result.append(logits)
            
#         print "seq_padded_tgt", seq_padded_tgt
        return result  
            
    def compute_loss_step_by_step(self, src_seq, tgt_seq, train=True):
        encoded_source, src_mask = self.encoder(src_seq, train=train)
        decoding_cell = self.decoder.get_conditionalized_cell(encoded_source, src_mask)
        
        logits, decoder_state = decoding_cell.get_initial_logits(train=train)            
        
        from nmt_chainer.models.feedforward.utils import pad_data
        padded_tgt = pad_data(tgt_seq, pad_value=0)#, add_eos=self.decoder.eos_idx)
        
        max_tgt_length = padded_tgt.shape[1]
        seq_padded_tgt = [padded_tgt[:, i] for i in range(max_tgt_length)]
        
        padded_target_with_eos = pad_data(tgt_seq, pad_value=-1, add_eos=self.decoder.eos_idx)
        
#         loss = F.softmax_cross_entropy(F.reshape(logits, (-1, self.decoder.V+1)), padded_target_with_eos.reshape(-1,))
        
        mb_size = len(src_seq)
        result = [[] for _ in xrange(mb_size)]
        
        num_step = 0
        while 1:
#             print "logits shape", logits.shape
            prev_word = padded_tgt[num_step]
#             print "prev w shape", prev_word.shape
            assert prev_word.shape == (mb_size, 1)
            for i in xrange(mb_size):
                result[i].append(prev_word[i, 0])
            
            prev_word = self.xp.where(prev_word == self.decoder.eos_idx, 0, prev_word)
            num_step += 1
            if num_step >= max_tgt_length:
                break
            logits, decoder_state = decoding_cell(decoder_state, prev_word, train=train)
        return result    
        
        