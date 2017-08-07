import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Chain, ChainList

from nmt_chainer.models.feedforward.encoder import Encoder
from nmt_chainer.models.feedforward.decoder import Decoder
from nmt_chainer.utilities.utils import batch_sort_and_split
from __builtin__ import True
    
import logging
logging.basicConfig()
log = logging.getLogger("ff:encdec")
log.setLevel(logging.INFO)
    
class EncoderDecoder(Chain):
    def __init__(self, V_src, V_tgt, d_model=512, n_heads=8, d_ff=2048, experimental_relu=False, dropout=None, 
                 nb_layers_src=6, nb_layers_tgt=6, residual_mode="normal", no_normalize=False):
        super(EncoderDecoder, self).__init__(
            encoder = Encoder(V_src, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                              experimental_relu=experimental_relu, dropout=dropout, nb_layers=nb_layers_src,
                              residual_mode=residual_mode, no_normalize=no_normalize),
            decoder = Decoder(V_tgt, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                              experimental_relu=experimental_relu, dropout=dropout, nb_layers=nb_layers_tgt,
                              residual_mode=residual_mode, no_normalize=no_normalize),
        )
        
        self.V_tgt = V_tgt
        
        log.info("Creating FF EncoderDecoder V: %i -> %i dm:%i h:%i dff:%i er:%i dropout:%r layers:%i -> %i resid:%s nonorm:%i", V_src, V_tgt, d_model, n_heads, d_ff, 
                 experimental_relu, dropout, 
                 nb_layers_src, nb_layers_tgt, residual_mode, no_normalize)
        
    def encdec_type(self):
        return "ff"
        
    def compute_test_loss(self, test_data, mb_size=64, nb_mb_for_sorting= 20):
        def mb_provider():
            required_data = nb_mb_for_sorting * mb_size
            cursor = 0
            while cursor < len(test_data):
                larger_batch = test_data[cursor:cursor+required_data]
                cursor += required_data
                for minibatch in batch_sort_and_split(larger_batch, size_parts = mb_size):
                    yield zip(*minibatch)
        
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            total_loss = 0
            total_nb_predictions = 0.0     
            for src_batch, tgt_batch in mb_provider():
                loss = self.compute_loss(src_batch, tgt_batch, reduce="no")
                nb_tgt_words = sum(len(seq) + 1 for seq in tgt_batch) # +1 for eos
                total_loss += self.xp.sum(loss.data)
                total_nb_predictions += nb_tgt_words
            return total_loss / total_nb_predictions
        
    def greedy_batch_translate(self, test_data,  mb_size=64, nb_mb_for_sorting= 20, nb_steps=50):
        test_data_with_index = zip(test_data, range(len(test_data)))
        def mb_provider(): #TODO: optimize by sorting by size
            required_data = nb_mb_for_sorting * mb_size
            cursor = 0
            while cursor < len(test_data):
                larger_batch = test_data_with_index[cursor:cursor+required_data]
                cursor += required_data
                for minibatch in batch_sort_and_split(larger_batch, size_parts = mb_size,
                                            sort_key=lambda x: len(x[0])):
                
                    yield minibatch
                
        result = []
        for src_batch_with_index in mb_provider():
            src_batch, indexes = zip(*src_batch_with_index)
            translated = self.greedy_translate(src_batch, nb_steps=nb_steps)
            result += zip(indexes, translated) 
            
        result.sort(key=lambda x:x[0])
        reordered_indexes, reordered_result = zip(*result)
        assert reordered_indexes == range(len(test_data))
        return reordered_result
        
    def compute_loss(self, src_seq, tgt_seq, reduce="mean"):
        encoded_source, src_mask = self.encoder(src_seq)
        loss = self.decoder.compute_loss(tgt_seq, encoded_source, src_mask, reduce=reduce)
        return loss
        
    def greedy_translate(self, src_seq, nb_steps, cut_eos=True, sample=False):
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            encoded_source, src_mask = self.encoder(src_seq)
            decoding_cell = self.decoder.get_conditionalized_cell(encoded_source, src_mask)
            
            logits, decoder_state = decoding_cell.get_initial_logits()
            
            mb_size = len(src_seq)
            result = [[] for _ in xrange(mb_size)]
            finished = [False for _ in xrange(mb_size)]
            
            num_step = 0
            while 1:
                if sample:
                    logits = logits + self.xp.random.gumbel(size=logits.data.shape).astype(self.xp.float32)
                
    #             print "logits shape", logits.shape
                prev_word = self.xp.argmax(logits.data, axis = 1).reshape(-1, 1).astype(self.xp.int32)
    #             print "prev w shape", prev_word.shape
                assert prev_word.shape == (mb_size, 1)
                for i in xrange(mb_size):
                    if not finished[i]:
                        if cut_eos and prev_word[i, 0] == self.decoder.eos_idx:
                            finished[i] = True
                            continue
                        result[i].append(prev_word[i, 0])
                
                prev_word = self.xp.where(prev_word == self.decoder.eos_idx, 0, prev_word)
                num_step += 1
                if num_step > nb_steps:
                    break
                logits, decoder_state = decoding_cell(decoder_state, prev_word)
                
            result = [[int(x) for x in seq] for seq in result]
            return result
            
    def compute_logits(self, src_seq, tgt_seq):
        encoded_source, src_mask = self.encoder(src_seq)
        logits = self.decoder.compute_logits(tgt_seq, encoded_source, src_mask)
        return logits
      
    def compute_logits_step_by_step(self, src_seq, tgt_seq):
        encoded_source, src_mask = self.encoder(src_seq)
        decoding_cell = self.decoder.get_conditionalized_cell(encoded_source, src_mask)
        
        logits, decoder_state = decoding_cell.get_initial_logits()            
        
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
            
            logits, decoder_state = decoding_cell(decoder_state, prev_word)
            result.append(logits)
            
#         print "seq_padded_tgt", seq_padded_tgt
        return result  
            
    def compute_loss_step_by_step(self, src_seq, tgt_seq):
        encoded_source, src_mask = self.encoder(src_seq)
        decoding_cell = self.decoder.get_conditionalized_cell(encoded_source, src_mask)
        
        logits, decoder_state = decoding_cell.get_initial_logits()            
        
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
        
        