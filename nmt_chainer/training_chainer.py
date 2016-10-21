#!/usr/bin/env python
"""training.py: training procedures."""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

from chainer import serializers
import time

import logging
import sys
# import h5py

import math

from utils import minibatch_provider, minibatch_provider_curiculum, make_batch_src_tgt
from evaluation import ( 
                  compute_loss_all, translate_to_file, sample_once)

import chainer.iterators
import chainer.dataset.iterator
import chainer.training
import chainer.training.extensions

logging.basicConfig()
log = logging.getLogger("rnns:training")
log.setLevel(logging.INFO)

def sent_complexity(sent):
    rank_least_common_word = max(sent)
    length = len(sent)
    return length * math.log(rank_least_common_word + 1)
    
def example_complexity(ex):
    return sent_complexity(ex[0]) + sent_complexity(ex[1])

class LengthBasedSerialIterator(chainer.dataset.iterator.Iterator):
    def __init__(self, dataset, batch_size, nb_of_batch_to_sort = 20, sort_key = lambda x:len(x[1]),
                 repeat=True, shuffle=True):
        
        self.sub_iterator = chainer.iterators.SerialIterator(dataset, batch_size * nb_of_batch_to_sort, 
                 repeat=repeat, shuffle=shuffle)
        self.dataset = dataset
        self.index_in_sub_batch = 0
        self.sub_batch = None
        self.sort_key = sort_key
        self.batch_size = batch_size
        self.nb_of_batch_to_sort = nb_of_batch_to_sort
                
    def update_sub_batch(self):
        self.sub_batch = list(self.sub_iterator.next()) # copy the result so that we can sort without side effects
        if len(self.sub_batch) != self.batch_size * self.nb_of_batch_to_sort:
            raise AssertionError
        self.sub_batch.sort(key = self.sort_key)
        self.index_in_sub_batch = 0
        
    def __next__(self):            
        if self.sub_batch is None or self.index_in_sub_batch >= self.nb_of_batch_to_sort:
            assert self.index_in_sub_batch == self.nb_of_batch_to_sort
            self.update_sub_batch()
        
        minibatch = self.sub_batch[self.index_in_sub_batch * self.batch_size: (self.index_in_sub_batch + 1) * self.batch_size]
            
        self.index_in_sub_batch += 1
        
        return minibatch

    next = __next__
        
        
    # It is a bit complicated to keep an accurate value for epoch detail. In practice, the beginning of a sub_batch crossing epoch will have its
    # epoch_detail pinned to epoch
    @property
    def epoch_detail(self):
        remaining_sub_batch_lenth = (self.nb_of_batch_to_sort - self.index_in_sub_batch) * self.batch_size
        assert remaining_sub_batch_lenth >= 0
        epoch_discount = remaining_sub_batch_lenth / float(len(self.dataset))
        sub_epoch_detail = self.sub_iterator.epoch_detail
        epoch_detail = sub_epoch_detail - epoch_discount
        epoch_detail = max(epoch_detail, self.epoch)
        return epoch_detail

    # epoch and is_new_epoch are updated as soon as the end of the current sub_batch has reached a new epoch.
    @property
    def epoch(self):
        return self.sub_iterator.epoch
    
    @property
    def is_new_epoch(self):
        if self.sub_iterator.is_new_epoch():
            assert self.sub_batch is None or self.index_in_sub_batch > 0
            if self.index_in_sub_batch == 1:
                return True
        return False


    # We do not serialize index_in_sub_batch. A deserialized iterator will start from the next sub_batch.
    def serialize(self, serializer):
        self.sub_iterator = serializer("sub_iterator", self.sub_iterator)


def train_on_data_chainer(encdec, optimizer, training_data, output_files_dict,
                  src_indexer, tgt_indexer, eos_idx, 
                  output_dir,
                  stop_trigger = None,
                  
                  
                  mb_size = 80,
                  nb_of_batch_to_sort = 20,
                  test_data = None, dev_data = None, valid_data = None,
                  gpu = None, report_every = 200, randomized = False,
                  reverse_src = False, reverse_tgt = False, max_nb_iters = None, do_not_save_data_for_resuming = False,
                  noise_on_prev_word = False, curiculum_training = False,
                  use_previous_prediction = 0, no_report_or_save = False,
                  use_memory_optimization = False, sample_every = 200,
                  use_reinf = False,
                  save_ckpt_every = 2000,
                  reshuffle_every_epoch = False):

#     iterator_training_data = chainer.iterators.SerialIterator(training_data, mb_size, 
#                                               repeat = True, 
#                                               shuffle = reshuffle_every_epoch)
    
    iterator_training_data = LengthBasedSerialIterator(training_data, mb_size, 
                                            nb_of_batch_to_sort = nb_of_batch_to_sort, 
                                            sort_key = lambda x:len(x[0]),
                                            repeat = True, 
                                            shuffle = reshuffle_every_epoch)
    
    def loss_func(src_batch, tgt_batch, src_mask):
        (total_loss, total_nb_predictions), attn = encdec(src_batch, tgt_batch, src_mask, raw_loss_info = True,
                                                          noise_on_prev_word = noise_on_prev_word,
                                                          use_previous_prediction = use_previous_prediction,
                                                          mode = "train")
        avg_loss = total_loss / total_nb_predictions
        chainer.reporter.report({"trg_loss": avg_loss.data})
        return avg_loss
    
    def convert_mb(mb_raw, device):
        return make_batch_src_tgt(mb_raw, eos_idx = eos_idx, padding_idx = 0, gpu = device, volatile = "off", need_arg_sort = False)    
    
    updater = chainer.training.StandardUpdater(iterator_training_data, optimizer,
                converter = convert_mb,#     iterator_training_data = chainer.iterators.SerialIterator(training_data, mb_size, 
#                                               repeat = True, 
#                                               shuffle = reshuffle_every_epoch)
                device = gpu, 
                loss_func = loss_func)

    trainer = chainer.training.Trainer(updater, stop_trigger, out = output_dir)
    trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'main/trg_loss']))
    trainer.run()

        