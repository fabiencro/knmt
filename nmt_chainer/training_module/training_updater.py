"""training_updater.py: training procedures."""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import math
import chainer
import time
import numpy

def sent_complexity(sent):
    rank_least_common_word = max(sent)
    length = len(sent)
    return length * math.log(rank_least_common_word + 1)


def example_complexity(ex):
    return sent_complexity(ex[0]) + sent_complexity(ex[1])




class SerialIteratorWithPeek(chainer.iterators.SerialIterator):

    def peek(self):
        """
        Return the next batch of data without updating its internal state.
        Several call to peek() should return the same result. A call to next()
        after a call to peek() will return the same result as the previous peek.
        """
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)
        if self._order is None:
            batch = self.dataset[i:i_end]
        else:
            batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    numpy.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch += list(self.dataset[:rest])
                    else:
                        batch += [self.dataset[index]
                                  for index in self._order[:rest]]
        return batch


class LengthBasedSerialIterator(chainer.dataset.iterator.Iterator):
    """
    This iterator will try to return batches with sequences of similar length.
    This is done by first extracting nb_of_batch_to_sort x batch_size sequences, sorting them by size,
    and then successively yielding nb_of_batch_to_sort batches of size batch_size.
    """

    def __init__(self, dataset, batch_size, nb_of_batch_to_sort=20, sort_key=lambda x: len(x[1]),
                 repeat=True, shuffle=True):

        self.sub_iterator = SerialIteratorWithPeek(dataset, batch_size * nb_of_batch_to_sort,
                                                   repeat=repeat, shuffle=shuffle)
        self.dataset = dataset
        self.index_in_sub_batch = 0
        self.sub_batch = None
        self.sort_key = sort_key
        self.batch_size = batch_size
        self.nb_of_batch_to_sort = nb_of_batch_to_sort
        self.repeat = repeat

    def update_sub_batch(self):
        # copy the result so that we can sort without side effects
        self.sub_batch = list(self.sub_iterator.next())
        if self.repeat and len(
    self.sub_batch) != self.batch_size * self.nb_of_batch_to_sort:
            raise AssertionError
        self.sub_batch.sort(key=self.sort_key)
        self.index_in_sub_batch = 0

    def __next__(self):
        if self.sub_batch is None or self.index_in_sub_batch >= self.nb_of_batch_to_sort:
            assert self.sub_batch is None or self.index_in_sub_batch == self.nb_of_batch_to_sort
            self.update_sub_batch()

        minibatch = self.sub_batch[self.index_in_sub_batch *
    self.batch_size: (self.index_in_sub_batch +
    1) *
     self.batch_size]

        self.index_in_sub_batch += 1

        return minibatch

    def peek(self):
        if self.sub_batch is None or self.index_in_sub_batch >= self.nb_of_batch_to_sort:
            assert self.sub_batch is None or self.index_in_sub_batch == self.nb_of_batch_to_sort
            # copy the result so that we can sort without side effects
            sub_batch = list(self.sub_iterator.peek())
            sub_batch.sort(key=self.sort_key)
            index_in_sub_batch = 0
        else:
            sub_batch = self.sub_batch
            index_in_sub_batch = self.index_in_sub_batch
        minibatch = sub_batch[index_in_sub_batch *
    self.batch_size: (index_in_sub_batch +
    1) *
     self.batch_size]
        return minibatch

    next = __next__

    # It is a bit complicated to keep an accurate value for epoch detail. In practice, the beginning of a sub_batch crossing epoch will have its
    # epoch_detail clipped to epoch
    @property
    def epoch_detail(self):
        remaining_sub_batch_lenth = (
    self.nb_of_batch_to_sort - self.index_in_sub_batch) * self.batch_size
        assert remaining_sub_batch_lenth >= 0
        epoch_discount = remaining_sub_batch_lenth / float(len(self.dataset))
        sub_epoch_detail = self.sub_iterator.epoch_detail
        epoch_detail = sub_epoch_detail - epoch_discount
        epoch_detail = max(epoch_detail, self.epoch)
        return epoch_detail

    # epoch and is_new_epoch are updated as soon as the end of the current
    # sub_batch has reached a new epoch.
    @property
    def epoch(self):
        return self.sub_iterator.epoch

    @property
    def is_new_epoch(self):
        if self.sub_iterator.is_new_epoch:
            assert self.sub_batch is None or self.index_in_sub_batch > 0
            if self.index_in_sub_batch == 1:
                return True
        return False

    # We do not serialize index_in_sub_batch. A deserialized iterator will start from the next sub_batch.
#     def serialize(self, serializer):
#         self.sub_iterator = serializer("sub_iterator", self.sub_iterator)
#         self.sub_iterator.serialize(serializer["sub_iterator"])


import six

def make_collection_of_variables(in_arrays, volatile = "off"):
    """Apply Variable constructor elementwise to a tuple, a dict or a single element"""
    if isinstance(in_arrays, tuple):
        in_vars = tuple(
    chainer.variable.Variable(
        x, volatile=volatile) for x in in_arrays)
    elif isinstance(in_arrays, dict):
        in_vars = {key: chainer.variable.Variable(x, volatile=volatile)
                   for key, x in six.iteritems(in_arrays)}
    else:
        in_vars = chainer.variable.Variable(in_arrays, volatile=volatile)
    return in_vars


class Updater(chainer.training.StandardUpdater):
    def __init__(self, iterator, optimizer, converter=chainer.dataset.convert.concat_examples,
                 device=None, loss_func=None, need_to_convert_to_variables=True):
        super(Updater, self).__init__(iterator, optimizer, converter=converter,
                                      device=device, loss_func=loss_func)
        self.need_to_convert_to_variables = need_to_convert_to_variables

    def update_core(self):
        t0 = time.clock()

        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if self.need_to_convert_to_variables:
            in_arrays = make_collection_of_variables(in_arrays)

        t1 = time.clock()

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)

        t2 = time.clock()
        update_duration = t2 - t0
        mb_preparation_duration = t1 - t0
        optimizer_update_cycle_duration = t2 - t1
        chainer.reporter.report({"update_duration": update_duration,
                                 "mb_preparation_duration": mb_preparation_duration,
                                 "optimizer_update_cycle_duration": optimizer_update_cycle_duration})

  
class UpdaterScheduledLearning(chainer.training.StandardUpdater):
    def __init__(self, iterator, optimizer, loss_per_example_func, converter=chainer.dataset.convert.concat_examples,
                 device=None, need_to_convert_to_variables = True):
        super(Updater, self).__init__(iterator, optimizer, converter=converter,
                 device=device, loss_func=loss_per_example_func)
        self.need_to_convert_to_variables = need_to_convert_to_variables
        
    def update_core(self):
        t0 = time.clock()
        
        batch, sent_ids = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        
        optimizer = self._optimizers['main']

        if self.need_to_convert_to_variables:
            in_arrays = make_collection_of_variables(in_arrays)
            
        t1 = time.clock()
        
        loss_each, loss = self.loss_func(*in_arrays)
        
        self._iterators['main'].update_knowledge(sent_ids, loss_each)
        
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()
        optimizer.target.cleargrads()

        t2 = time.clock()
        update_duration = t2 - t0
        mb_preparation_duration = t1-t0
        optimizer_update_cycle_duration = t2-t1
        chainer.reporter.report({"update_duration": update_duration,
                                 "mb_preparation_duration": mb_preparation_duration,
                                 "optimizer_update_cycle_duration": optimizer_update_cycle_duration})
      