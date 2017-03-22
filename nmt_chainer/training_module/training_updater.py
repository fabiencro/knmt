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
import heapq

from chainer.dataset import iterator

def sent_complexity(sent):
    rank_least_common_word = max(sent)
    length = len(sent)
    return length * math.log(rank_least_common_word + 1)


def example_complexity(ex):
    return sent_complexity(ex[0]) + sent_complexity(ex[1])

class SpacedRepetitor(object):
    def __init__(self, ratios = numpy.array([0.5] + [0.5**num for num in range(1, 4)]), cap_unknown = None):
        nb_boxes = len(ratios)
        self.unk_box = []
        self.boxes = [[] for _ in range(nb_boxes)] # + unk box
        self.nb_boxes = nb_boxes
        self.being_evaluated = {}
        
        self.ratios = ratios
        self.ratio_remainders = numpy.zeros((nb_boxes,))
        self.cap_unknown = cap_unknown
        
    def __repr__(self):
        res = ["#UNK:%i"%len(self.boxes[0])]
        for num_box in range(1, self.nb_boxes):
            res += ["#%i:%i "%(num_box, len(self.boxes[num_box]))]
        return "SR<" + " ".join(res) + ">"
    
    def __str__(self):
        res = ["UNK:%s"%(self.boxes[0],)]
        for num_box in range(1, self.nb_boxes):
            res += ["%i:%s "%(num_box, self.boxes[num_box])]
        return "\n".join(res)
        
    def add_dataset(self, dataset):
        self.boxes[0] = sorted(range(len(dataset)), key = lambda i:example_complexity(dataset[i]))
    
    def get_n_examples(self, nb_ex, just_peek = False):
        boxes_sizes = numpy.array([float(len(box)) for box in self.boxes])
        assert numpy.sum(boxes_sizes) >= nb_ex
        
        boxes_ratios = boxes_sizes * self.ratios
        
        if self.cap_unknown is not None and boxes_sizes[1] > self.cap_unknown:
            boxes_ratios[0] = 0
        
        boxes_ratios /= sum(boxes_ratios)
        
        
        fractional_nb = nb_ex * boxes_ratios + self.ratio_remainders
        rounded_nb = numpy.maximum(numpy.round(fractional_nb), 0) * (boxes_sizes > 0)
        rounded_nb = rounded_nb.astype(numpy.int)
        
        over_value = numpy.sum(rounded_nb) - nb_ex
        
        if over_value < 0:
            rounded_nb[rounded_nb.argmax()] += -over_value
        elif over_value > 0:
            for _ in xrange(over_value):
                rounded_nb[rounded_nb.argmax()] -= 1

        if not just_peek:
            self.ratio_remainders = fractional_nb - rounded_nb
        
        res = []
        
        for num_box in xrange(self.nb_boxes):
            box = self.boxes[num_box]
            expected_nb = rounded_nb[num_box]
            if len(box) < expected_nb:
                print "using duplication for", num_box, box, len(box), expected_nb
                duplicated_box = box[:]
                remaining = expected_nb - len(box)
                while remaining > 0:
                    duplicated_box += box[:remaining]
                    remaining -= len(box)
                    
                print "duplicated box", duplicated_box
                res += duplicated_box
                if not just_peek:
                    for ex in box:
                        self.being_evaluated[ex] = num_box
                    self.boxes[num_box] = []
            else:
                res += box[:expected_nb]
                if not just_peek:
                    for ex in box[:expected_nb]:
                        self.being_evaluated[ex] = num_box
                    self.boxes[num_box] = box[expected_nb:]
        
        assert len(res) == nb_ex
        return res
        
        
    def update_known(self, good, bad):
        seen = set()
        for ex in good:
            if ex in seen:
                continue #probably a duplicated ex
            assert ex in self.being_evaluated
            box = self.being_evaluated.pop(ex)
            if box == 0:
                box = 2 #case unk -> 2
            elif box+1 < self.nb_boxes:
                box += 1
            self.boxes[box].append(ex)
            seen.add(ex)
        for ex in bad:
            if ex in seen:
                continue #probably a duplicated ex
            assert ex in self.being_evaluated
            box = self.being_evaluated.pop(ex)
            self.boxes[1].append(ex)            
            seen.add(ex)
    
    def serialize(self, serializer):
        self.sr = serializer('sr', self.sr)
        
    
class ScheduledIterator(iterator.Iterator):

    """Dataset iterator that serially reads the examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.

    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.

    """

    def __init__(self, dataset, batch_size, repeat = False, shuffle = False, 
                 sr_ratios = numpy.array([0.5] + [0.5**num for num in range(1, 4)]),
                 sr_cap = None):
        self.dataset = dataset
        self.batch_size = batch_size

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        self.sr = SpacedRepetitor(ratios=sr_ratios, cap_unknown=sr_cap)
        self.sr.add_dataset(dataset)
    
    def __next__(self):
        print "__next__ sr state:", repr(self.sr),
        indices = self.sr.get_n_examples(self.batch_size)
        print " -> ", repr(self.sr)
        
        batch = [self.dataset[i] for i in indices]
    
        return indices, batch            
         
    def update_known(self, good, bad):
        self.sr.update_known(good, bad)

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    def serialize(self, serializer):
        self.sr = serializer('sr', self.sr)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)

    def peek(self):
        """
        Return the next batch of data without updating its internal state.
        Several call to peek() should return the same result. A call to next()
        after a call to peek() will return the same result as the previous peek.
        """
        print "peek sr state:", repr(self.sr),
        indices = self.sr.get_n_examples(self.batch_size, just_peek = True)
        print " -> ", repr(self.sr)
        
        batch = [self.dataset[i] for i in indices]
    
        return indices, batch   


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
                 repeat=True, shuffle=True, subiterator_type = SerialIteratorWithPeek, composed_batch = False,
                 subiterator_keyword_args = {}):

        self.sub_iterator = subiterator_type(dataset, batch_size * nb_of_batch_to_sort,
                                                   repeat=repeat, shuffle=shuffle, **subiterator_keyword_args)
        self.dataset = dataset
        self.index_in_sub_batch = 0
        self.sub_batch = None
        self.sort_key = sort_key
        self.batch_size = batch_size
        self.nb_of_batch_to_sort = nb_of_batch_to_sort
        self.repeat = repeat
        self.composed_batch = composed_batch

    def update_sub_batch(self):
        self.sub_batch = self.sort_batch(self.sub_iterator.next())
        self.index_in_sub_batch = 0

    def __next__(self):
        if self.sub_batch is None or self.index_in_sub_batch >= self.nb_of_batch_to_sort:
            assert self.sub_batch is None or self.index_in_sub_batch == self.nb_of_batch_to_sort
            self.update_sub_batch()

        minibatch = self.extract_sub_batch(self.index_in_sub_batch, self.sub_batch)

        self.index_in_sub_batch += 1

        return minibatch

    def update_known(self, good, bad):
        self.sub_iterator.update_known(good, bad)
        
    
    def sort_batch(self, batch):
        if self.composed_batch:
            batch=  zip(*sorted(zip(*batch), key=self.sort_key))
        else:
            # copy the result so that we can sort without side effects
            batch = list(self.sub_iterator.next())
            if self.repeat and len(batch) != self.batch_size * self.nb_of_batch_to_sort:
                raise AssertionError
            batch.sort(key=self.sort_key)
        return batch
        
    def extract_sub_batch(self, index_in_sub_batch, batch):
        start_index = index_in_sub_batch * self.batch_size
        end_index = (index_in_sub_batch + 1) *  self.batch_size
        
        if self.composed_batch:
            minibatch = [component[start_index: end_index] for component in batch]
        else:
            minibatch = batch[start_index: end_index]
        return minibatch
            
    def peek(self):
        if self.sub_batch is None or self.index_in_sub_batch >= self.nb_of_batch_to_sort:
            assert self.sub_batch is None or self.index_in_sub_batch == self.nb_of_batch_to_sort
            # copy the result so that we can sort without side effects
            sub_batch = list(self.sub_iterator.peek())
            sub_batch = self.sort_batch(sub_batch)
            index_in_sub_batch = 0
        else:
            sub_batch = self.sub_batch
            index_in_sub_batch = self.index_in_sub_batch
            
        minibatch = self.extract_sub_batch(index_in_sub_batch, sub_batch)

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
                 device=None, need_to_convert_to_variables = True, loss_threshold = 1):
        super(UpdaterScheduledLearning, self).__init__(iterator, optimizer, converter=converter,
                 device=device, loss_func=loss_per_example_func)
        self.need_to_convert_to_variables = need_to_convert_to_variables
        self.loss_threshold = loss_threshold
        
    def update_core(self):
        t0 = time.clock()
        
        sent_ids, batch = self._iterators['main'].next()
        src_batch, tgt_batch_v, src_mask, argsort = self.converter(batch, self.device)
        in_arrays = [src_batch, tgt_batch_v, src_mask]
        
        optimizer = self._optimizers['main']

        if self.need_to_convert_to_variables:
            in_arrays = make_collection_of_variables(in_arrays)
            
        t1 = time.clock()
        
        loss_each, loss = self.loss_func(*in_arrays)
        
        good = set()
        bad = set()
        for i in xrange(loss_each.data.shape[0]):
            tgt_length = len(batch[argsort[i]][1])
            loss_per_word = loss_each.data[i] / tgt_length
            if loss_per_word < self.loss_threshold:
                good.add(sent_ids[argsort[i]])
            else:
                bad.add(sent_ids[argsort[i]])
        
        self._iterators['main'].update_known(good=good, bad=bad)
        
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
      