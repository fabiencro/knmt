#!/usr/bin/env python
"""training_chainer_tests.py: Some unit tests for the training classes"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import pytest
import numpy as np
import chainer
from chainer import Link, Chain, ChainList, Variable
import chainer.functions as F
import chainer.links as L
import math
import nmt_chainer.models as models
import nmt_chainer.utilities.utils as utils

from nmt_chainer.utilities.utils import de_batch
import nmt_chainer.training_module.training_chainer as iterators
from numpy.random import RandomState


def generate_random_dataset(dataset_size, sequence_avg_size):
    """
    Build a dummy data structure with random numbers similar to our
    usual datasets that is a list of pairs of sequences of numbers:
    [ ( [1, 2, 10, ...], [3, 5, 6, 7, ...]),
      ( [5, 3], [34, 23, 44, 1, ...] ),
      ... ]
    """
    random_generator = RandomState(42)

    dataset = []
    for i in range(0, dataset_size):
        item = []
        # Each item contains 2 sequences.
        for j in range(0, 2):
            sequence_length = random_generator.randint(sequence_avg_size - 5, sequence_avg_size + 5)
            # sequence_length = random_generator.randint(1, 5)
            sequence = random_generator.randint(0, 100, size=sequence_length)
            item.append(sequence)
        item = tuple(item)
        dataset.append(item)

    return dataset


class TestSerialIterator:

    @pytest.mark.parametrize("dataset_size, batch_size, sequence_avg_size", [
        (64, 20, 16),
        (200, 20, 20)
    ])
    def test_repeat_loop(self, dataset_size, batch_size, sequence_avg_size):
        """
        Test if the order of the elements of the dataset is the same when repeat = True.
        """
        dataset = generate_random_dataset(dataset_size, sequence_avg_size)

        random_generator = RandomState(0)

        iter = iterators.SerialIteratorWithPeek(dataset, batch_size, shuffle=False, repeat=True)
        m = 0
        L = len(dataset)
        # for i in range(0, len(dataset) * random_generator.randint(5, 20)):
        for i in range(0, 1):
            batch = iter.next()
            if m >= L:
                m -= L
                items = dataset[m:m + batch_size]
            elif m < L and m + batch_size >= L:
                items = dataset[m:L] + dataset[0:m + batch_size - L]
                m = batch_size - (L - m)
            else:
                items = dataset[m:m + batch_size]
                m += batch_size
            assert batch == items

    # The test where shuffle, repeat, and check_peek fails because the peek() method
    # cannot foresee how the elements will be shuffled by the next() method at the end of an epoch. - FB
    @pytest.mark.parametrize("dataset_size, batch_size, sequence_avg_size, shuffle, repeat, check_peek",
                             [
                                 (64, 20, 16, True, True, False),
                                 (64, 20, 16, False, True, False),
                                 (64, 20, 16, True, False, False),
                                 (64, 20, 16, False, False, False),
                                 # (64, 20, 16, True, True, True),
                                 (64, 20, 16, False, True, True),
                                 (64, 20, 16, True, False, True),
                                 (64, 20, 16, False, False, True),
                                 (200, 20, 20, True, True, False),
                                 (200, 20, 20, False, True, False),
                                 (200, 20, 20, True, False, False),
                                 (200, 20, 20, False, False, False),
                                 # (200, 20, 20, True, True, True),
                                 (200, 20, 20, False, True, True),
                                 (200, 20, 20, True, False, True),
                                 (200, 20, 20, False, False, True),
                             ])
    def test_retrieve_all_items(self, dataset_size, batch_size, sequence_avg_size, shuffle, repeat, check_peek):
        """
        Test if we can retrieve all the items of the dataset from the iterator,
        no matter what options we choose.  Essentially, this tests that the next()
        method works properly.  When check_peek is True, it also tests if
        iter.peek() == iter.next() right before calling iter.next().
        """
        dataset = generate_random_dataset(dataset_size, sequence_avg_size)

        random_generator = RandomState(0)

        iter = iterators.SerialIteratorWithPeek(dataset, batch_size, shuffle=shuffle, repeat=repeat)
        first_iter_items = None
        work_dataset = []
        batch = []
        item_count = 0
        if repeat:
            dataset_count = random_generator.randint(5, 20)
        else:
            dataset_count = 1
        for i in range(0, len(dataset) * dataset_count):
            if i % dataset_size == 0:
                assert len(work_dataset) == 0
                work_dataset = list(dataset)
            if i % batch_size == 0:
                assert len(batch) == 0
                if check_peek:
                    peek_item = iter.peek()
                batch = iter.next()
                if check_peek:
                    assert batch == peek_item
                item_count = 0
            first_item = batch.pop(0)
            # work_dataset.remove(first_item)
            first_item_index = -1
            for item_idx, item in enumerate(work_dataset):
                if np.array_equal(item, first_item):
                    first_item_index = item_idx
                    break
            del work_dataset[first_item_index]

            item_count += 1
        assert len(work_dataset) == 0
        if repeat:
            assert len(batch) == 0 or len(batch) == batch_size - ((len(dataset) * dataset_count) % batch_size)
        else:
            assert len(batch) == 0


class TestLengthBasedSerialIterator():

    @pytest.mark.parametrize("dataset_size, batch_size, sequence_avg_size", [
        (64, 20, 16),
        (200, 20, 20)
    ])
    def test_peek_egal_next(self, dataset_size, batch_size, sequence_avg_size):
        """
        Test if the value returned by the peek() method matches
        the value returned by the next() method right after calling peek().
        """
        dataset = generate_random_dataset(dataset_size, sequence_avg_size)

        iter = iterators.LengthBasedSerialIterator(dataset, batch_size, shuffle=False, repeat=False)
        while True:
            try:
                peek_value = iter.peek()
                next_value = iter.next()
                # print "{0} vs {1}".format(peek_value, next_value)
                assert peek_value == next_value
            except StopIteration:
                break

    @pytest.mark.parametrize("dataset_size, batch_size, sequence_avg_size", [
        (64, 20, 16),
        (200, 20, 20)
    ])
    def test_batch_size(self, dataset_size, batch_size, sequence_avg_size):
        """
        Test if the batches have sequences of similar length.
        We start by measuring the average sequence length.
        Then, for each batch, we examine each of its pairs.
        We compare the length of the second sequence of the pair
        with the previous one and make sure that the length is not
        too different.
        For each pair, we also make sure that the length of the second
        sequence does not differ too much with the average sequence length.
        """
        dataset = generate_random_dataset(dataset_size, sequence_avg_size)

        iter = iterators.LengthBasedSerialIterator(dataset, batch_size, shuffle=False, repeat=False)

        # Compute the average sequence length.
        total = 0
        count = 0
        for batch in iter:
            for pairs in batch:
                # Consider the second sequence because the default sort_key
                # uses this sequence.
                seq_length = len(pairs[1])
                if seq_length > 0:
                    count += 1
                    total += seq_length
        avg_seq_length = total / count
        # print "avg={0}".format(avg_seq_length)

        prev_seq_length = None
        for batch in iter:
            # print "batch size={0}".format(len(batch))
            for pairs in batch:
                # print "seq1.length={0} seq.length={1}".format(len(pairs[0]),len(pairs[1]))
                # Consider the second sequence because the default sort_key uses this sequence.
                seq_length = len(pairs[1])
                if seq_length > 0:
                    count += 1
                    total += seq_length
                if prev_seq_length is not None:
                    assert abs(prev_seq_length - seq_length) <= 5
                    assert abs(seq_length - avg_seq_length) <= 10
