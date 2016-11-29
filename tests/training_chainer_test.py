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
import nmt_chainer.utils as utils

from nmt_chainer.utils import de_batch
import nmt_chainer.training_chainer as iterators
from numpy.random import RandomState

def generate_random_dataset():
    random_generator = RandomState(42)
    
    dataset_size = 64
    dataset = []
    for i in range(0, dataset_size):
        item_count = random_generator.randint(17, 24)
        item = tuple(random_generator.randint(0, 100, size = item_count))
        dataset.append(item)

    return dataset

class TestSerialIterator:

    dataset_size = 64
    batch_size = 20

    def test_repeat_loop(self):
        """
        Test if the order of the elements of the dataset is the same when repeat = True.
        """
        dataset = generate_random_dataset()

        random_generator = RandomState(0)

        iter = iterators.SerialIteratorWithPeek(dataset, TestSerialIterator.batch_size, shuffle = False, repeat = True)
        m = 0
        L = len(dataset)
        for i in range(0, len(dataset) * random_generator.randint(5, 20)):
            batch = iter.next()
            if m >= L:
                m -= L
                items = dataset[m:m + TestSerialIterator.batch_size]
            elif m < L and m + TestSerialIterator.batch_size >= L:
                items = dataset[m:L] + dataset[0:m + TestSerialIterator.batch_size - L]
                m = TestSerialIterator.batch_size - (L - m) 
            else:
                items = dataset[m:m + TestSerialIterator.batch_size]
                m += TestSerialIterator.batch_size
            assert batch == items
        
    # The test where shuffle, repeat, and check_peek fails because the peek() method
    # cannot foresee how the element will be shuffled by the next() method at the end of an epoch. - FB
    @pytest.mark.parametrize("shuffle,repeat,check_peek", 
        [
         (True, True, False), 
         (False, True, False), 
         (True, False, False), 
         (False, False, False),
         #(True, True, True),
         (False, True, True), 
         (True, False, True), 
         (False, False, True)
         ])
    def test_retrieve_all_items(self, shuffle, repeat, check_peek):
        """
        Test if we can retrieve all the items of the dataset from the iterator,
        no matter what options we choose.  Essentially, this tests that the next()
        method works properly.  When check_peek is True, it also tests if
        iter.peek() == iter.next() right before calling iter.next().
        """
        dataset = generate_random_dataset()
        
        random_generator = RandomState(0)

        iter = iterators.SerialIteratorWithPeek(dataset, TestSerialIterator.batch_size, shuffle = shuffle, repeat = repeat)
        first_iter_items = None
        work_dataset = []
        batch = []
        item_count = 0
        if repeat:
            dataset_count = random_generator.randint(5, 20)
        else:
            dataset_count = 1
        for i in range(0, len(dataset) * dataset_count):
            if i % TestSerialIterator.dataset_size == 0:
                assert len(work_dataset) == 0
                work_dataset = list(dataset)
            if i % TestSerialIterator.batch_size == 0:
                assert len(batch) == 0
                if check_peek:
                    peek_item = iter.peek()
                batch = iter.next()
                if check_peek:
                    assert batch == peek_item
                item_count = 0
            item = batch.pop(0)
            work_dataset.remove(item)
            item_count += 1
        assert len(work_dataset) == 0
        if repeat:
            assert len(batch) == TestSerialIterator.batch_size - ((len(dataset) * dataset_count) % TestSerialIterator.batch_size) 
        else:
            assert len(batch) == 0
