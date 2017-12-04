#!/usr/bin/env python
"""utils_tests.py: Some correctness tests"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import numpy as np
import chainer
from chainer import Link, Chain, ChainList, Variable
import chainer.functions as F
import chainer.links as L

import nmt_chainer.models as models
import nmt_chainer.utilities.utils as utils


from nmt_chainer.__main__ import main
from nmt_chainer.utilities.utils import de_batch


class TestDeBatch:
    def test_multiple_length(self):
        batch = [np.array([1, 3, 4, 8]), np.array([1, 5, 6, 9]), np.array([7, 5]), np.array([10])]
        seq_list = de_batch(batch)

        assert seq_list == [[1, 1, 7, 10], [3, 5, 5], [4, 6], [8, 9]]

    def test_multiple_length_variable(self):
        batch = [Variable(np.array(x, dtype=np.int32)) for x in [[1, 3, 4, 8], [1, 5, 6, 9], [7, 5], [10]]]
        seq_list = de_batch(batch, is_variable=True)

        assert seq_list == [[1, 1, 7, 10], [3, 5, 5], [4, 6], [8, 9]]

    def test_multiple_length_variable_raw(self):
        batch = [Variable(np.array(x, dtype=np.int32)) for x in [[1, 3, 4, 8], [1, 5, 6, 9], [[7, 9], [5, 8]], [10]]]
        seq_list = de_batch(batch, is_variable=True, raw=True)
        assert len(seq_list) == 4
        for seq1, seq2 in zip(seq_list, [[1, 1, [7, 9], 10], [3, 5, [5, 8]], [4, 6], [8, 9]]):
            assert len(seq1) == len(seq2)
            for elem1, elem2 in zip(seq1, seq2):
                assert np.all(elem1 == elem2)

    def test_multiple_length_eos_idx(self):
        batch = [np.array([1, 3, 4, 8]), np.array([3, 3, 6, 9]), np.array([7, 5]), np.array([10])]
        seq_list = de_batch(batch, eos_idx=3)
        assert seq_list == [[1, 3], [3], [4, 6], [8, 9]]

    def test_mask1(self):
        batch = [np.array([1, 3, 4, 8]), np.array([1, 5, 6, 9]), np.array([7, 5, 3, 4])]
        mask = [np.array([True, True, True, True]), np.array([True, True, True, True]), np.array([True, True, True, True])]
        seq_list = de_batch(batch, mask=mask)
        assert seq_list == [[1, 1, 7], [3, 5, 5], [4, 6, 3], [8, 9, 4]]

    def test_mask2(self):
        batch = [np.array([1, 3, 4, 8]), np.array([1, 5, 6, 9]), np.array([7, 5, 3, 4])]
        mask = [np.array([True, True, True, True]), np.array([True, True, False, True]), np.array([True, True, False, False])]
        seq_list = de_batch(batch, mask=mask)
        assert seq_list == [[1, 1, 7], [3, 5, 5], [4], [8, 9]]

    def test_mask3(self):
        batch = [np.array([1, 3, 4, 8]), np.array([1, 5, 6, 9]), np.array([7, 5, 3, 4])]
        mask = [np.array([True, True, False, True]), np.array([True, True, False, False])]
        seq_list = de_batch(batch, mask=mask)
        assert seq_list == [[1, 1, 7], [3, 5, 5], [4], [8, 9]]
#         print seq_list
#         print seq_list == [[1,1,[7,9],10], [3,5,[5,8]], [4,6], [8,9]]
#         assert np.all(seq_list == [[1,1,[7,9],10], [3,5,[5,8]], [4,6], [8,9]])


import nmt_chainer.dataprocessing.make_data as make_data
import nmt_chainer.training_module.train as train
import os.path

# test_data_dir = "tests_data"


class TestMakeData:
    def test_data_creation(self, tmpdir, gpu):
        test_data_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "../tests_data")
        train_dir = tmpdir.mkdir("train")
        data_prefix = str(train_dir.join("test1.data"))
        train_prefix = str(train_dir.join("test1.train"))
        args = ["make_data", os.path.join(test_data_dir, "src2.txt"), os.path.join(test_data_dir, "tgt2.txt"), data_prefix]
        main(arguments=args)

        args_train = ["train", data_prefix, train_prefix] + "--max_nb_iters 5 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23".split(" ")
        if gpu is not None:
            args_train += ['--gpu', gpu]
        main(arguments=args_train)
