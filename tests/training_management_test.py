#!/usr/bin/env python
"""macro_tests.py: Some macro tests"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import nmt_chainer.make_data as make_data
import nmt_chainer.train as train
import nmt_chainer.eval as eval
import os.path
import pytest

class TestTrainingManagement:
    
    def test_checkpoint_saving(self, tmpdir, gpu):
        """
        Test no error happens during checkpoint saving.
        """
        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_data")
        train_dir = tmpdir.mkdir("train")
        data_prefix = str(train_dir.join("test1.data"))
        train_prefix = str(train_dir.join("test1.train"))
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        args = '{0} {1} {2} --dev_src {0} --dev_tgt {1}'.format(
            data_src_file, data_tgt_file, data_prefix).split(' ')
        make_data.cmdline(arguments = args)
        
        args_train = [data_prefix, train_prefix] + "--max_nb_iters 10 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23 --save_ckpt_every 5".split(" ")
        if gpu is not None:
            args_train += ['--gpu', gpu]
        train.command_line(arguments = args_train)