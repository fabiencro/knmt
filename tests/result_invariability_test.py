#!/usr/bin/env python
"""result_invariabitility_tests.py: Check if the code gives the same results."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import nmt_chainer.make_data as make_data
import nmt_chainer.train as train
import nmt_chainer.eval as eval
import os.path
import pytest

class TestResultInvariability:
    
    def test_result_invariability(self, tmpdir, gpu):
        """
        Performs some translations with a preexisting model and compare the results
        with previous results of the same experiment.  The result should be identical.
        If not, it means that a recent commit have changed the behavior of the system.
        """
        
        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_data")
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        train_dir = os.path.join(test_data_dir, "models")
        train_prefix = os.path.join(train_dir, "result_invariability.train")
        beam_search_eval_dir = tmpdir.mkdir("eval_beam_search")
        beam_search_file = os.path.join(str(beam_search_eval_dir), 'translations.txt')
        args_eval_beam_search = [train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, beam_search_file] + \
            '--mode beam_search --beam_width 30'.split(' ') 
        if gpu is not None:
            args_eval_beam_search += ['--gpu', gpu]
        eval.command_line(arguments = args_eval_beam_search)
        
        with open(os.path.join(str(test_data_dir), "models/result_invariability.translations.txt")) as f:
            expected_translations = f.readlines()
        with open(beam_search_file) as f:
            actual_translations = f.readlines()
        print "expected_translations"
        for p in expected_translations:
            print p
        print "actual_translations"
        for p in actual_translations:
            print p

        assert(actual_translations == expected_translations)

