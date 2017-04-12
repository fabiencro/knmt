#!/usr/bin/env python
"""macro_tests.py: Some macro tests"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

# import nmt_chainer.make_data as make_data
# import nmt_chainer.training_module.train as train
# import nmt_chainer.eval as eval
# import nmt_chainer.utilities.utils as utils

from nmt_chainer.__main__ import main

import os.path
import pytest


class TestMacro:

    def test_overfitting(self, tmpdir, gpu):
        """
        Test whether the translation results are equal to the target translations or not
        when the model is overtrained.
        """
        test_data_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "../tests_data")
        train_dir = tmpdir.mkdir("train")
        data_prefix = str(train_dir.join("test1.data"))
        train_prefix = str(train_dir.join("test1.train"))
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        args = 'make_data {0} {1} {2} --dev_src {0} --dev_tgt {1}'.format(
            data_src_file, data_tgt_file, data_prefix).split(' ')
        main(arguments=args)

        args_train = ["train", data_prefix, train_prefix] + "--max_nb_iters 1500 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23".split(" ")
        if gpu is not None:
            args_train += ['--gpu', gpu]
        main(arguments=args_train)

        eval_dir = tmpdir.mkdir("eval")
        translation_file = os.path.join(str(eval_dir), 'translations.txt')
        args_eval = ["eval", train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, translation_file] + '--mode beam_search --beam_width 30'.split(' ')
        if gpu is not None:
            args_eval += ['--gpu', gpu]
        main(arguments=args_eval)

        with open(data_tgt_file) as f:
            expected_translations = f.readlines()
        with open(translation_file) as f:
            actual_translations = f.readlines()
        print "expected_translations"
        for p in expected_translations:
            print p
        print "actual_translations"
        for p in actual_translations:
            print p

        assert(actual_translations == expected_translations)

    def test_compare_beam_search_vs_greedy_search(self, tmpdir, gpu):
        """
        Compare a beam search using a width of 1 with a greedy search and check
        whether the translation results are equal or not.
        """
        # At this moment, this test fails once in a while.
        # To increase the chance of finding a case where this test fails, I execute several times.
        for i in range(0, 10):
            test_data_dir = os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                "../tests_data")
            train_dir = tmpdir.mkdir("train_{0}".format(i))
            data_prefix = str(train_dir.join("test1.data"))
            train_prefix = str(train_dir.join("test1.train"))
            data_src_file = os.path.join(test_data_dir, "src2.txt")
            data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
            args = 'make_data {0} {1} {2} --dev_src {0} --dev_tgt {1}'.format(
                data_src_file, data_tgt_file, data_prefix).split(' ')
            main(arguments=args)

            args_train = ["train", data_prefix, train_prefix] + "--max_nb_iters 200 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23".split(" ")
            if gpu is not None:
                args_train += ['--gpu', gpu]
            main(arguments=args_train)

            beam_search_eval_dir = tmpdir.mkdir("eval_beam_search_{0}".format(i))
            beam_search_file = os.path.join(str(beam_search_eval_dir), 'translations.txt')
            args_eval = ["eval", train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, beam_search_file] + '--mode beam_search --beam_width 1'.split(' ')
            if gpu is not None:
                args_eval += ['--gpu', gpu]
            main(arguments=args_eval)

            greedy_search_eval_dir = tmpdir.mkdir("eval_greedy_search_{0}".format(i))
            greedy_search_file = os.path.join(str(greedy_search_eval_dir), 'translations.txt')
            args_eval = ["eval", train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, greedy_search_file] + '--mode translate'.split(' ')
            if gpu is not None:
                args_eval += ['--gpu', gpu]
            main(arguments=args_eval)

            with open(beam_search_file) as f:
                beam_search_translations = f.readlines()
            with open(greedy_search_file) as f:
                greedy_search_translations = f.readlines()
            print "beam_search_translations"
            for p in beam_search_translations:
                print p
            print "greedy_search_translations"
            for p in greedy_search_translations:
                print p

            assert(beam_search_translations == greedy_search_translations)

    def test_compare_beam_search_vs_same_ensemble_search(self, tmpdir, gpu):
        """
        Compare beam_search and a ensemble_beam_search using 3 identical models and
        check whether the translation results are equal or not.
        """
        test_data_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "../tests_data")
        train_dir = tmpdir.mkdir("train")
        data_prefix = str(train_dir.join("test1.data"))
        train_prefix = str(train_dir.join("test1.train"))
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        args = 'make_data {0} {1} {2} --dev_src {0} --dev_tgt {1}'.format(
            data_src_file, data_tgt_file, data_prefix).split(' ')
        main(arguments=args)

        args_train = ["train", data_prefix, train_prefix] + "--max_nb_iters 200 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23".split(" ")
        if gpu is not None:
            args_train += ['--gpu', gpu]
        main(arguments=args_train)

        beam_search_eval_dir = tmpdir.mkdir("eval_beam_search")
        beam_search_file = os.path.join(str(beam_search_eval_dir), 'translations.txt')
        args_eval = ["eval", train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, beam_search_file] + '--mode beam_search --beam_width 30'.split(' ')
        if gpu is not None:
            args_eval += ['--gpu', gpu]
        main(arguments=args_eval)

        ensemble_search_eval_dir = tmpdir.mkdir("eval_ensemble_search")
        ensemble_search_file = os.path.join(str(ensemble_search_eval_dir), 'translations.txt')
        args_eval = ["eval", train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, ensemble_search_file] + \
            '--mode beam_search --beam_width 30 --additional_training_config {0} {0} --additional_trained_model {1} {1}'.format(train_prefix + '.train.config', train_prefix + '.model.best.npz').split(' ')
        if gpu is not None:
            args_eval += ['--gpu', gpu]
        main(arguments=args_eval)

        with open(beam_search_file) as f:
            beam_search_translations = f.readlines()
        with open(ensemble_search_file) as f:
            ensemble_search_translations = f.readlines()
        print "beam_search_translations"
        for p in beam_search_translations:
            print p
        print "ensemble_search_translations"
        for p in ensemble_search_translations:
            print p

        assert(beam_search_translations == ensemble_search_translations)

    def test_compare_beam_search_vs_diff_ensemble_search(self, tmpdir, gpu):
        """
        Compare beam_search and a ensemble_beam_search using 3 different models and
        check whether the translation results are equal or not.  The results should
        differ most of the time although in theory, it's possible to be equal.
        """
        for i in range(0, 4):
            print i
            test_data_dir = os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                "../tests_data")
            train_dir = tmpdir.mkdir("train_{0}".format(i))
            data_prefix = str(train_dir.join("test1.data"))
            train_prefix = str(train_dir.join("test1.train"))
            data_src_file = os.path.join(test_data_dir, "src2.txt")
            data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
            args = 'make_data {0} {1} {2} --dev_src {0} --dev_tgt {1}'.format(
                data_src_file, data_tgt_file, data_prefix).split(' ')
            main(arguments=args)

            args_train = ["train", data_prefix, train_prefix] + "--max_nb_iters 200 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23".split(" ")
            if gpu is not None:
                args_train += ['--gpu', gpu]
            main(arguments=args_train)

        train_dir = str(tmpdir.join("train_0"))
        train_prefix = os.path.join(train_dir, "test1.train")
        beam_search_eval_dir = tmpdir.mkdir("eval_beam_search")
        beam_search_file = os.path.join(str(beam_search_eval_dir), 'translations.txt')
        args_eval_beam_search = ["eval", train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, beam_search_file] + '--mode beam_search --beam_width 30'.split(' ')
        if gpu is not None:
            args_eval_beam_search += ['--gpu', gpu]
        main(arguments=args_eval_beam_search)

        ensemble_search_eval_dir = tmpdir.mkdir("eval_ensemble_search")
        ensemble_search_file = os.path.join(
            str(ensemble_search_eval_dir), 'translations.txt')
        train_dir_1 = str(tmpdir.join("train_1"))
        train_prefix_1 = os.path.join(train_dir_1, "test1.train")
        train_dir_2 = str(tmpdir.join("train_2"))
        train_prefix_2 = os.path.join(train_dir_2, "test1.train")
        train_dir_3 = str(tmpdir.join("train_3"))
        train_prefix_3 = os.path.join(train_dir_3, "test1.train")
        args_eval_ensemble_search = ["eval", train_prefix_1 + '.train.config', train_prefix_1 + '.model.best.npz', data_src_file, ensemble_search_file] + \
            '--mode beam_search --beam_width 30 --additional_training_config {0} {1} --additional_trained_model {2} {3}'.format(
                train_prefix_2 + '.train.config', train_prefix_3 + '.train.config', train_prefix_2 + '.model.best.npz', train_prefix_3 + '.model.best.npz').split(' ')
        if gpu is not None:
            args_eval_ensemble_search += ['--gpu', gpu]
        main(arguments=args_eval_ensemble_search)

        with open(beam_search_file) as f:
            beam_search_translations = f.readlines()
        with open(ensemble_search_file) as f:
            ensemble_search_translations = f.readlines()
        print "beam_search_translations"
        for p in beam_search_translations:
            print p
        print "ensemble_search_translations"
        for p in ensemble_search_translations:
            print p

        assert(beam_search_translations != ensemble_search_translations)
