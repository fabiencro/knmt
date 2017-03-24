#!/usr/bin/env python
"""result_invariabitility_tests.py: Check if the code gives the same results."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

from nmt_chainer.__main__ import main
import os.path
import pytest


class TestResultInvariability:

    @pytest.mark.parametrize("model_name, search_type", [
        ("result_invariability", "beam_search"),
        ("result_invariability", "greedy_search"),
        ("result_invariability_untrained", "beam_search"),
        ("result_invariability_untrained", "greedy_search")
    ])
    def test_result_invariability(self, tmpdir, gpu, model_name, search_type):
        """
        Performs some translations with a preexisting models and compare the results
        using various search types translation with previous results of the same experiment.
        The results should be identical.
        If not, it means that a recent commit have changed the behavior of the system.
        """

        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_data")
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        train_dir = os.path.join(test_data_dir, "models")
        train_prefix = os.path.join(train_dir, "{0}.train".format(model_name))
        search_eval_dir = tmpdir.mkdir("eval_{0}".format(search_type))
        search_file = os.path.join(
            str(search_eval_dir),
            'translations_using_{0}.txt'.format(search_type))
        search_mode = 'beam_search'
        other_params = ' --beam_width 30'
        if search_type == 'greedy_search':
            search_mode = 'translate'
            other_params = ''
        args_eval_search = [train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, search_file] + \
            '--mode {0}{1}'.format(search_mode, other_params).split(' ')
        if gpu is not None:
            args_eval_search += ['--gpu', gpu]
        main(arguments=["eval"] + args_eval_search)

        with open(os.path.join(str(test_data_dir), "models/{0}.translations_using_{1}.txt".format(model_name, search_type))) as f:
            expected_translations = f.readlines()
        with open(search_file) as f:
            actual_translations = f.readlines()
        print "expected_translations"
        for p in expected_translations:
            print p
        print "actual_translations"
        for p in actual_translations:
            print p

        assert(actual_translations == expected_translations)

    @pytest.mark.parametrize("model_name, variant_name, variant_options", [
        ("result_invariability", "prob_space_combination", "--prob_space_combination"),
        ("result_invariability_untrained", "prob_space_combination", "--prob_space_combination"),
        ("result_invariability", "google_options_1", "--beam_pruning_margin 1.5 --beam_score_coverage_penalty google --beam_score_coverage_penalty_strength 0.3 --beam_score_length_normalization simple --post_score_coverage_penalty google --post_score_coverage_penalty_strength 0.4 --post_score_length_normalization simple"),
        ("result_invariability_untrained", "google_options_1", "--beam_pruning_margin 1.5 --beam_score_coverage_penalty google --beam_score_coverage_penalty_strength 0.3 --beam_score_length_normalization simple --post_score_coverage_penalty google --post_score_coverage_penalty_strength 0.4 --post_score_length_normalization simple"),
        ("result_invariability", "google_options_2", "--beam_pruning_margin 1.5 --beam_score_coverage_penalty google --beam_score_coverage_penalty_strength 0.3 --beam_score_length_normalization google --beam_score_length_normalization_strength 0.25 --post_score_coverage_penalty google --post_score_coverage_penalty_strength 0.4 --post_score_length_normalization google --post_score_length_normalization_strength 0.33"),
        ("result_invariability_untrained", "google_options_2", "--beam_pruning_margin 1.5 --beam_score_coverage_penalty google --beam_score_coverage_penalty_strength 0.3 --beam_score_length_normalization google --beam_score_length_normalization_strength 0.25 --post_score_coverage_penalty google --post_score_coverage_penalty_strength 0.4 --post_score_length_normalization google --post_score_length_normalization_strength 0.33")
    ])
    def test_result_invariability_using_prob_space_combination(self, tmpdir, gpu, model_name, variant_name, variant_options):
        """
        Performs some translations with a preexisting models and compare the results
        using beam_search using different variants with previous results of the same experiment.
        The results should be identical.
        If not, it means that a recent commit have changed the behavior of the system.
        """

        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_data")
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        train_dir = os.path.join(test_data_dir, "models")
        train_prefix = os.path.join(train_dir, "{0}.train".format(model_name))
        search_eval_dir = tmpdir.mkdir("eval_beam_search")
        search_file = os.path.join(
            str(search_eval_dir),
            'translations_using_beam_search_and_{0}.txt'.format(variant_name))
        search_mode = 'beam_search'
        other_params = ' --beam_width 30 {0}'.format(variant_options)
        args_eval_search = [train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, search_file] + \
            '--mode {0}{1}'.format(search_mode, other_params).split(' ')
        if gpu is not None:
            args_eval_search += ['--gpu', gpu]
        main(arguments=["eval"] + args_eval_search)

        with open(os.path.join(str(test_data_dir), "models/{0}.translations_using_beam_search_and_{1}.txt".format(model_name, variant_name))) as f:
            expected_translations = f.readlines()
        with open(search_file) as f:
            actual_translations = f.readlines()
        print "expected_translations"
        for p in expected_translations:
            print p
        print "actual_translations"
        for p in actual_translations:
            print p

        assert(actual_translations == expected_translations)
