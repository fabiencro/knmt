#!/usr/bin/env python
"""result_invariabitility_tests.py: Check if the code gives the same results."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

from nmt_chainer.__main__ import main
import numpy as np
import os.path
import pytest
import random


class TestResultInvariability:

    @pytest.mark.parametrize("model_name, variant_name, variant_options", [
        ("result_invariability", "beam_search",
            "--mode beam_search --beam_width 30"),
        ("result_invariability_untrained", "beam_search",
            "--mode beam_search --beam_width 30"),
        ("result_invariability", "greedy_search",
            "--mode translate"),
        ("result_invariability_untrained", "greedy_search",
            "--mode translate"),
        ("result_invariability", "beam_search_and_prob_space_combination",
            "--mode beam_search --beam_width 30 "
            "--prob_space_combination"),
        ("result_invariability_untrained", "beam_search_and_prob_space_combination",
            "--mode beam_search --beam_width 30 "
            "--prob_space_combination"),
        ("result_invariability", "beam_search_and_google_options_1",
            "--mode beam_search --beam_width 30 "
            "--beam_pruning_margin 1.5 "
            "--beam_score_coverage_penalty google --beam_score_coverage_penalty_strength 0.3 "
            "--beam_score_length_normalization simple "
            "--post_score_coverage_penalty google --post_score_coverage_penalty_strength 0.4 "
            "--post_score_length_normalization simple"),
        ("result_invariability_untrained", "beam_search_and_google_options_1",
            "--mode beam_search --beam_width 30 "
            "--beam_pruning_margin 1.5 "
            "--beam_score_coverage_penalty google --beam_score_coverage_penalty_strength 0.3 "
            "--beam_score_length_normalization simple "
            "--post_score_coverage_penalty google --post_score_coverage_penalty_strength 0.4 "
            "--post_score_length_normalization simple"),
        ("result_invariability", "beam_search_and_google_options_2",
            "--mode beam_search --beam_width 30 "
            "--beam_pruning_margin 1.5 "
            "--beam_score_coverage_penalty google --beam_score_coverage_penalty_strength 0.3 "
            "--beam_score_length_normalization google --beam_score_length_normalization_strength 0.25 "
            "--post_score_coverage_penalty google --post_score_coverage_penalty_strength 0.4 "
            "--post_score_length_normalization google --post_score_length_normalization_strength 0.33"),
        ("result_invariability_untrained", "beam_search_and_google_options_2",
            "--mode beam_search --beam_width 30 "
            "--beam_pruning_margin 1.5 "
            "--beam_score_coverage_penalty google --beam_score_coverage_penalty_strength 0.3 "
            "--beam_score_length_normalization google --beam_score_length_normalization_strength 0.25 "
            "--post_score_coverage_penalty google --post_score_coverage_penalty_strength 0.4 "
            "--post_score_length_normalization google --post_score_length_normalization_strength 0.33"),
        ("result_invariability", "ensemble_search",
            "--mode beam_search --beam_width 30 "
            "--additional_training_config tests/tests_data/models/result_invariability.train.train.config "
            "--additional_trained_model tests/tests_data/models/result_invariability.train.model.best_loss.npz"),
        ("result_invariability_untrained", "ensemble_search",
            "--mode beam_search --beam_width 30 "
            "--additional_training_config tests/tests_data/models/result_invariability_untrained.train.train.config "
            "--additional_trained_model tests/tests_data/models/result_invariability_untrained.train.model.best_loss.npz"),
        ("result_invariability_with_lex_prob_dict", "beam_search",
            "--mode beam_search --beam_width 30"),
        ("result_invariability_untrained_with_lex_prob_dict", "beam_search",
            "--mode beam_search --beam_width 30"),
    ])
    def test_eval_result_invariability(self, tmpdir, gpu, model_name, variant_name, variant_options):
        """
        Performs some translations with a preexisting models and compare the results
        using different options with previous results of the same experiment.
        The results should be identical.
        If not, it means that a recent commit have changed the behavior of the system.
        """

        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tests_data")
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        train_dir = os.path.join(test_data_dir, "models")
        train_prefix = os.path.join(train_dir, "{0}.train".format(model_name))
        search_eval_dir = tmpdir.mkdir("eval")
        search_file = os.path.join(
            str(search_eval_dir),
            'translations_using_{0}.txt'.format(variant_name))
        args_eval_search = [train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, search_file] + variant_options.split(' ')
        if gpu is not None:
            args_eval_search += ['--gpu', gpu]
        main(arguments=["eval"] + args_eval_search)

        with open(os.path.join(str(test_data_dir), "models/{0}.translations_using_{1}.txt".format(model_name, variant_name))) as f:
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

    @pytest.mark.parametrize("model_name, options", [
        ("result_invariability", "--max_nb_iters 2000 --mb_size 2 --Ei 5 --Eo 12 --Hi 6 --Ha 70 --Ho 15 --Hl 12"),
        ("result_invariability_untrained", "--max_nb_iters 800 --mb_size 2 --Ei 5 --Eo 12 --Hi 6 --Ha 70 --Ho 15 --Hl 12"),
        ("result_invariability_with_lex_prob_dict", "--max_nb_iters 2000 --mb_size 2 --Ei 5 --Eo 12 --Hi 6 --Ha 70 --Ho 15 --Hl 12 --lexical_probability_dictionary tests/tests_data/lexical_prob_dict.json.gz"),
        ("result_invariability_untrained_with_lex_prob_dict", "--max_nb_iters 800 --mb_size 2 --Ei 5 --Eo 12 --Hi 6 --Ha 70 --Ho 15 --Hl 12 --lexical_probability_dictionary tests/tests_data/lexical_prob_dict.json.gz")
    ])
    def test_train_result_invariability(self, tmpdir, gpu, model_name, options):
        """
        Train some models and check if the result is the same as the expected result.
        The result should be identical.
        If not, it means that a recent commit have changed the behavior of the system.
        """

        seed = 1234
        random.seed(seed)
        np.random.seed(seed)

        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tests_data")
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        work_dir = tmpdir.mkdir("work")
        test_prefix = "{0}/tests/tests_data/models/{1}".format(str(work_dir), model_name)
        ref_prefix = "tests/tests_data/models/{0}".format(model_name)

        args_make_data = [data_src_file, data_tgt_file, test_prefix + "_test.data"] + '--dev_src {0} --dev_tgt {1}'.format(data_src_file, data_tgt_file).split(' ')
        main(arguments=["make_data"] + args_make_data)

        args_train = [test_prefix + "_test.data", test_prefix + "_test.train"] + options.split(' ')
        if gpu is not None:
            args_train += ['--gpu', gpu]
        main(arguments=["train"] + args_train)

        with np.load(test_prefix + '_test.train.model.best.npz') as test_model_data:
            with np.load(ref_prefix + '.train.model.best.npz') as ref_model_data:
                assert(len(test_model_data.keys()) == len(ref_model_data.keys()))
                for test_key, test_value in test_model_data.iteritems():
                    np.testing.assert_array_almost_equal(test_value, ref_model_data[test_key], 5)
