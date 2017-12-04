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
from nmt_chainer.__main__ import main
import os.path
import pytest


class TestTrainingManagement:

    def test_checkpoint_saving(self, tmpdir, gpu):
        """
        Test no error happens during checkpoint saving.
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

        args_train = ["train", data_prefix, train_prefix] + "--max_nb_iters 10 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23 --save_ckpt_every 5".split(" ")
        if gpu is not None:
            args_train += ['--gpu', gpu]
        main(arguments=args_train)

    def test_config_saving(self, tmpdir, gpu):
        """
        Test no error happens during checkpoint saving.
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

        args_train = ["train", data_prefix, train_prefix] + "--max_nb_iters 5 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23".split(" ")
        if gpu is not None:
            args_train += ['--gpu', gpu]
        main(arguments=args_train)

        config_filename = train_prefix + ".train.config"

        train_prefix_2 = train_prefix + ".2"
        args_train = ["train", "--config", config_filename, "--save_prefix", train_prefix_2]

        if gpu is not None:
            args_train += ['--gpu', gpu]
        main(arguments=args_train)

        config_filename2 = train_prefix_2 + ".train.config"

        import json
        config1 = json.load(open(config_filename))
        config2 = json.load(open(config_filename2))

        def compare_dict_except(d1, d2, except_fields=None):
            k_list_1 = set(d1.keys())
            k_list_2 = set(d2.keys())
            k_xor = (k_list_1 - k_list_2) | (k_list_2 - k_list_1)
            for k_diff in k_xor:
                if except_fields is None or k_diff not in except_fields:
                    return False
            for k in k_list_1 & k_list_2:
                v1 = d1[k]
                if isinstance(v1, dict):
                    compare_result = compare_dict_except(d1[k], d2[k], except_fields=except_fields)
                    if not compare_result:
                        return False
                else:
                    if v1 != d2[k] and (
                            except_fields is None or k not in except_fields):
                        return False
            return True

        assert compare_dict_except(config1, config2, except_fields="metadata save_prefix config".split())
