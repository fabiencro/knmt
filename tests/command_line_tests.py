import os.path
import pytest

from nmt_chainer.__main__ import main


class TestCommandLine:
    """
    Some simple tests checking that no command line option lead to an error
    """

    def test_make_data(self, tmpdir, gpu):
        test_data_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "tests_data")
        train_dir = tmpdir.mkdir("train")
        data_prefix = str(train_dir.join("test1.data"))
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        args = 'make_data {0} {1} {2} --dev_src {0} --dev_tgt {1} --src_voc_size 512 --tgt_voc_size 3 --max_nb_ex 3 --test_src {0} --test_tgt {1} --tgt_segmentation_type char --src_segmentation_type word'.format(
            data_src_file, data_tgt_file, data_prefix).split(' ')
        main(arguments=args)

    def test_train(self, tmpdir, gpu):
        test_data_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "tests_data")
        train_dir = tmpdir.mkdir("train")
        data_prefix = str(train_dir.join("test1.data"))
        train_prefix = str(train_dir.join("test1.train"))
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        args = 'make_data {0} {1} {2} --dev_src {0} --dev_tgt {1}'.format(
            data_src_file, data_tgt_file, data_prefix).split(' ')
        main(arguments=args)

        args_train = ["train"] + [data_prefix, train_prefix] +\
            "--lexicon_prob_epsilon 0.5 --max_nb_iters 3 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23".split(" ") +\
            "--encoder_cell_type lstm --decoder_cell_type lstm --nb_batch_to_sort 2 --noise_on_prev_word --l2_gradient_clipping 1".split(" ") +\
            "--weight_decay 0.001 --optimizer momentum --learning_rate 0.23 --momentum 0.56 --randomized_data".split(" ") +\
            "--no_shuffle_of_training_data --max_src_tgt_length 56 --report_every 34 --sample_every 45".split(" ") +\
            "--sample_every 45 --save_ckpt_every 56".split(" ")

        if gpu is not None:
            args_train += ['--gpu', gpu]
        main(arguments=args_train)

    def test_eval(self, tmpdir, gpu):
        test_data_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "tests_data")
        train_dir = tmpdir.mkdir("train")
        data_prefix = str(train_dir.join("test1.data"))
        train_prefix = str(train_dir.join("test1.train"))
        data_src_file = os.path.join(test_data_dir, "src2.txt")
        data_tgt_file = os.path.join(test_data_dir, "tgt2.txt")
        args = 'make_data {0} {1} {2} --dev_src {0} --dev_tgt {1}'.format(
            data_src_file, data_tgt_file, data_prefix).split(' ')
        main(arguments=args)

        args_train = ["train"] + [data_prefix, train_prefix] + "--max_nb_iters 6 --report_every 2 --mb_size 2 --Ei 10 --Eo 12 --Hi 30 --Ha 70 --Ho 15 --Hl 23".split(" ")
        if gpu is not None:
            args_train += ['--gpu', gpu]
        main(arguments=args_train)

        eval_dir = tmpdir.mkdir("eval")
        translation_file = os.path.join(str(eval_dir), 'translations.txt')
        args_eval = ["eval", train_prefix + '.train.config', train_prefix + '.model.best.npz', data_src_file, translation_file] +\
            '--mode beam_search --beam_width 30'.split(' ') +\
            ["--additional_training_config", train_prefix + '.train.config', "--additional_trained_model", train_prefix + '.model.best_loss.npz'] +\
            ["--tgt_fn", data_tgt_file, "--ref", data_tgt_file] + "--max_nb_ex 3 --mb_size 1 --beam_pruning_margin 10".split(" ") +\
            "--nb_steps 23 --nb_steps_ratio 2.8 --nb_batch_to_sort 2 --prob_space_combination".split(" ")
        if gpu is not None:
            args_eval += ['--gpu', gpu]
        main(arguments=args_eval)
