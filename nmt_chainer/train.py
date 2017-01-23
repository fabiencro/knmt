#!/usr/bin/env python
"""train.py: Train a RNNSearch Model"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import chainer
from chainer import cuda, optimizers, serializers
import chainer.function_hooks
import operator

import models
from training import train_on_data
from make_data import Indexer
import versioning_tools
from collections import OrderedDict

import logging
import json
import os.path
import gzip
# import h5py

from utils import ensure_path
# , make_batch_src_tgt, make_batch_src, minibatch_provider, compute_bleu_with_unk_as_wrong,de_batch
# from evaluation import (
#                   greedy_batch_translate, convert_idx_to_string, 
#                   compute_loss_all, translate_to_file, sample_once)



import time

import numpy

from chainer import cuda
from chainer import function
from collections import defaultdict

import rnn_cells

def function_namer(function, in_data):
    in_shapes = []
    for in_elem in in_data:
        if hasattr(in_elem, "shape"):
            in_shapes.append("@%r"%(in_elem.shape,))
        elif isinstance(in_elem, (int, float, str)):
            in_shapes.append(in_elem)
        else:
            in_shapes.append("OTHER")
    if isinstance(function, chainer.functions.array.split_axis.SplitAxis):
        in_shapes.append(("s/a", function.indices_or_sections, function.axis))
        
    in_shapes = tuple(in_shapes)
    return (function.__class__, in_shapes)
    
class TimerElem(object):
    def __init__(self):
        self.fwd = 0
        self.bwd = 0
        self.total = 0
        self.nb_fwd = 0
        self.nb_bwd = 0
    def add_fwd(self, fwd):
        self.fwd += fwd
        self.total += fwd
        self.nb_fwd += 1
    def add_bwd(self, bwd):
        self.bwd += bwd
        self.total += bwd
        self.nb_bwd += 1
    def __repr__(self):
        return "<T:%f F[%i]:%f B[%i]:%f>"%(self.total, self.nb_fwd, self.fwd, self.nb_bwd, self.bwd)
    __str__ = __repr__
    
    
class MyTimerHook(function.FunctionHook):
    """Function hook for measuring elapsed time of functions.

    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the function that calls this hook and the elapsed time
            the function consumes.
    """

    name = 'TimerHook'

    def __init__(self):
        self.call_times_per_classes = defaultdict(TimerElem)

    def _preprocess(self):
        if self.xp == numpy:
            self.start = time.time()
        else:
            self.start = cuda.Event()
            self.stop = cuda.Event()
            self.start.record()

    def forward_preprocess(self, function, in_data):
        self.xp = cuda.get_array_module(*in_data)
        self._preprocess()

    def backward_preprocess(self, function, in_data, out_grad):
        self.xp = cuda.get_array_module(*(in_data + out_grad))
        self._preprocess()

    def _postprocess(self, function_repr, bwd = False):
        if self.xp == numpy:
            self.stop = time.time()
            elapsed_time = self.stop - self.start
        else:
            self.stop.record()
            self.stop.synchronize()
            # Note that `get_elapsed_time` returns result in milliseconds
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                self.start, self.stop) / 1000.0
        if bwd:
            self.call_times_per_classes[function_repr].add_bwd(elapsed_time)
        else:
            self.call_times_per_classes[function_repr].add_fwd(elapsed_time)
#         self.call_history.append((function, elapsed_time))

    def forward_postprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        assert xp == self.xp
        self._postprocess(function_namer(function, in_data))

    def backward_postprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        assert xp == self.xp
        self._postprocess(function_namer(function, in_data), bwd = True)

    def total_time(self):
        """Returns total elapsed time in seconds."""
        return sum(t.total for (_, t) in self.call_times_per_classes.iteritems())

    def print_sorted(self):
        for name, time in sorted(self.call_times_per_classes.items(), key = lambda x:x[1].total):
            print name, time
            
logging.basicConfig()
log = logging.getLogger("rnns:train")
log.setLevel(logging.INFO)

def define_parser(parser):
    parser.add_argument("data_prefix", help = "prefix of the training data created by make_data.py")
    parser.add_argument("save_prefix", help = "prefix to be added to all files created during the training")
    
    
    model_description_group = parser.add_argument_group("Model Description")
    model_description_group.add_argument("--Ei", type = int, default= 640, help = "Source words embedding size.")
    model_description_group.add_argument("--Eo", type = int, default= 640, help = "Target words embedding size.")
    model_description_group.add_argument("--Hi", type = int, default= 1024, help = "Source encoding layer size.")
    model_description_group.add_argument("--Ho", type = int, default= 1024, help = "Target hidden layer size.")
    model_description_group.add_argument("--Ha", type = int, default= 1024, help = "Attention Module Hidden layer size.")
    model_description_group.add_argument("--Hl", type = int, default= 512, help = "Maxout output size.")
    model_description_group.add_argument("--lexical_probability_dictionary", help = "lexical translation probabilities in zipped JSON format. Used to implement https://arxiv.org/abs/1606.02006")
    model_description_group.add_argument("--lexicon_prob_epsilon", default = 1e-3, type = float, help = "epsilon value for combining the lexical probabilities")
    model_description_group.add_argument("--encoder_cell_type", default = "lstm", help = "cell type of encoder. format: type,param1:val1,param2:val2,... where type is in [%s]"%(" ".join(rnn_cells.cell_dict.keys())))
    model_description_group.add_argument("--decoder_cell_type", default = "lstm", help = "cell type of decoder. format same as for encoder")
    model_description_group.add_argument("--use_deep_attn", default = False, action = "store_true")
    model_description_group.add_argument("--use_accumulated_attn", default = False, action = "store_true")

    training_paramenters_group = parser.add_argument_group("Training Parameters")
    training_paramenters_group.add_argument("--mb_size", type = int, default= 64, help = "Minibatch size")
    training_paramenters_group.add_argument("--nb_batch_to_sort", type = int, default= 20, help = "Sort this many batches by size.")
    training_paramenters_group.add_argument("--noise_on_prev_word", default = False, action = "store_true") 
    training_paramenters_group.add_argument("--l2_gradient_clipping", type = float, default = 1, help = "L2 gradient clipping. 0 for None")
    training_paramenters_group.add_argument("--hard_gradient_clipping", type = float, nargs = 2, help = "hard gradient clipping.")
    training_paramenters_group.add_argument("--weight_decay", type = float, help = "Weight decay value. ")
    training_paramenters_group.add_argument("--optimizer", choices=["sgd", "rmsprop", "rmspropgraves", 
                            "momentum", "nesterov", "adam", "adagrad", "adadelta"], 
                        default = "adam", help = "Optimizer type.")
    training_paramenters_group.add_argument("--learning_rate", type = float, default= 0.01, help = "Learning Rate")
    training_paramenters_group.add_argument("--momentum", type = float, default= 0.9, help = "Momentum term")
    training_paramenters_group.add_argument("--randomized_data", default = False, action = "store_true")
    training_paramenters_group.add_argument("--no_shuffle_of_training_data", default = False, action = "store_true")
    training_paramenters_group.add_argument("--use_reinf", default = False, action = "store_true")
    training_paramenters_group.add_argument("--use_bn_length", default = 0, type = int)
    training_paramenters_group.add_argument("--use_previous_prediction", default = 0, type = float)
    training_paramenters_group.add_argument("--curiculum_training", default = False, action = "store_true")
    training_paramenters_group.add_argument("--reverse_src", default = False, action = "store_true")
    training_paramenters_group.add_argument("--reverse_tgt", default = False, action = "store_true")
    
    training_monitoring_group = parser.add_argument_group("Training Management and Monitoring")
    training_monitoring_group.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
    training_monitoring_group.add_argument("--load_model", help = "load the parameters of a previously trained model")
    training_monitoring_group.add_argument("--load_optimizer_state", help = "load previously saved optimizer states")
    training_monitoring_group.add_argument("--load_trainer_snapshot", help = "load previously saved trainer states")
    training_monitoring_group.add_argument("--use_memory_optimization", default = False, action = "store_true",
                        help = "Experimental option that could strongly reduce memory used.")
    training_monitoring_group.add_argument("--max_nb_iters", type = int, default= None, help = "maximum number of iterations")
    training_monitoring_group.add_argument("--max_nb_epochs", type = int, default= None, help = "maximum number of epochs")
    training_monitoring_group.add_argument("--max_src_tgt_length", type = int, help = "Limit length of training sentences")
    training_monitoring_group.add_argument("--report_every", type = int, default = 200, help = "report every x iterations")
    training_monitoring_group.add_argument("--no_resume", default = False, action = "store_true")
    training_monitoring_group.add_argument("--init_orth", default = False, action = "store_true")
    training_monitoring_group.add_argument("--no_report_or_save", default = False, action = "store_true")
    training_monitoring_group.add_argument("--sample_every", default = 200, type = int)
    training_monitoring_group.add_argument("--save_ckpt_every", default = 4000, type = int)
    training_monitoring_group.add_argument("--save_initial_model_to", help = "save the initial model parameters to given file in npz format")
    
    
def command_line(arguments = None):
    import argparse
    parser = argparse.ArgumentParser(description= "Train a RNNSearch model", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    define_parser(parser)
    args = parser.parse_args(args = arguments)
    
    do_train(args)
    
def do_train(args):
    
    output_files_dict = {}
    output_files_dict["train_config"] = args.save_prefix + ".train.config"
    output_files_dict["model_ckpt"] = args.save_prefix + ".model." + "ckpt" + ".npz"
    output_files_dict["model_final"] = args.save_prefix + ".model." + "final" + ".npz"
    output_files_dict["model_best"] = args.save_prefix + ".model." + "best" + ".npz"
    output_files_dict["model_best_loss"] = args.save_prefix + ".model." + "best_loss" + ".npz"
    
    output_files_dict["test_translation_output"] = args.save_prefix + ".test.out"
    output_files_dict["test_src_output"] = args.save_prefix + ".test.src.out"
    output_files_dict["dev_translation_output"] = args.save_prefix + ".dev.out"
    output_files_dict["dev_src_output"] = args.save_prefix + ".dev.src.out"
    output_files_dict["valid_translation_output"] = args.save_prefix + ".valid.out"
    output_files_dict["valid_src_output"] = args.save_prefix + ".valid.src.out"
    output_files_dict["sqlite_db"] = args.save_prefix + ".result.sqlite"
    output_files_dict["optimizer_ckpt"] = args.save_prefix + ".optimizer." + "ckpt" + ".npz"
    output_files_dict["optimizer_final"] = args.save_prefix + ".optimizer." + "final" + ".npz"
    
    
    
    save_prefix_dir, save_prefix_fn = os.path.split(args.save_prefix)
    ensure_path(save_prefix_dir)
    
    already_existing_files = []
    for key_info, filename in output_files_dict.iteritems():#, valid_data_fn]:
        if os.path.exists(filename):
            already_existing_files.append(filename)
    if len(already_existing_files) > 0:
        print "Warning: existing files are going to be replaced / updated: ",  already_existing_files
        raw_input("Press Enter to Continue")
    
    
    config_fn = args.data_prefix + ".data.config"
    voc_fn = args.data_prefix + ".voc"
    data_fn = args.data_prefix + ".data.json.gz"
    
    log.info("loading training data from %s"% data_fn)
    training_data_all = json.load(gzip.open(data_fn, "rb"))
    
    training_data = training_data_all["train"]
    
    log.info("loaded %i sentences as training data" % len(training_data))
    
    if "test" in  training_data_all:
        test_data = training_data_all["test"]
        log.info("Found test data: %i sentences" % len(test_data))
    else:
        test_data = None
        log.info("No test data found")
    
    if "dev" in  training_data_all:
        dev_data = training_data_all["dev"]
        log.info("Found dev data: %i sentences" % len(dev_data))
    else:
        dev_data = None
        log.info("No dev data found")
        
    if "valid" in  training_data_all:
        valid_data = training_data_all["valid"]
        log.info("Found valid data: %i sentences" % len(valid_data))
    else:
        valid_data = None
        log.info("No valid data found")
        
    log.info("loading voc from %s"% voc_fn)
    src_voc, tgt_voc = json.load(open(voc_fn))
    
    src_indexer = Indexer.make_from_serializable(src_voc)
    tgt_indexer = Indexer.make_from_serializable(tgt_voc)
    tgt_voc = None
    src_voc = None
    
    
#     Vi = len(src_voc) + 1 # + UNK
#     Vo = len(tgt_voc) + 1 # + UNK
    
    Vi = len(src_indexer) # + UNK
    Vo = len(tgt_indexer) # + UNK
    
    if args.lexical_probability_dictionary is not None:
        log.info("opening lexical_probability_dictionary %s" % args.lexical_probability_dictionary)
        lexical_probability_dictionary_all = json.load(gzip.open(args.lexical_probability_dictionary, "rb"))
        log.info("computing lexical_probability_dictionary_indexed")
        lexical_probability_dictionary_indexed = {}
        for ws in lexical_probability_dictionary_all:
            ws_idx = src_indexer.convert([ws])[0]
            if ws_idx in lexical_probability_dictionary_indexed:
                assert src_indexer.is_unk_idx(ws_idx)
            else:
                lexical_probability_dictionary_indexed[ws_idx] = {}
            for wt in lexical_probability_dictionary_all[ws]:
                wt_idx = tgt_indexer.convert([wt])[0]
                if wt_idx in lexical_probability_dictionary_indexed[ws_idx]:
                    assert src_indexer.is_unk_idx(ws_idx) or tgt_indexer.is_unk_idx(wt_idx)
                    lexical_probability_dictionary_indexed[ws_idx][wt_idx] += lexical_probability_dictionary_all[ws][wt]
                else:
                    lexical_probability_dictionary_indexed[ws_idx][wt_idx] = lexical_probability_dictionary_all[ws][wt]
        lexical_probability_dictionary = lexical_probability_dictionary_indexed
    else:
        lexical_probability_dictionary = None
    
    
    if args.max_src_tgt_length is not None:
        log.info("filtering sentences of length larger than %i"%(args.max_src_tgt_length))
        filtered_training_data = []
        nb_filtered = 0
        for src, tgt in training_data:
            if len(src) <= args.max_src_tgt_length and len(tgt) <= args.max_src_tgt_length:
                filtered_training_data.append((src, tgt))
            else:
                nb_filtered += 1
        log.info("filtered %i sentences of length larger than %i"%(nb_filtered, args.max_src_tgt_length))
        training_data = filtered_training_data
    
    if not args.no_shuffle_of_training_data:
        log.info("shuffling")
        import random
        random.shuffle(training_data)
        log.info("done")
    
#     
#     Vi = len(src_voc) + 1 # + UNK
#     Vo = len(tgt_voc) + 1 # + UNK
    
#     config_training = {"command_line" : args.__dict__, "Vi": Vi, "Vo" : Vo, "voc" : voc_fn, "data" : data_fn}
    
    config_training = OrderedDict() # using ordered for improved readability of json
    config_training["command_line"] = args.__dict__
    config_training["Vi"] = Vi
    config_training["Vo"] = Vo
    config_training["voc"] = voc_fn
    config_training["data"] = data_fn
    config_training["knmt_version"] = versioning_tools.get_version_dict()
    
    save_train_config_fn = output_files_dict["train_config"]
    log.info("Saving training config to %s" % save_train_config_fn)
    json.dump(config_training, open(save_train_config_fn, "w"), indent=2, separators=(',', ': '))
    
    eos_idx = Vo
    
    # Selecting Attention type
    attn_cls = models.AttentionModule
    if args.use_accumulated_attn:
        raise NotImplemented
#         encdec = models.EncoderDecoder(Vi, args.Ei, args.Hi, Vo + 1, args.Eo, args.Ho, args.Ha, args.Hl,
#                                        attn_cls= models.AttentionModuleAcumulated,
#                                        init_orth = args.init_orth)
    if args.use_deep_attn:
        attn_cls = models.DeepAttentionModule
    
    
    # Creating encoder/decoder
    encdec = models.EncoderDecoder(Vi, args.Ei, args.Hi, Vo + 1, args.Eo, args.Ho, args.Ha, args.Hl,
                                       init_orth = args.init_orth, use_bn_length = args.use_bn_length,
                                       attn_cls = attn_cls,
                                       encoder_cell_type = rnn_cells.create_cell_model_from_string(args.encoder_cell_type),
                                       decoder_cell_type = rnn_cells.create_cell_model_from_string(args.decoder_cell_type),
                                       lexical_probability_dictionary = lexical_probability_dictionary, 
                                       lex_epsilon = args.lexicon_prob_epsilon)
    
    if args.load_model is not None:
        serializers.load_npz(args.load_model, encdec)
    
    if args.gpu is not None:
        encdec = encdec.to_gpu(args.gpu)
    
    if args.optimizer == "adadelta":
        optimizer = optimizers.AdaDelta()
    elif args.optimizer == "adam":
        optimizer = optimizers.Adam()
    elif args.optimizer == "adagrad":
        optimizer = optimizers.AdaGrad(lr = args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = optimizers.SGD(lr = args.learning_rate)
    elif args.optimizer == "momentum":
        optimizer = optimizers.MomentumSGD(lr = args.learning_rate,
                                           momentum = args.momentum)
    elif args.optimizer == "nesterov":
        optimizer = optimizers.NesterovAG(lr = args.learning_rate,
                                           momentum = args.momentum)
    elif args.optimizer == "rmsprop":
        optimizer = optimizers.RMSprop(lr = args.learning_rate)
    elif args.optimizer == "rmspropgraves":
        optimizer = optimizers.RMSpropGraves(lr = args.learning_rate,
                                           momentum = args.momentum)    
    else:
        raise NotImplemented
    with cuda.get_device(args.gpu):
        optimizer.setup(encdec)
    
    if args.l2_gradient_clipping is not None and args.l2_gradient_clipping > 0:
        optimizer.add_hook(chainer.optimizer.GradientClipping(args.l2_gradient_clipping))

    if args.hard_gradient_clipping is not None and args.hard_gradient_clipping > 0:
        optimizer.add_hook(chainer.optimizer.GradientHardClipping(*args.hard_gradient_clipping))

    if args.weight_decay is not None:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    if args.load_optimizer_state is not None:
        with cuda.get_device(args.gpu):
            serializers.load_npz(args.load_optimizer_state, optimizer)    
    

    import training_chainer
    with cuda.get_device(args.gpu):
        if args.max_nb_iters is not None:
            stop_trigger = (args.max_nb_iters, "iteration")
            if args.max_nb_epochs is not None:
                log.warn("max_nb_iters and max_nb_epochs both specified. Only max_nb_iters will be considered.")
        elif args.max_nb_epochs is not None:
            stop_trigger = (args.max_nb_epochs, "epoch")
        else:
            stop_trigger = None
        training_chainer.train_on_data_chainer(encdec, optimizer, training_data, output_files_dict,
                      src_indexer, tgt_indexer, eos_idx = eos_idx, 
                      output_dir = args.save_prefix,
                      stop_trigger = stop_trigger,
                      mb_size = args.mb_size,
                      nb_of_batch_to_sort = args.nb_batch_to_sort,
                      test_data = test_data, dev_data = dev_data, valid_data = valid_data, gpu = args.gpu, report_every = args.report_every,
                      randomized = args.randomized_data, reverse_src = args.reverse_src, reverse_tgt = args.reverse_tgt,
                      do_not_save_data_for_resuming = args.no_resume,
                      noise_on_prev_word = args.noise_on_prev_word, curiculum_training = args.curiculum_training,
                      use_previous_prediction = args.use_previous_prediction, no_report_or_save = args.no_report_or_save,
                      use_memory_optimization = args.use_memory_optimization,
                      sample_every = args.sample_every,
                      use_reinf = args.use_reinf,
                      save_ckpt_every = args.save_ckpt_every,
                      trainer_snapshot = args.load_trainer_snapshot,
#                     lexical_probability_dictionary = lexical_probability_dictionary,
#                     V_tgt = Vo + 1,
#                     lexicon_prob_epsilon = args.lexicon_prob_epsilon
                      save_initial_model_to = args.save_initial_model_to
                      )

# 
#     import sys
#     sys.exit(0)
#     with cuda.get_device(args.gpu):
# #         with MyTimerHook() as timer:
# #             try:
#                 train_on_data(encdec, optimizer, training_data, output_files_dict,
#                       src_indexer, tgt_indexer, eos_idx = eos_idx, 
#                       mb_size = args.mb_size,
#                       nb_of_batch_to_sort = args.nb_batch_to_sort,
#                       test_data = test_data, dev_data = dev_data, valid_data = valid_data, gpu = args.gpu, report_every = args.report_every,
#                       randomized = args.randomized_data, reverse_src = args.reverse_src, reverse_tgt = args.reverse_tgt,
#                       max_nb_iters = args.max_nb_iters, do_not_save_data_for_resuming = args.no_resume,
#                       noise_on_prev_word = args.noise_on_prev_word, curiculum_training = args.curiculum_training,
#                       use_previous_prediction = args.use_previous_prediction, no_report_or_save = args.no_report_or_save,
#                       use_memory_optimization = args.use_memory_optimization,
#                       sample_every = args.sample_every,
#                       use_reinf = args.use_reinf,
#                       save_ckpt_every = args.save_ckpt_every
# #                     lexical_probability_dictionary = lexical_probability_dictionary,
# #                     V_tgt = Vo + 1,
# #                     lexicon_prob_epsilon = args.lexicon_prob_epsilon
#                       )
# #             finally:
# #                 print timer
# #                 timer.print_sorted()
# #                 print "total time:"
# #                 print(timer.total_time())
#                 
#                 

if __name__ == '__main__':
    command_line()
