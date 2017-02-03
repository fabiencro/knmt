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
import argument_parsing_tools
import logging
import json
import os.path
import gzip
import sys
import pprint
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


_CONFIG_SECTION_TO_DESCRIPTION = {"model": "Model Description",
  "training": "Training Parameters",
  "training_management": "Training Management and Monitoring"}


def define_parser(parser):
    parser.add_argument("data_prefix", nargs = "?", 
                        action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                        help = "prefix of the training data created by make_data.py")
    parser.add_argument("save_prefix", nargs = "?", 
                        action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                        help = "prefix to be added to all files created during the training")
    
    
    model_description_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["model"])
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
    model_description_group.add_argument("--init_orth", default = False, action = "store_true")
    model_description_group.add_argument("--use_bn_length", default = 0, type = int)

    training_paramenters_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["training"])
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
    training_paramenters_group.add_argument("--use_previous_prediction", default = 0, type = float)
    training_paramenters_group.add_argument("--curiculum_training", default = False, action = "store_true")
    training_paramenters_group.add_argument("--reverse_src", default = False, action = "store_true")
    training_paramenters_group.add_argument("--reverse_tgt", default = False, action = "store_true")
    
    training_monitoring_group = parser.add_argument_group(_CONFIG_SECTION_TO_DESCRIPTION["training_management"])
    training_monitoring_group.add_argument("--config", help = "load a training config file")
    training_monitoring_group.add_argument("--data_prefix", dest = "data_prefix", 
                                           action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                                           help = "same as positional argument --data_prefix")
    training_monitoring_group.add_argument("--save_prefix", dest = "save_prefix",
                                           action = argument_parsing_tools.ArgumentActionNotOverwriteWithNone,
                                           help = "same as positional argument --save_prefix")
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
    training_monitoring_group.add_argument("--no_report_or_save", default = False, action = "store_true")
    training_monitoring_group.add_argument("--sample_every", default = 200, type = int)
    training_monitoring_group.add_argument("--save_ckpt_every", default = 4000, type = int)
    training_monitoring_group.add_argument("--save_initial_model_to", help = "save the initial model parameters to given file in npz format")
    training_monitoring_group.add_argument("--reshuffle_every_epoch", default = False, action = "store_true", help = "reshuffle training data at the end of each epoch")
    training_monitoring_group.add_argument("--resume", default = False, action = "store_true", help = "resume training from checkpoint config")
#     training_monitoring_group.add_argument("--resume", help = "resume training from checkpoint config")
    
    
#     
# def load_training_config_file(filename):
#     file_content = json.load(open(filename))
    
def get_parse_option_orderer():
    description_to_config_section = dict( (v, k) for (k,v) in _CONFIG_SECTION_TO_DESCRIPTION.iteritems())
    por = argument_parsing_tools.ParseOptionRecorder(group_title_to_section = description_to_config_section,
                                                     ignore_positional_arguments = set(["save_prefix", "data_prefix"]))
    define_parser(por)
    return por
    
def load_voc_and_make_training_config(args):
    config_base = None
    if args.config is not None:
        log.info("loading training config file %s", args.config)
        config_base = load_config_train(args.config, readonly = False)
    
    parse_option_orderer = get_parse_option_orderer()
    config_training = parse_option_orderer.convert_args_to_ordered_dict(args)
    
    if config_base is not None:
        pwndan = argument_parsing_tools.ParserWithNoneDefaultAndNoGroup()
        define_parser(pwndan)
        args_given_set = pwndan.get_args_given(sys.argv)
        for argname in set(args_given_set):
            if getattr(args, argname) is None:
                args_given_set.remove(argname)
                
        print "args_given_set", args_given_set
        config_base.update_recursive(config_training, valid_keys = args_given_set) 
        config_training = config_base
    else:
        assert "data" not in config_training
        assert "metadata" not in config_training
        
#     config_data_fn = config_training["data_prefix"] + ".data.config"

#     print "yyyy"
#     config_training.pretty_print()
#     print "xxxxx"
    if config_training["training_management"]["data_prefix"] is None or config_training["training_management"]["save_prefix"] is None:
        raise CommandLineValuesException("save_prefix and data_prefix need to be set either on the command line or in a config file")
    
    data_prefix = config_training["training_management"]["data_prefix"]
    voc_fn = data_prefix + ".voc"
    data_fn = data_prefix + ".data.json.gz"
    
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
    
    config_training["data"] = argument_parsing_tools.OrderedNamespace()
    config_training["data"]["data_fn"] = data_fn
    config_training["data"]["Vi"] = Vi
    config_training["data"]["Vo"] = Vo
    config_training["data"]["voc"] = voc_fn
    
    config_training.add_metadata_infos(version_num = 1)
    
    config_training.set_readonly()
    
    return config_training, src_indexer, tgt_indexer
    
def load_config_train(filename, readonly = True):
    config_as_ordered_dict = json.load(open(filename), object_pairs_hook=OrderedDict)
    if "metadata" not in config_as_ordered_dict: # older config file
        parse_option_orderer = get_parse_option_orderer()
        config_training = parse_option_orderer.convert_args_to_ordered_dict(config_as_ordered_dict["command_line"], args_is_namespace = False)
        
        assert "data" not in config_training
        config_training["data"] = argument_parsing_tools.OrderedNamespace()
        config_training["data"]["data_fn"] = config_as_ordered_dict["data"]
        config_training["data"]["Vi"] = config_as_ordered_dict["Vi"]
        config_training["data"]["Vo"] = config_as_ordered_dict["Vo"]
        config_training["data"]["voc"] = config_as_ordered_dict["voc"]

        assert "metadata" not in config_training
        config_training["metadata"] = argument_parsing_tools.OrderedNamespace()
        config_training["metadata"]["config_version_num"] = 0.9
        config_training["metadata"]["command_line"] = None
        config_training["metadata"]["knmt_version"] = None     
    elif config_as_ordered_dict["metadata"]["config_version_num"] == 1.0:
        argument_parsing_tools.OrderedNamespace.convert_to_ordered_namespace(config_as_ordered_dict)
        config_training = config_as_ordered_dict
    else:
        raise ValueError("The config version of %s is not supported by this version of the program" % filename)
    
    if readonly:
        config_training.set_readonly()
    return config_training
    
def command_line(arguments = None):
    import argparse
    parser = argparse.ArgumentParser(description= "Train a RNNSearch model", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    define_parser(parser)
    args = parser.parse_args(args = arguments)
    
    do_train(args)
    
def generate_lexical_probability_dictionary_indexed(lexical_probability_dictionary_all, src_indexer, tgt_indexer):
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
    return lexical_probability_dictionary_indexed 

def create_encdec_from_config_dict(config_dict, src_indexer, tgt_indexer):
    Ei = config_dict["Ei"]
    Hi = config_dict["Hi"]
    Eo = config_dict["Eo"]
    Ho = config_dict["Ho"]
    Ha = config_dict["Ha"]
    Hl = config_dict["Hl"]
    
    Vi = len(src_indexer) # + UNK
    Vo = len(tgt_indexer) # + UNK
    
    encoder_cell_type = config_dict.get("encoder_cell_type", "gru")
    decoder_cell_type = config_dict.get("decoder_cell_type", "gru")
    
    use_bn_length = config_dict.get("use_bn_length", None)
    
    # Selecting Attention type
    attn_cls = models.AttentionModule
    if config_dict.get("use_accumulated_attn", False):
        raise NotImplemented
    if config_dict.get("use_deep_attn", False):
        attn_cls = models.DeepAttentionModule
    
    init_orth = config_dict.get("init_orth", False)
    
    if "lexical_probability_dictionary" in config_dict and config_dict["lexical_probability_dictionary"] is not None:
        log.info("opening lexical_probability_dictionary %s" % config_dict["lexical_probability_dictionary"])
        lexical_probability_dictionary_all = json.load(gzip.open(config_dict["lexical_probability_dictionary"], "rb"))
        
        lexical_probability_dictionary = generate_lexical_probability_dictionary_indexed(
            lexical_probability_dictionary_all, src_indexer, tgt_indexer)
    else:
        lexical_probability_dictionary = None
    lex_epsilon = config_dict.get("lexicon_prob_epsilon", 0.001)
    
    # Creating encoder/decoder
    encdec = models.EncoderDecoder(Vi, Ei, Hi, Vo + 1, Eo, Ho, Ha, Hl, use_bn_length = use_bn_length,
                                   attn_cls = attn_cls,
                                   init_orth = init_orth,
                                   encoder_cell_type = rnn_cells.create_cell_model_from_string(encoder_cell_type),
                                    decoder_cell_type = rnn_cells.create_cell_model_from_string(decoder_cell_type),
                                    lexical_probability_dictionary = lexical_probability_dictionary,
                                    lex_epsilon = lex_epsilon)

    return encdec



def create_encdec_and_indexers_from_config_dict(config_dict, src_indexer = None, tgt_indexer = None, load_config_model = "no"):
    assert load_config_model in "yes no if_exists".split()
    
    if src_indexer is None or tgt_indexer is None:
        voc_fn = config_dict.data["voc"]
        log.info("loading voc from %s"% voc_fn)
        src_voc, tgt_voc = json.load(open(voc_fn))
    
    if src_indexer is None:
        src_indexer = Indexer.make_from_serializable(src_voc)
        
    if tgt_indexer is None:
        tgt_indexer = Indexer.make_from_serializable(tgt_voc)
        
    tgt_voc = None
    src_voc = None

    encdec = create_encdec_from_config_dict(config_dict["model"], src_indexer, tgt_indexer)
    
    eos_idx = len(tgt_indexer)
    
    
    if load_config_model != "no":
        if "model_parameters" not in config_dict:
            if load_config_model == "yes":
                log.error("cannot find model parameters in config file")
                raise ValueError("Config file do not contain model_parameters section")
        else:
            if config_dict.model_parameters.type == "model":
                model_filename = config_dict.model_parameters.filename
                log.info("loading model parameters from file specified by config file:%s" % model_filename)
                serializers.load_npz(model_filename, encdec)
            else:
                if load_config_model == "yes":
                    log.error("model parameters in config file is of type snapshot, not model")
                    raise ValueError("Config file model is not of type model")
    
    return encdec, eos_idx, src_indexer, tgt_indexer
    
    
class CommandLineValuesException(Exception):
    pass

def do_train(args):
    
    config_training, src_indexer, tgt_indexer = load_voc_and_make_training_config(args)
    
    save_prefix = config_training.training_management.save_prefix

    output_files_dict = {}
    output_files_dict["train_config"] = save_prefix + ".train.config"
    output_files_dict["model_ckpt"] = save_prefix + ".model." + "ckpt" + ".npz"
    output_files_dict["model_final"] = save_prefix + ".model." + "final" + ".npz"
    output_files_dict["model_best"] = save_prefix + ".model." + "best" + ".npz"
    output_files_dict["model_best_loss"] = save_prefix + ".model." + "best_loss" + ".npz"
    
    output_files_dict["test_translation_output"] = save_prefix + ".test.out"
    output_files_dict["test_src_output"] = save_prefix + ".test.src.out"
    output_files_dict["dev_translation_output"] = save_prefix + ".dev.out"
    output_files_dict["dev_src_output"] = save_prefix + ".dev.src.out"
    output_files_dict["valid_translation_output"] = save_prefix + ".valid.out"
    output_files_dict["valid_src_output"] = save_prefix + ".valid.src.out"
    output_files_dict["sqlite_db"] = save_prefix + ".result.sqlite"
    output_files_dict["optimizer_ckpt"] = save_prefix + ".optimizer." + "ckpt" + ".npz"
    output_files_dict["optimizer_final"] = save_prefix + ".optimizer." + "final" + ".npz"
    
    
    
    save_prefix_dir, save_prefix_fn = os.path.split(save_prefix)
    ensure_path(save_prefix_dir)
    
    already_existing_files = []
    for key_info, filename in output_files_dict.iteritems():#, valid_data_fn]:
        if os.path.exists(filename):
            already_existing_files.append(filename)
    if len(already_existing_files) > 0:
        print "Warning: existing files are going to be replaced / updated: ",  already_existing_files
        raw_input("Press Enter to Continue")
    
    
    save_train_config_fn = output_files_dict["train_config"]
    log.info("Saving training config to %s" % save_train_config_fn)
    config_training.save_to(save_train_config_fn)
#     json.dump(config_training, open(save_train_config_fn, "w"), indent=2, separators=(',', ': '))
    
    Vi = len(src_indexer) # + UNK
    Vo = len(tgt_indexer) # + UNK
    
    eos_idx = Vo
    
    data_fn = config_training.data.data_fn
    
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
        

    
    max_src_tgt_length = config_training.training_management.max_src_tgt_length
    if max_src_tgt_length is not None:
        log.info("filtering sentences of length larger than %i"%(max_src_tgt_length))
        filtered_training_data = []
        nb_filtered = 0
        for src, tgt in training_data:
            if len(src) <= max_src_tgt_length and len(tgt) <= max_src_tgt_length:
                filtered_training_data.append((src, tgt))
            else:
                nb_filtered += 1
        log.info("filtered %i sentences of length larger than %i"%(nb_filtered, max_src_tgt_length))
        training_data = filtered_training_data
    
    if not config_training.training.no_shuffle_of_training_data:
        log.info("shuffling")
        import random
        random.shuffle(training_data)
        log.info("done")
    
    
    encdec = create_encdec_and_indexers_from_config_dict(config_training, 
                            src_indexer = src_indexer, tgt_indexer = tgt_indexer,
                            load_config_model = "if_exists" if config_training.training_management.resume else "no")
#     create_encdec_from_config_dict(config_training.model, src_indexer, tgt_indexer, 
#                             load_config_model = "if_exists" if config_training.training_management.resume else "no")
    
#     if config_training.training_management.resume:
#         if "model_parameters" not in config_training:
#             log.error("cannot find model parameters in config file")
#         if config_training.model_parameters.type == "model":
#             model_filename = config_training.model_parameters.filename
#             log.info("resuming from model parameters %s" % model_filename)
#             serializers.load_npz(model_filename, encdec)
    
    if config_training.training_management.load_model is not None:
        log.info("loading model parameters from %s", config_training.training_management.load_model)
        serializers.load_npz(config_training.training_management.load_model, encdec)
    
    
    gpu = config_training.training_management.gpu
    if gpu is not None:
        encdec = encdec.to_gpu(gpu)
    
    if config_training.training.optimizer == "adadelta":
        optimizer = optimizers.AdaDelta()
    elif config_training.training.optimizer == "adam":
        optimizer = optimizers.Adam()
    elif config_training.training.optimizer == "adagrad":
        optimizer = optimizers.AdaGrad(lr = config_training.training.learning_rate)
    elif config_training.training.optimizer == "sgd":
        optimizer = optimizers.SGD(lr = config_training.training.learning_rate)
    elif config_training.training.optimizer == "momentum":
        optimizer = optimizers.MomentumSGD(lr = config_training.training.learning_rate,
                                           momentum = config_training.training.momentum)
    elif config_training.training.optimizer == "nesterov":
        optimizer = optimizers.NesterovAG(lr = config_training.training.learning_rate,
                                           momentum = config_training.training.momentum)
    elif config_training.training.optimizer == "rmsprop":
        optimizer = optimizers.RMSprop(lr = config_training.training.learning_rate)
    elif config_training.training.optimizer == "rmspropgraves":
        optimizer = optimizers.RMSpropGraves(lr = config_training.training.learning_rate,
                                           momentum = config_training.training.momentum)    
    else:
        raise NotImplemented
    
    with cuda.get_device(gpu):
        optimizer.setup(encdec)
    
    if config_training.training.l2_gradient_clipping is not None and config_training.training.l2_gradient_clipping > 0:
        optimizer.add_hook(chainer.optimizer.GradientClipping(config_training.training.l2_gradient_clipping))

    if config_training.training.hard_gradient_clipping is not None and config_training.training.hard_gradient_clipping > 0:
        optimizer.add_hook(chainer.optimizer.GradientHardClipping(*config_training.training.hard_gradient_clipping))

    if config_training.training.weight_decay is not None:
        optimizer.add_hook(chainer.optimizer.WeightDecay(config_training.training.weight_decay))

    if config_training.training_management.load_optimizer_state is not None:
        with cuda.get_device(gpu):
            log.info("loading optimizer parameters from %s", config_training.training_management.load_optimizer_state)
            serializers.load_npz(config_training.training_management.load_optimizer_state, optimizer)    
    

    import training_chainer
    with cuda.get_device(gpu):
        if config_training.training_management.max_nb_iters is not None:
            stop_trigger = (config_training.training_management.max_nb_iters, "iteration")
            if config_training.training_management.max_nb_epochs is not None:
                log.warn("max_nb_iters and max_nb_epochs both specified. Only max_nb_iters will be considered.")
        elif config_training.training_management.max_nb_epochs is not None:
            stop_trigger = (config_training.training_management.max_nb_epochs, "epoch")
        else:
            stop_trigger = None
        training_chainer.train_on_data_chainer(encdec, optimizer, training_data, output_files_dict,
                      src_indexer, tgt_indexer, eos_idx = eos_idx, 
                      config_training = config_training,
                      stop_trigger = stop_trigger,
                      test_data = test_data, dev_data = dev_data, valid_data = valid_data
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
