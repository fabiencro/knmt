#!/usr/bin/env python
"""train.py: Train a RNNSearch Model"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import chainer
from chainer import cuda, optimizers, serializers

from training import train_on_data
from nmt_chainer.dataprocessing.indexer import Indexer
from nmt_chainer.utilities.file_infos import create_filename_infos
from nmt_chainer.utilities.argument_parsing_tools import OrderedNamespace
import nmt_chainer.models.feedforward.encoder_decoder
import numpy as np
import logging
import json
import os.path
import gzip
import sys
import pprint
import time
import os.path
# import h5py

from nmt_chainer.utilities.utils import ensure_path
# , make_batch_src_tgt, make_batch_src, minibatch_provider, compute_bleu_with_unk_as_wrong,de_batch
# from evaluation import (
#                   greedy_batch_translate, convert_idx_to_string,
#                   compute_loss_all, translate_to_file, sample_once)

import nmt_chainer.models.attention
import nmt_chainer.models.encoder_decoder

import nmt_chainer.models.rnn_cells as rnn_cells
import nmt_chainer.dataprocessing.processors as processors

import nmt_chainer.utilities.profiling_tools as profiling_tools

logging.basicConfig()
log = logging.getLogger("rnns:train")
log.setLevel(logging.INFO)


def generate_lexical_probability_dictionary_indexed(lexical_probability_dictionary_all, src_indexer, tgt_indexer):
    log.info("computing lexical_probability_dictionary_indexed")
    lexical_probability_dictionary_indexed = {}
    for ws in lexical_probability_dictionary_all:
        ws_idx_array = src_indexer.convert(ws)
        if len(ws_idx_array) > 1:
            log.warning("Converting an entry of the lexical probability dictionary resulted in several ids. "
                        "Be aware that --lexical_probability_dictionary option is not fully compatible "
                        "with fancier preprocessing options such as BPE.")
        ws_idx = ws_idx_array[0]
        if ws_idx in lexical_probability_dictionary_indexed:
            assert src_indexer.is_unk_idx(ws_idx)
        else:
            lexical_probability_dictionary_indexed[ws_idx] = {}
        for wt in lexical_probability_dictionary_all[ws]:
            wt_idx_array = tgt_indexer.convert(wt)
            if len(wt_idx_array) > 1:
                log.warning("Converting an entry of the lexical probability dictionary resulted in several ids. "
                            "Be aware that --lexical_probability_dictionary option is not fully compatible "
                            "with fancier preprocessing options such as BPE.")
            wt_idx = wt_idx_array[0]
            if wt_idx in lexical_probability_dictionary_indexed[ws_idx]:
                assert src_indexer.is_unk_idx(
                    ws_idx) or tgt_indexer.is_unk_idx(wt_idx)
                lexical_probability_dictionary_indexed[ws_idx][wt_idx] += lexical_probability_dictionary_all[ws][wt]
            else:
                lexical_probability_dictionary_indexed[ws_idx][wt_idx] = lexical_probability_dictionary_all[ws][wt]
    return lexical_probability_dictionary_indexed


def create_encdec_from_config_dict(config_dict, src_indexer, tgt_indexer):
    Vi = len(src_indexer)  # + UNK
    Vo = len(tgt_indexer)  # + UNK
    
    if config_dict.get("use_ff_model", False):
        d_model = config_dict["ff_d_model"]
        n_heads = config_dict["ff_n_heads"]
        nb_layers_src = config_dict["ff_nb_layers_src"]
        nb_layers_tgt = config_dict["ff_nb_layers_tgt"]
        use_exp_relu = config_dict["ff_use_exp_relu"]
        dropout = config_dict["ff_dropout"]
        d_ff = config_dict.get("ff_d_ff", 2048)
        
        if config_dict.get("use_own_layer_normalization", False):
            from nmt_chainer.additional_links.layer_normalization import turn_on_own_layer_normalization
            turn_on_own_layer_normalization()
        
        no_add = config_dict.get("ff_no_add", False) #backward compatibility
        if no_add:
            residual_mode = None
        else:
            residual_mode = config_dict.get("ff_residual_mode", "normal")
        
        no_normalize = config_dict.get("ff_no_normalize", False)
        encdec = nmt_chainer.models.feedforward.encoder_decoder.EncoderDecoder(Vi, Vo, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                                     experimental_relu=use_exp_relu, dropout=dropout, 
                                                     nb_layers_src=nb_layers_src, nb_layers_tgt=nb_layers_tgt,
                                                     residual_mode = residual_mode, no_normalize = no_normalize)
    else:
        Ei = config_dict["Ei"]
        Hi = config_dict["Hi"]
        Eo = config_dict["Eo"]
        Ho = config_dict["Ho"]
        Ha = config_dict["Ha"]
        Hl = config_dict["Hl"]
    
        encoder_cell_type = config_dict.get("encoder_cell_type", "gru")
        decoder_cell_type = config_dict.get("decoder_cell_type", "gru")
    
        use_bn_length = config_dict.get("use_bn_length", None)
    
        # Selecting Attention type
        attn_cls = nmt_chainer.models.attention.AttentionModule
        if config_dict.get("use_accumulated_attn", False):
            raise NotImplemented
        if config_dict.get("use_deep_attn", False):
            attn_cls = nmt_chainer.models.attention.DeepAttentionModule
    
        init_orth = config_dict.get("init_orth", False)
    
        if "lexical_probability_dictionary" in config_dict and config_dict["lexical_probability_dictionary"] is not None:
            log.info("opening lexical_probability_dictionary %s" % config_dict["lexical_probability_dictionary"])
            lexical_probability_dictionary_all = json.load(gzip.open(config_dict["lexical_probability_dictionary"], "rb"))
    
            lexical_probability_dictionary = generate_lexical_probability_dictionary_indexed(
                lexical_probability_dictionary_all, src_indexer, tgt_indexer)
        else:
            lexical_probability_dictionary = None
        lex_epsilon = config_dict.get("lexicon_prob_epsilon", 0.001)
    
        use_goto_attention = config_dict.get("use_goto_attention", False)
    
        # Creating encoder/decoder
        encdec = nmt_chainer.models.encoder_decoder.EncoderDecoder(Vi, Ei, Hi, Vo + 1, Eo, Ho, Ha, Hl, use_bn_length=use_bn_length,
                                                                   attn_cls=attn_cls,
                                                                   init_orth=init_orth,
                                                                   encoder_cell_type=rnn_cells.create_cell_model_from_config(encoder_cell_type),
                                                                   decoder_cell_type=rnn_cells.create_cell_model_from_config(decoder_cell_type),
                                                                   lexical_probability_dictionary=lexical_probability_dictionary,
                                                                   lex_epsilon=lex_epsilon,
                                                                   use_goto_attention=use_goto_attention)

    return encdec


class NpzDeserializerAverage(chainer.serializer.Deserializer):

    def __init__(self, npz_list, path='', strict=True):
        self.npz_list = npz_list
        self.path = path
        self.strict = strict

    def __getitem__(self, key):
        key = key.strip('/')
        return NpzDeserializerAverage(
            self.npz_list, self.path + key + '/', strict=self.strict)

    def __call__(self, key, value):
        key = self.path + key.lstrip('/')
        if not self.strict and key not in self.npz:
            return value
        
        dataset = None
        for npz in self.npz_list:
            try:
                this_d = npz[key]
            except KeyError:
                this_d = npz["updater/model:main/"+key]
            if dataset is None:
                dataset = this_d
            else:
                dataset = dataset + this_d
        dataset /= len(self.npz_list)
            
        if value is None:
            return dataset
        elif isinstance(value, np.ndarray):
            np.copyto(value, dataset)
        elif isinstance(value, cuda.ndarray):
            value.set(np.asarray(dataset))
        else:
            value = type(value)(np.asarray(dataset))
        return value


def load_npz_average(filename_list, obj):
    d = NpzDeserializerAverage([np.load(filename) for filename in filename_list])
    d.load(obj)
    
    
def load_model_flexible(filename_list, encdec):
    mode = "normal"
    if isinstance(filename_list, tuple) or isinstance(filename_list, list):
        if len(filename_list) == 1:
            filename_list = filename_list[0]
        else:
            mode = "average"
            
    if mode == "normal":
        log.info("loading model parameters from %s", filename_list)
        try:
            serializers.load_npz(filename_list, encdec)
        except KeyError:
            log.info("not model format, trying snapshot format")
            with np.load(filename_list) as fseri:
                dicseri = serializers.NpzDeserializer(fseri, path="updater/model:main/")
                dicseri.load(encdec)        
    else:
        assert mode == "average"
        log.info("loading averaged model parameters from %r", filename_list)
        dseri = NpzDeserializerAverage([np.load(filename) for filename in filename_list])
        dseri.load(encdec)

def create_encdec_and_indexers_from_config_dict(config_dict, src_indexer=None, tgt_indexer=None, load_config_model="no",
                                                return_model_infos=False, additional_models_parameters_for_averaging=None):
    assert load_config_model in "yes no if_exists".split()

    if src_indexer is None or tgt_indexer is None:
        voc_fn = config_dict.data["voc"]
        log.info("loading voc from %s" % voc_fn)
#         src_voc, tgt_voc = json.load(open(voc_fn))

        bi_idx = processors.load_pp_pair_from_file(voc_fn)

    if src_indexer is None:
        src_indexer = bi_idx.src_processor()

    if tgt_indexer is None:
        tgt_indexer = bi_idx.tgt_processor()

#     tgt_voc = None
#     src_voc = None

    encdec = create_encdec_from_config_dict(config_dict["model"], src_indexer, tgt_indexer)

    eos_idx = len(tgt_indexer)

    model_infos = None

    if load_config_model != "no":
        if "model_parameters" not in config_dict:
            assert additional_models_parameters_for_averaging is None
            if load_config_model == "yes":
                log.error("cannot find model parameters in config file")
                raise ValueError(
                    "Config file do not contain model_parameters section")
        else:
            model_filename = config_dict.model_parameters.filename
            if additional_models_parameters_for_averaging is not None:
                load_model_flexible([model_filename]+additional_models_parameters_for_averaging, encdec)
            else:
                load_model_flexible(model_filename, encdec)
#             if config_dict.model_parameters.type == "model":
#                 log.info(
#                     "loading model parameters from file specified by config file:%s" %
#                     model_filename)
#                 serializers.load_npz(model_filename, encdec)
#                 if return_model_infos:
#                     model_infos = create_filename_infos(model_filename)
#             else:
#                 log.info("loading model parameters from snapshot file specified by config file:%s" %model_filename)
#                 with np.load(model_filename) as fs:
#                     dics = serializers.NpzDeserializer(fs, path="updater/model:main/")
#                     dics.load(encdec)
            if return_model_infos:
                model_infos = create_filename_infos(model_filename)
    else:
        assert additional_models_parameters_for_averaging is None

    result = encdec, eos_idx, src_indexer, tgt_indexer
    if return_model_infos:
        return result, model_infos
    else:
        return result


def load_voc_and_update_training_config(config_training):
    data_prefix = config_training["training_management"]["data_prefix"]
    voc_fn = data_prefix + ".voc"
    data_fn = data_prefix + ".data.json.gz"

    log.info("loading voc from %s" % voc_fn)
#     src_voc, tgt_voc = json.load(open(voc_fn))

    bi_idx = processors.load_pp_pair_from_file(voc_fn)

    src_indexer, tgt_indexer = bi_idx.src_processor(), bi_idx.tgt_processor()
#     src_indexer = processors.PreProcessor.make_from_serializable(src_voc)
#     tgt_indexer = processors.PreProcessor.make_from_serializable(tgt_voc)
#     tgt_voc = None
#     src_voc = None

#     Vi = len(src_voc) + 1 # + UNK
#     Vo = len(tgt_voc) + 1 # + UNK

    Vi = len(src_indexer)  # + UNK
    Vo = len(tgt_indexer)  # + UNK

    config_training.add_section("data", keep_at_bottom="metadata", overwrite=False)
    config_training["data"]["data_fn"] = data_fn
    config_training["data"]["Vi"] = Vi
    config_training["data"]["Vo"] = Vo
    config_training["data"]["voc"] = voc_fn

    config_training.set_readonly()

    return src_indexer, tgt_indexer


def do_train(config_training):
    if config_training["training_management"]["disable_cudnn_softmax"]:
        import nmt_chainer.models.feedforward.multi_attention
        nmt_chainer.models.feedforward.multi_attention.disable_cudnn_softmax = True

    src_indexer, tgt_indexer = load_voc_and_update_training_config(config_training)

    save_prefix = config_training.training_management.save_prefix

    output_files_dict = {}
    output_files_dict["train_config"] = save_prefix + ".train.config"
    output_files_dict["model_ckpt"] = save_prefix + ".model." + "ckpt" + ".npz"
    output_files_dict["model_final"] = save_prefix + \
        ".model." + "final" + ".npz"
    output_files_dict["model_best"] = save_prefix + ".model." + "best" + ".npz"
    output_files_dict["model_best_loss"] = save_prefix + ".model." + "best_loss" + ".npz"

#     output_files_dict["model_ckpt_config"] = save_prefix + ".model." + "ckpt" + ".config"
#     output_files_dict["model_final_config"] = save_prefix + ".model." + "final" + ".config"
#     output_files_dict["model_best_config"] = save_prefix + ".model." + "best" + ".config"
#     output_files_dict["model_best_loss_config"] = save_prefix + ".model." + "best_loss" + ".config"

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
    for key_info, filename in output_files_dict.iteritems():  # , valid_data_fn]:
        if os.path.exists(filename):
            already_existing_files.append(filename)
    if len(already_existing_files) > 0:
        print "Warning: existing files are going to be replaced / updated: ", already_existing_files
        if not config_training.training_management.force_overwrite:
            raw_input("Press Enter to Continue")

    save_train_config_fn = output_files_dict["train_config"]
    log.info("Saving training config to %s" % save_train_config_fn)
    config_training.save_to(save_train_config_fn)
#     json.dump(config_training, open(save_train_config_fn, "w"), indent=2, separators=(',', ': '))

    Vi = len(src_indexer)  # + UNK
    Vo = len(tgt_indexer)  # + UNK

    eos_idx = Vo

    data_fn = config_training.data.data_fn

    log.info("loading training data from %s" % data_fn)
    training_data_all = json.load(gzip.open(data_fn, "rb"))

    training_data = training_data_all["train"]

    log.info("loaded %i sentences as training data" % len(training_data))

    if "test" in training_data_all:
        test_data = training_data_all["test"]
        log.info("Found test data: %i sentences" % len(test_data))
    else:
        test_data = None
        log.info("No test data found")

    if "dev" in training_data_all:
        dev_data = training_data_all["dev"]
        log.info("Found dev data: %i sentences" % len(dev_data))
    else:
        dev_data = None
        log.info("No dev data found")

    if "valid" in training_data_all:
        valid_data = training_data_all["valid"]
        log.info("Found valid data: %i sentences" % len(valid_data))
    else:
        valid_data = None
        log.info("No valid data found")

    max_src_tgt_length = config_training.training_management.max_src_tgt_length
    if max_src_tgt_length is not None:
        log.info("filtering sentences of length larger than %i" % (max_src_tgt_length))
        filtered_training_data = []
        nb_filtered = 0
        for src, tgt in training_data:
            if len(src) <= max_src_tgt_length and len(
                    tgt) <= max_src_tgt_length:
                filtered_training_data.append((src, tgt))
            else:
                nb_filtered += 1
        log.info("filtered %i sentences of length larger than %i" % (nb_filtered, max_src_tgt_length))
        training_data = filtered_training_data

    if not config_training.training.no_shuffle_of_training_data:
        log.info("shuffling")
        import random
        random.shuffle(training_data)
        log.info("done")

    encdec, _, _, _ = create_encdec_and_indexers_from_config_dict(config_training,
                                                                  src_indexer=src_indexer, tgt_indexer=tgt_indexer,
                                                                  load_config_model="if_exists" if config_training.training_management.resume else "no")
    
    if (config_training.training.get("load_initial_source_embeddings", None) is not None or
        config_training.training.get("load_initial_target_embeddings", None) is not None):
        src_emb = None
        tgt_emb = None
        
        src_emb_fn = config_training.training.get("load_initial_source_embeddings", None)
        tgt_emb_fn = config_training.training.get("load_initial_target_embeddings", None)
        
        if src_emb_fn is not None:
            log.info("loading source embeddings from %s", src_emb_fn)
            src_emb = np.load(src_emb_fn)
        
        if tgt_emb_fn is not None:
            log.info("loading target embeddings from %s", tgt_emb_fn)
            tgt_emb = np.load(tgt_emb_fn)
        
        encdec.initialize_embeddings(src_emb, tgt_emb, no_unk_src=True, no_unk_tgt=True)
        
    
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
        load_model_flexible(config_training.training_management.load_model, encdec)
#         try:
#             serializers.load_npz(config_training.training_management.load_model, encdec)
#         except KeyError:
#             log.info("not model format, trying snapshot format")
#             with np.load(config_training.training_management.load_model) as fseri:
#                 dicseri = serializers.NpzDeserializer(fseri, path="updater/model:main/")
#                 dicseri.load(encdec)
                
    gpu = config_training.training_management.gpu
    if gpu is not None:
        encdec = encdec.to_gpu(gpu)

    if config_training.training.optimizer == "adadelta":
        optimizer = optimizers.AdaDelta()
    elif config_training.training.optimizer == "adam":
        optimizer = optimizers.Adam()
    elif config_training.training.optimizer == "scheduled_adam":
        from nmt_chainer.additional_links.scheduled_adam import ScheduledAdam
        optimizer = ScheduledAdam(d_model=config_training.model.ff_d_model)
    elif config_training.training.optimizer == "adagrad":
        optimizer = optimizers.AdaGrad(lr=config_training.training.learning_rate)
    elif config_training.training.optimizer == "sgd":
        optimizer = optimizers.SGD(lr=config_training.training.learning_rate)
    elif config_training.training.optimizer == "momentum":
        optimizer = optimizers.MomentumSGD(lr=config_training.training.learning_rate,
                                           momentum=config_training.training.momentum)
    elif config_training.training.optimizer == "nesterov":
        optimizer = optimizers.NesterovAG(lr=config_training.training.learning_rate,
                                          momentum=config_training.training.momentum)
    elif config_training.training.optimizer == "rmsprop":
        optimizer = optimizers.RMSprop(lr=config_training.training.learning_rate)
    elif config_training.training.optimizer == "rmspropgraves":
        optimizer = optimizers.RMSpropGraves(lr=config_training.training.learning_rate,
                                             momentum=config_training.training.momentum)
    else:
        raise NotImplemented

    with cuda.get_device(gpu):
        optimizer.setup(encdec)

    if config_training.training.l2_gradient_clipping is not None and config_training.training.l2_gradient_clipping > 0:
        optimizer.add_hook(chainer.optimizer.GradientClipping(
            config_training.training.l2_gradient_clipping))

    if config_training.training.hard_gradient_clipping is not None and config_training.training.hard_gradient_clipping > 0:
        optimizer.add_hook(chainer.optimizer.GradientHardClipping(
            *config_training.training.hard_gradient_clipping))

    if config_training.training.weight_decay is not None:
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(
                config_training.training.weight_decay))

    if config_training.training_management.load_optimizer_state is not None:
        with cuda.get_device(gpu):
            log.info("loading optimizer parameters from %s", config_training.training_management.load_optimizer_state)
            serializers.load_npz(config_training.training_management.load_optimizer_state, optimizer)

    if config_training.training_management.timer_hook:
        timer_hook = profiling_tools.MyTimerHook
    else:
        import contextlib

        @contextlib.contextmanager
        def timer_hook():
            yield

    import training_chainer
    with cuda.get_device(gpu):
        with timer_hook() as timer_infos:

            if config_training.training_management.max_nb_iters is not None:
                stop_trigger = (
                    config_training.training_management.max_nb_iters,
                    "iteration")
                if config_training.training_management.max_nb_epochs is not None:
                    log.warn(
                        "max_nb_iters and max_nb_epochs both specified. Only max_nb_iters will be considered.")
            elif config_training.training_management.max_nb_epochs is not None:
                stop_trigger = (
                    config_training.training_management.max_nb_epochs, "epoch")
            else:
                stop_trigger = None
            training_chainer.train_on_data_chainer(encdec, optimizer, training_data, output_files_dict,
                                                   src_indexer, tgt_indexer, eos_idx=eos_idx,
                                                   config_training=config_training,
                                                   stop_trigger=stop_trigger,
                                                   test_data=test_data, dev_data=dev_data, valid_data=valid_data
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
