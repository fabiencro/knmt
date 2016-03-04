#!/usr/bin/env python
"""train.py: Train a RNNSearch Model"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import chainer
from chainer import cuda, optimizers, serializers

import models
from training import train_on_data

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

logging.basicConfig()
log = logging.getLogger("rnns:train")
log.setLevel(logging.INFO)

def command_line():
    import argparse
    parser = argparse.ArgumentParser(description= "Train a RNNSearch model", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_prefix", help = "prefix of the training data created by make_data.py")
    parser.add_argument("save_prefix", help = "prefix to be added to all files created during the training")
    parser.add_argument("--gpu", type = int, help = "specify gpu number to use, if any")
    parser.add_argument("--load_model", help = "load the parameters of a previously trained model")
    parser.add_argument("--load_optimizer_state", help = "load previously saved optimizer states")
    parser.add_argument("--Ei", type = int, default= 620, help = "Source words embedding size.")
    parser.add_argument("--Eo", type = int, default= 620, help = "Target words embedding size.")
    parser.add_argument("--Hi", type = int, default= 1000, help = "Source encoding layer size.")
    parser.add_argument("--Ho", type = int, default= 1000, help = "Target hidden layer size.")
    parser.add_argument("--Ha", type = int, default= 1000, help = "Attention Module Hidden layer size.")
    parser.add_argument("--Hl", type = int, default= 500, help = "Maxout output size.")
    parser.add_argument("--mb_size", type = int, default= 80, help = "Minibatch size")
    parser.add_argument("--nb_batch_to_sort", type = int, default= 20, help = "Sort this many batches by size.")
    
    
    parser.add_argument("--max_src_tgt_length", type = int, help = "Limit length of training sentences")
    
    parser.add_argument("--l2_gradient_clipping", type = float, help = "L2 gradient clipping")
    parser.add_argument("--weight_decay", type = float, help = "weight decay")
    
    parser.add_argument("--optimizer", choices=["sgd", "rmsprop", "rmspropgraves", 
                            "momentum", "nesterov", "adam", "adagrad", "adadelta"], 
                        default = "adadelta")
    parser.add_argument("--learning_rate", type = float, default= 0.01, help = "Learning Rate")
    parser.add_argument("--momentum", type = float, default= 0.9, help = "Momentum term")
    parser.add_argument("--report_every", type = int, default = 200, help = "report every x iterations")
    parser.add_argument("--randomized_data", default = False, action = "store_true")
    parser.add_argument("--use_accumulated_attn", default = False, action = "store_true")
    
    parser.add_argument("--shuffle_training_data", default = False, action = "store_true")
    
    parser.add_argument("--init_orth", default = False, action = "store_true")
    
    args = parser.parse_args()
    
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
    
    output_files_dict["sqlite_db"] = args.save_prefix + ".result.sqlite"
    
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
        log.info("No test data found")
        
    log.info("loading voc from %s"% voc_fn)
    src_voc, tgt_voc = json.load(open(voc_fn))
    
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
    
    if args.shuffle_training_data:
        log.info("shuffling")
        import random
        random.shuffle(training_data)
        log.info("done")
    
    
    Vi = len(src_voc) + 1 # + UNK
    Vo = len(tgt_voc) + 1 # + UNK
    
    config_training = {"command_line" : args.__dict__, "Vi": Vi, "Vo" : Vo, "voc" : voc_fn, "data" : data_fn}
    save_train_config_fn = output_files_dict["train_config"]
    log.info("Saving training config to %s" % save_train_config_fn)
    json.dump(config_training, open(save_train_config_fn, "w"), indent=2, separators=(',', ': '))
    
    eos_idx = Vo
    
    if args.use_accumulated_attn:
        encdec = models.EncoderDecoder(Vi, args.Ei, args.Hi, Vo + 1, args.Eo, args.Ho, args.Ha, args.Hl,
                                       attn_cls= models.AttentionModuleAcumulated,
                                       init_orth = args.init_orth)
    else:
        encdec = models.EncoderDecoder(Vi, args.Ei, args.Hi, Vo + 1, args.Eo, args.Ho, args.Ha, args.Hl,
                                       init_orth = args.init_orth)
    
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
    optimizer.setup(encdec)
    
    if args.l2_gradient_clipping is not None:
        optimizer.add_hook(chainer.optimizer.GradientClipping(args.l2_gradient_clipping))

    if args.weight_decay is not None:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    if args.load_optimizer_state is not None:
        serializers.load_npz(args.load_optimizer_state, optimizer)    
    
    with cuda.cupy.cuda.Device(args.gpu):
        train_on_data(encdec, optimizer, training_data, output_files_dict,
                      src_voc + ["#S_UNK#"], tgt_voc + ["#T_UNK#"], eos_idx = eos_idx, 
                      mb_size = args.mb_size,
                      nb_of_batch_to_sort = args.nb_batch_to_sort,
                      test_data = test_data, dev_data = dev_data, gpu = args.gpu, report_every = args.report_every,
                      randomized = args.randomized_data)

if __name__ == '__main__':
    command_line()
