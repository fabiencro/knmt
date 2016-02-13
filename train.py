#!/usr/bin/env python
"""train.py: Train a RNNSearch Model"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils

import models

import collections
import logging
import codecs
import json
import exceptions
import itertools, operator
import os.path
import gzip
# import h5py

from utils import ensure_path, make_batch_src_tgt, make_batch_src, minibatch_provider, compute_bleu_with_unk_as_wrong,de_batch
from eval import (
                  greedy_batch_translate, convert_idx_to_string, greedy_batch_translate_with_attn, 
                  compute_loss_all, translate_to_file, sample_once)

logging.basicConfig()
log = logging.getLogger("rnns:train")
log.setLevel(logging.INFO)

def train_on_data(encdec, optimizer, training_data, output_files_dict,
                  src_voc, tgt_voc, eos_idx, mb_size = 80,
                  nb_of_batch_to_sort = 20,
                  test_data = None, dev_data = None, gpu = None):
    
    mb_provider = minibatch_provider(training_data, eos_idx, mb_size, nb_of_batch_to_sort, gpu = gpu)
    
    def save_model(suffix):
        if suffix == "final":
            fn_save = output_files_dict["model_final"]
        elif suffix =="ckpt":
            fn_save = output_files_dict["model_ckpt"]
        elif suffix =="best":
            fn_save = output_files_dict["model_best"]
        else:
            assert False
        log.info("saving model to %s" % fn_save)
        serializers.save_npz(fn_save, encdec)
        
    def train_once(src_batch, tgt_batch, src_mask):
        encdec.zerograds()
        loss, attn = encdec(src_batch, tgt_batch, src_mask)
        print loss.data
        loss.backward()
        optimizer.update()
        
    if test_data is not None:
        test_src_data = [x for x,y in test_data]
        test_references = [y for x,y in test_data]
        def translate_test():
            translations_fn = output_files_dict["test_translation_output"] #save_prefix + ".test.out"
            control_src_fn = output_files_dict["test_src_output"] #save_prefix + ".test.src.out"
            return translate_to_file(encdec, eos_idx, test_src_data, mb_size, tgt_voc, 
                   translations_fn, test_references = test_references, control_src_fn = control_src_fn,
                   src_voc = src_voc, gpu = gpu)
        def compute_test_loss():
            log.info("computing test loss")
            test_loss = compute_loss_all(encdec, test_data, eos_idx, mb_size, gpu = gpu)
            log.info("test loss: %f" % test_loss)
            return test_loss
    else:
        def translate_test():
            log.info("translate_test: No test data given")
        def compute_test_loss():
            log.info("compute_test_loss: No test data given")
            
    if dev_data is not None:
        dev_src_data = [x for x,y in dev_data]
        dev_references = [y for x,y in dev_data]
        def translate_dev():
            translations_fn = output_files_dict["dev_translation_output"] #save_prefix + ".test.out"
            control_src_fn = output_files_dict["dev_src_output"] #save_prefix + ".test.src.out"
            return translate_to_file(encdec, eos_idx, dev_src_data, mb_size, tgt_voc, 
                   translations_fn, test_references = dev_references, control_src_fn = control_src_fn,
                   src_voc = src_voc, gpu = gpu)
        def compute_dev_loss():
            log.info("computing dev loss")
            dev_loss = compute_loss_all(encdec, dev_data, eos_idx, mb_size, gpu = gpu)
            log.info("dev loss: %f" % dev_loss)
            return dev_loss
    else:
        def translate_dev():
            log.info("translate_dev: No dev data given")
        def compute_dev_loss():
            log.info("compute_dev_loss: No dev data given")        
    
    try:
        best_dev_bleu = 0
        for i in xrange(100000):
            print i,
            src_batch, tgt_batch, src_mask = mb_provider.next()
            train_once(src_batch, tgt_batch, src_mask)
#             if i%100 == 0:
#                 print "valid", 
#                 compute_valid()
            if i%200 == 0:
                sample_once(encdec, src_batch, tgt_batch, src_mask, src_voc, tgt_voc, eos_idx)
                
            if i%200 == 0:
                bc_test = translate_test()
                test_loss = compute_test_loss()
                bc_dev = translate_dev()
                dev_loss = compute_dev_loss()
                if bc_test is not None:
                    assert test_loss is not None
                    import sqlite3, datetime
                    db_path = output_files_dict["sqlite_db"]
                    log.info("saving test results to %s" %(db_path))
                    db_connection = sqlite3.connect(db_path)
                    db_cursor = db_connection.cursor()
                    db_cursor.execute('''CREATE TABLE IF NOT EXISTS exp_data 
                             (date text, bleu_info text, iteration real, loss real, bleu real, dev_loss real, dev_bleu real)''')
                    infos = (datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"), 
                             repr(bc_test), i, float(test_loss), bc_test.bleu(), float(dev_loss), bc_dev.bleu())
                    db_cursor.execute("INSERT INTO exp_data VALUES (?,?,?,?,?,?,?)", infos)
                    db_connection.commit()
                    db_connection.close()
                    
                    if bc_dev.bleu() > best_dev_bleu:
                        best_dev_bleu = bc_dev.bleu()
                        log.info("saving best model %f" % best_dev_bleu)
                        save_model("best")
                    
            if i%1000 == 0:       
                save_model("ckpt")
    finally:
        save_model("final")


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
    parser.add_argument("--l2_gradient_clipping", type = float, help = "L2 gradient clipping")
    args = parser.parse_args()
    
    output_files_dict = {}
    output_files_dict["train_config"] = args.save_prefix + ".train.config"
    output_files_dict["model_ckpt"] = args.save_prefix + ".model." + "ckpt" + ".npz"
    output_files_dict["model_final"] = args.save_prefix + ".model." + "final" + ".npz"
    output_files_dict["model_best"] = args.save_prefix + ".model." + "best" + ".npz"
    
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
    
    Vi = len(src_voc) + 1 # + UNK
    Vo = len(tgt_voc) + 1 # + UNK
    
    config_training = {"command_line" : args.__dict__, "Vi": Vi, "Vo" : Vo, "voc" : voc_fn, "data" : data_fn}
    save_train_config_fn = output_files_dict["train_config"]
    log.info("Saving training config to %s" % save_train_config_fn)
    json.dump(config_training, open(save_train_config_fn, "w"), indent=2, separators=(',', ': '))
    
    eos_idx = Vo
    encdec = models.EncoderDecoder(Vi, args.Ei, args.Hi, Vo + 1, args.Eo, args.Ho, args.Ha, args.Hl)
    
    if args.load_model is not None:
        serializers.load_npz(args.load_model, encdec)
    
    if args.gpu is not None:
        encdec = encdec.to_gpu(args.gpu)
    
    optimizer = optimizers.AdaDelta()
    optimizer.setup(encdec)
    
    if args.l2_gradient_clipping is not None:
        optimizer.add_hook(chainer.optimizer.GradientClipping(args.l2_gradient_clipping))

    if args.load_optimizer_state is not None:
        serializers.load_npz(args.load_optimizer_state, optimizer)    
    
    with cuda.cupy.cuda.Device(args.gpu):
        train_on_data(encdec, optimizer, training_data, output_files_dict,
                      src_voc + ["#S_UNK#"], tgt_voc + ["#T_UNK#"], eos_idx = eos_idx, 
                      mb_size = args.mb_size,
                      nb_of_batch_to_sort = args.nb_batch_to_sort,
                      test_data = test_data, dev_data = dev_data, gpu = args.gpu)

if __name__ == '__main__':
    command_line()
