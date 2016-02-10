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

logging.basicConfig()
log = logging.getLogger("rnns:train")
log.setLevel(logging.INFO)

def ensure_path(path):
    import os
    try: 
        os.makedirs(path)
        log.info("Created directory %s" % path)
    except OSError:
        if not os.path.isdir(path):
            raise

def make_batch_src(src_data, eos_idx, padding_idx = 0, gpu = None):
    max_src_size = max(len(x) for x  in src_data)
    mb_size = len(src_data)
    src_batch = [np.empty((mb_size,), dtype = np.int32) for _ in xrange(max_src_size + 1)]
    src_mask = [np.empty((mb_size,), dtype = np.bool) for _ in xrange(max_src_size + 1)]
    
    for num_ex in xrange(mb_size):
        this_src_len = len(src_data[num_ex])
        for i in xrange(max_src_size + 1):
            if i < this_src_len:
                src_batch[i][num_ex] = src_data[num_ex][i]
                src_mask[i][num_ex] = True
            else:
                src_batch[i][num_ex] = padding_idx
                src_mask[i][num_ex] = False

    if gpu is not None:
        return ([Variable(cuda.to_gpu(x, gpu)) for x in src_batch],
                [Variable(cuda.to_gpu(x, gpu)) for x in src_mask])
    else:
        return [Variable(x) for x in src_batch], [Variable(x) for x in src_mask]                
                
def make_batch_src_tgt(training_data, eos_idx = 1, padding_idx = 0, gpu = None):
    training_data = sorted(training_data, key = lambda x:len(x[1]), reverse = True)
#     max_src_size = max(len(x) for x, y  in training_data)
    max_tgt_size = max(len(y) for x, y  in training_data)
    mb_size = len(training_data)
    
#     src_batch = [np.empty((mb_size,), dtype = np.int32) for _ in xrange(max_src_size + 1)]
#     tgt_batch = [] #[np.empty((mb_size,), dtype = np.int32) for _ in xrange(max_tgt_size + 1)]
#     src_mask = [np.empty((mb_size,), dtype = np.bool) for _ in xrange(max_src_size + 1)]
    
    src_batch, src_mask = make_batch_src(
                [x for x,y in training_data], eos_idx = eos_idx, padding_idx = padding_idx, gpu = gpu)
    
#     for num_ex in xrange(mb_size):
#         this_src_len = len(training_data[num_ex][0])
#         for i in xrange(max_src_size + 1):
#             if i < this_src_len:
#                 src_batch[i][num_ex] = training_data[num_ex][0][i]
#                 src_mask[i][num_ex] = True
# #             elif i == this_src_len:
# #                 src_batch[i][num_ex] = eos_idx
# #                 src_mask[i][num_ex] = True
#             else:
#                 src_batch[i][num_ex] = padding_idx
#                 src_mask[i][num_ex] = False
            
    lengths_list = []
    lowest_non_finished = mb_size -1
    for pos in xrange(max_tgt_size + 1):
        while pos > len(training_data[lowest_non_finished][1]):
            lowest_non_finished -= 1
            assert lowest_non_finished >= 0
        mb_length_at_this_pos = lowest_non_finished + 1
        assert len(lengths_list) == 0 or mb_length_at_this_pos <= lengths_list[-1]
        lengths_list.append(mb_length_at_this_pos)
        
    tgt_batch = []
    for i in xrange(max_tgt_size + 1):
        current_mb_size = lengths_list[i]
        assert current_mb_size > 0
        tgt_batch.append(np.empty((current_mb_size,), dtype = np.int32))
        for num_ex in xrange(current_mb_size):
#             print num_ex, training_data[num_ex][1]
            assert len(training_data[num_ex][1]) >= i
            if len(training_data[num_ex][1]) == i:
                tgt_batch[-1][num_ex] = eos_idx
            else:
                tgt_batch[-1][num_ex] = training_data[num_ex][1][i]
        
    if gpu is not None:
        tgt_batch_v = [Variable(cuda.to_gpu(x, gpu)) for x in tgt_batch]
    else:
        tgt_batch_v = [Variable(x) for x in tgt_batch]
    
    return src_batch, tgt_batch_v, src_mask
#         return ([Variable(cuda.to_gpu(x, gpu)) for x in src_batch], [Variable(cuda.to_gpu(x, gpu)) for x in tgt_batch],
#                 [Variable(cuda.to_gpu(x, gpu)) for x in src_mask])
#     else:
#         return [Variable(x) for x in src_batch], [Variable(x) for x in tgt_batch], [Variable(x) for x in src_mask]

def greedy_batch_translate(encdec, eos_idx, src_data, batch_size = 80, gpu = None):
    nb_ex = len(src_data)
    nb_batch = nb_ex / batch_size + (1 if nb_ex % batch_size != 0 else 0)
    res = []
    for i in range(nb_batch):
        current_batch_raw_data = src_data[i * batch_size : (i + 1) * batch_size]
        src_batch, src_mask = make_batch_src(current_batch_raw_data, eos_idx = eos_idx, gpu = gpu)
        sample_greedy, score = encdec(src_batch, 50, src_mask, use_best_for_sample = True)
        for sent_num in xrange(len(src_batch[0].data)):
            res.append([])
            for smp_pos in range(len(sample_greedy)):
                idx_smp = cuda.to_cpu(sample_greedy[smp_pos][sent_num])
                if idx_smp == eos_idx:
                    break
                res[-1].append(idx_smp)
    return res
            
def convert_idx_to_string(seq, voc):
    trans = []
    for idx_tgt in seq:
        if idx_tgt >= len(voc):
            log.warn("found unknown idx in tgt : %i / %i"% (idx_tgt, len(voc)))
        else:
            trans.append(voc[idx_tgt])
    return " ".join(trans)

def minibatch_looper(data, mb_size, loop = True, avoid_copy = False):
    current_start = 0
    data_exhausted = False
    while not data_exhausted:
        if avoid_copy and len(data) >= current_start + mb_size:
            training_data_sampled = data[current_start: current_start + mb_size]
            current_start += mb_size
            if current_start >= len(data):
                if loop:
                    current_start = 0
                else:
                    data_exhausted = True
                    break
        else:
            training_data_sampled = []
            while len(training_data_sampled) < mb_size:
                remaining = mb_size - len(training_data_sampled)
                training_data_sampled += data[current_start:current_start + remaining]
                current_start += remaining
                if current_start >= len(data):
                    if loop:
                        current_start = 0
                    else:
                        data_exhausted = True
                        break
        
        yield training_data_sampled
        
def batch_sort_and_split(batch, size_parts, sort_key = lambda x:len(x[1]), inplace = False):
#             training_data_sampled = training_data[current_start:current_start + mb_size * nb_mb_for_sorting]
    if not inplace:
        batch = list(batch)
    batch.sort(key = sort_key)
    nb_mb_for_sorting = len(batch) / size_parts + (1 if len(batch) % size_parts != 0 else 0)
    for num_batch in xrange(nb_mb_for_sorting):
        mb_raw = batch[num_batch * size_parts : (num_batch + 1) * size_parts]
        yield mb_raw
        
def minibatch_provider(data, eos_idx, mb_size, nb_mb_for_sorting = 1, loop = True, inplace_sorting = False, gpu = None):
    if nb_mb_for_sorting == -1:
        assert loop == False
        for mb_raw in batch_sort_and_split(data, mb_size, inplace = inplace_sorting):
            src_batch, tgt_batch, src_mask = make_batch_src_tgt(mb_raw, eos_idx = eos_idx, gpu = gpu)
            yield src_batch, tgt_batch, src_mask
    else:
        assert nb_mb_for_sorting > 0
        required_data = nb_mb_for_sorting * mb_size
        for large_batch in minibatch_looper(data, required_data, loop = loop, avoid_copy = False):
            # ok to sort in place since minibatch_looper will return copies
            for mb_raw in batch_sort_and_split(large_batch, mb_size, inplace = True):
                src_batch, tgt_batch, src_mask = make_batch_src_tgt(mb_raw, eos_idx = eos_idx, gpu = gpu)
                yield src_batch, tgt_batch, src_mask
             
def compute_bleu_with_unk_as_wrong(references, candidates, unk_id, new_unk_id_ref, new_unk_id_cand):
    import bleu_computer
    assert new_unk_id_ref != new_unk_id_cand
    bc = bleu_computer.BleuComputer()
    for ref, cand in zip(references, candidates):
        ref_mod = tuple((x if x != unk_id else new_unk_id_ref) for x in ref)
        cand_mod = tuple((int(x) if int(x) != unk_id else new_unk_id_cand) for x in cand)
#         print ref_mod, type(ref_mod)
#         print cand_mod, type(cand_mod)
        bc.update(ref_mod, cand_mod)
    return bc
        
        
        
             
def train_on_data(encdec, optimizer, training_data, output_files_dict,
                  src_voc, tgt_voc, eos_idx, mb_size = 80,
                  nb_of_batch_to_sort = 20,
                  test_data = None, gpu = None):
    
    mb_provider = minibatch_provider(training_data, eos_idx, mb_size, nb_of_batch_to_sort, gpu = gpu)
    
    def save_model(suffix):
        if suffix == "final":
            fn_save = output_files_dict["model_ckpt"]
        elif suffix =="ckpt":
            fn_save = output_files_dict["model_ckpt"]
        else:
            assert False
#         fn_save = save_prefix + ".model." + suffix + ".npz"
        log.info("saving model to %s" % fn_save)
        serializers.save_npz(fn_save, encdec)
#         optimizer_fn_save = save_prefix + ".optim.ckpt.npz"
#         log.info("saving optimizer state to %s" % optimizer_fn_save)
#         serializers.save_npz(optimizer_fn_save, optimizer)
        
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
            log.info("writing translation of test set to %s"% translations_fn)
            translations = greedy_batch_translate(encdec, eos_idx, test_src_data, batch_size = mb_size, gpu = gpu)
            
            unk_id = len(tgt_voc) - 1
            new_unk_id_ref = unk_id + 7777
            new_unk_id_cand = unk_id + 9999
            bc = compute_bleu_with_unk_as_wrong(test_references, translations, unk_id, new_unk_id_ref, new_unk_id_cand)
            log.info("test bleu: %r"%bc)
            
            out = codecs.open(translations_fn, "w", encoding = "utf8")
            for t in translations:
                out.write(convert_idx_to_string(t, tgt_voc) + "\n")
            
            control_src_fn = output_files_dict["test_src_output"] #save_prefix + ".test.src.out"
            control_out = codecs.open(control_src_fn, "w", encoding = "utf8")
            log.info("writing src of test set to %s"% control_src_fn)
            for s in test_src_data:
                control_out.write(convert_idx_to_string(s, src_voc) + "\n")
                
            return bc
                
        def compute_test_loss():
            log.info("computing test loss")
            mb_provider_test = minibatch_provider(test_data, eos_idx, mb_size, nb_mb_for_sorting = -1, loop = False,
                                                  gpu = gpu)
            test_loss = 0
            test_nb_predictions = 0
            for src_batch, tgt_batch, src_mask in mb_provider_test:
                loss, attn = encdec(src_batch, tgt_batch, src_mask, raw_loss_info = True)
                test_loss += loss[0].data
                test_nb_predictions += loss[1]
            test_loss /= test_nb_predictions
            log.info("test loss: %f" % test_loss)
            
            return test_loss
            
    else:
        def translate_test():
            log.info("translate_test: No test data given")
        def compute_test_loss():
            log.info("compute_test_loss: No test data given")   
        
        
    def sample_once(src_batch, tgt_batch, src_mask):
        print "sample"
        sample_greedy, score = encdec(src_batch, 50, src_mask, use_best_for_sample = True)
#                 sample, score = encdec(src_batch, 50, src_mask, use_best_for_sample = False)
        assert len(src_batch[0].data) == len(tgt_batch[0].data)
        assert len(sample_greedy[0]) == len(src_batch[0].data)
        for sent_num in xrange(len(src_batch[0].data)):
            print "sent num", sent_num
            print "src idx:",
            for src_pos in range(len(src_batch)):
                if src_mask[src_pos].data[sent_num]:
                    idx_src = cuda.to_cpu(src_batch[src_pos].data[sent_num])
                    print idx_src,
            print
            print "src:",
            for src_pos in range(len(src_batch)):
                if src_mask[src_pos].data[sent_num]:
                    idx_src = cuda.to_cpu(src_batch[src_pos].data[sent_num])
#                             print idx_src, type(idx_src)
                    if idx_src >= len(src_voc):
                        log.warn("found unknown idx in src : %i / %i"% (idx_src, len(src_voc)))
                    else:
                        print src_voc[idx_src],
            print
            print "tgt idx:",
            for tgt_pos in range(len(tgt_batch)):
                if sent_num >= len(tgt_batch[tgt_pos].data):
                    break
                idx_tgt = cuda.to_cpu(tgt_batch[tgt_pos].data[sent_num])
                print idx_tgt,
            print
            print "tgt:",
            for tgt_pos in range(len(tgt_batch)):
                if sent_num >= len(tgt_batch[tgt_pos].data):
                    break
                idx_tgt = cuda.to_cpu(tgt_batch[tgt_pos].data[sent_num])
                if idx_tgt == eos_idx:
                    print "EOS",
                elif idx_tgt >= len(tgt_voc):
                    log.warn("found unknown idx in tgt : %i / %i"% (idx_tgt, len(tgt_voc)))
                else:
                    print tgt_voc[idx_tgt],
            print
#                     print "sample idx:"
#                     for smp_pos in range(len(sample)):
#                         idx_smp = cuda.to_cpu(sample[smp_pos][sent_num])
#                         print idx_smp,
#                     print
#                     print "sample:"
#                     for smp_pos in range(len(sample)):
#                         idx_smp = cuda.to_cpu(sample[smp_pos][sent_num])
#                         if idx_smp == eos_idx:
#                             print "EOS",
#                             break
#                         if idx_smp >= len(tgt_voc):
#                             log.warn("found unknown idx in tgt during sampling : %i / %i"% (idx_smp, len(tgt_voc)))
#                         else:
#                             print tgt_voc[idx_smp],
#                     print
            print "greedy idx:"
            for smp_pos in range(len(sample_greedy)):
                idx_smp = cuda.to_cpu(sample_greedy[smp_pos][sent_num])
                print idx_smp,
            print
            print "greedy:"
            for smp_pos in range(len(sample_greedy)):
                idx_smp = cuda.to_cpu(sample_greedy[smp_pos][sent_num])
                if idx_smp == eos_idx:
                    print "EOS",
                    break
                if idx_smp >= len(tgt_voc):
                    log.warn("found unknown idx in tgt during sampling : %i / %i"% (idx_smp, len(tgt_voc)))
                else:
                    print tgt_voc[idx_smp],
            print
            print
            
            
    try:
        for i in xrange(100000):
            print i,
            src_batch, tgt_batch, src_mask = mb_provider.next()
            train_once(src_batch, tgt_batch, src_mask)
#             if i%100 == 0:
#                 print "valid", 
#                 compute_valid()
            if i%200 == 0:
                sample_once(src_batch, tgt_batch, src_mask)
                
            if i%500 == 0:
                bc_test = translate_test()
                test_loss = compute_test_loss()
                if bc_test is not None:
                    assert test_loss is not None
                    import sqlite3, datetime
                    db_path = output_files_dict["sqlite_db"]
                    log.info("saving test results to %s" %(db_path))
                    db_connection = sqlite3.connect(db_path)
                    db_cursor = db_connection.cursor()
                    db_cursor.execute('''CREATE TABLE IF NOT EXISTS exp_data 
                             (date text, bleu_info text, iteration real, loss real, bleu real)''')
                    infos = (datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"), repr(bc_test), i, float(test_loss), bc_test.bleu())
                    db_cursor.execute("INSERT INTO exp_data VALUES (?,?,?,?,?)", infos)
                    db_connection.commit()
                    db_connection.close()
                    
            if i%1000 == 0:       
                save_model("ckpt")
                
#                 fn_save = save_prefix + ".model.ckpt.npz"
#                 log.info("saving model to %s" % fn_save)
#                 serializers.save_npz(fn_save, encdec)
#                 optimizer_fn_save = save_prefix + ".optim.ckpt.npz"
#                 log.info("saving optimizer state to %s" % optimizer_fn_save)
#                 serializers.save_npz(optimizer_fn_save, optimizer)
#                 print sample
#                 print score
    finally:
        save_model("final")
#         fn_save = save_prefix + ".model.final.npz"
#         log.info("saving model to %s"% fn_save)
#         serializers.save_npz(fn_save, encdec)
#         optimizer_fn_save = save_prefix + ".optim.final.npz"
#         log.info("saving optimizer state to %s" % optimizer_fn_save)
#         serializers.save_npz(optimizer_fn_save, optimizer)


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
    output_files_dict["test_translation_output"] = args.save_prefix + ".test.out"
    output_files_dict["test_src_output"] = args.save_prefix + ".test.src.out"
    
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
    
    
    log.info("loading voc from %s"% voc_fn)
    src_voc, tgt_voc = json.load(open(voc_fn))
    
    Vi = len(src_voc) + 1 # + UNK
    Vo = len(tgt_voc) + 1 # + UNK
    
    config_training = {"command_line:" : args.__dict__, "Vi": Vi, "Vo" : Vo, "voc" : voc_fn, "data" : data_fn}
    save_train_config_fn = output_files_dict["train_config"]
    log.info("Saving training config to %s" % save_train_config_fn)
    json.dump(config_training, open(save_train_config_fn, "w"), indent=2, separators=(',', ': '))
#     Ei = 620, Eo = 620, Hi = 1000, Ho = 1000, Ha = 1000, Hl = 500
    
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
#     with cuda.get_device(args.gpu)
        train_on_data(encdec, optimizer, training_data, output_files_dict,
                      src_voc + ["#S_UNK#"], tgt_voc + ["#T_UNK#"], eos_idx = eos_idx, 
                      mb_size = args.mb_size,
                      nb_of_batch_to_sort = args.nb_batch_to_sort,
                      test_data = test_data, gpu = args.gpu)
    
#     test_double(args.gpu)

if __name__ == '__main__':
    command_line()
