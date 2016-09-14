#!/usr/bin/env python
"""training.py: training procedures."""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

from chainer import serializers
import time

import logging
import sys
# import h5py

import math

from utils import minibatch_provider, minibatch_provider_curiculum
from evaluation import ( 
                  compute_loss_all, translate_to_file, sample_once)

logging.basicConfig()
log = logging.getLogger("rnns:training")
log.setLevel(logging.INFO)

def sent_complexity(sent):
    rank_least_common_word = max(sent)
    length = len(sent)
    return length * math.log(rank_least_common_word + 1)
    
def example_complexity(ex):
    return sent_complexity(ex[0]) + sent_complexity(ex[1])

def train_on_data(encdec, optimizer, training_data, output_files_dict,
                  src_indexer, tgt_indexer, eos_idx, mb_size = 80,
                  nb_of_batch_to_sort = 20,
                  test_data = None, dev_data = None, valid_data = None,
                  gpu = None, report_every = 200, randomized = False,
                  reverse_src = False, reverse_tgt = False, max_nb_iters = None, do_not_save_data_for_resuming = False,
                  noise_on_prev_word = False, curiculum_training = False,
                  use_previous_prediction = 0, no_report_or_save = False,
                  use_memory_optimization = False, sample_every = 200,
                  use_reinf = False,
                  save_ckpt_every = 2000):
#     ,
#                   lexical_probability_dictionary = None,
#                   V_tgt = None,
#                   lexicon_prob_epsilon = 1e-3):
    
    if curiculum_training:
        log.info("Sorting training data by complexity")
        training_data_sorted_by_complexity = sorted(training_data, key = example_complexity)
        log.info("done")
        
        for s,t in training_data_sorted_by_complexity[:400]:
            print example_complexity((s,t))
            print " ".join(src_indexer.deconvert(s))
            print " ".join(tgt_indexer.deconvert(t))
            print
            
        mb_provider = minibatch_provider_curiculum(training_data_sorted_by_complexity, eos_idx, mb_size, nb_of_batch_to_sort, gpu = gpu,
                                     randomized = randomized, sort_key = lambda x:len(x[0]),
                                     reverse_src = reverse_src, reverse_tgt = reverse_tgt)
    else:
        mb_provider = minibatch_provider(training_data, eos_idx, mb_size, nb_of_batch_to_sort, gpu = gpu,
                                     randomized = randomized, sort_key = lambda x:len(x[0]),
                                     reverse_src = reverse_src, reverse_tgt = reverse_tgt)
    
#     mb_provider = minibatch_provider(training_data, eos_idx, mb_size, nb_of_batch_to_sort, gpu = gpu,
#                                      randomized = randomized, sort_key = lambda x:len(x[1]))
    
    s_unk_tag = lambda num,utag:"S_UNK_%i"%utag
    t_unk_tag = lambda num,utag:"T_UNK_%i"%utag
    
    def save_model(suffix):
        if suffix == "final":
            fn_save = output_files_dict["model_final"]
        elif suffix =="ckpt":
            fn_save = output_files_dict["model_ckpt"]
        elif suffix =="best":
            fn_save = output_files_dict["model_best"]
        elif suffix =="best_loss":
            fn_save = output_files_dict["model_best_loss"]
        else:
            assert False
        log.info("saving model to %s" % fn_save)
        serializers.save_npz(fn_save, encdec)
        
    def train_once(src_batch, tgt_batch, src_mask): #, lexicon_matrix = None):
        t0 = time.clock()
        encdec.zerograds()
        t1 = time.clock()
        (total_loss, total_nb_predictions), attn = encdec(src_batch, tgt_batch, src_mask, raw_loss_info = True,
                                                          noise_on_prev_word = noise_on_prev_word,
                                                          use_previous_prediction = use_previous_prediction,
                                                          mode = "train")
#         ,
#                                                           lexicon_probability_matrix = lexicon_matrix, 
#                                                           lex_epsilon = lexicon_prob_epsilon)
        loss = total_loss / total_nb_predictions
        t2 = time.clock()
        loss.backward()
        t3 = time.clock()
        optimizer.update()
        t4 = time.clock()
        print "loss:", loss.data,
        print " time %f zgrad:%f fwd:%f bwd:%f upd:%f"%(t4-t0, t1-t0, t2-t1, t3-t2, t4-t3)
        return float(total_loss.data), total_nb_predictions
        
    def train_once_optim(src_batch, tgt_batch, src_mask):
        t0 = time.clock()
        encdec.zerograds()
        t1 = time.clock()
        loss, total_nb_predictions = encdec.compute_loss_and_backward(src_batch, tgt_batch, src_mask)
        t2 = time.clock()
        print "loss:", loss,
        t3 = time.clock()
        optimizer.update()
        t4 = time.clock()
        print " time %f zgrad:%f fwd:%f bwd:%f upd:%f"%(t4-t0, t1-t0, t2-t1, t3-t2, t4-t3)
        return float(loss)*total_nb_predictions, total_nb_predictions  
        
        
    def train_once_reinf(src_batch, tgt_batch, src_mask): #, lexicon_matrix = None):
        t0 = time.clock()
        encdec.zerograds()
        t1 = time.clock()
        
        import utils
        test_ref = utils.de_batch(tgt_batch, is_variable = True)
        
        reinf_loss = encdec.get_reinf_loss(src_batch, src_mask, eos_idx, 
                    test_ref, nb_steps = 50, nb_samples = 5, 
                    use_best_for_sample = False,
                    temperature = None,
                    mode = "test")

        t2 = time.clock()
        reinf_loss.backward()
        t3 = time.clock()
        optimizer.update()
        t4 = time.clock()
        print "reinf loss:", reinf_loss.data, reinf_loss.data/len(src_batch)
        print " time %f zgrad:%f fwd:%f bwd:%f upd:%f"%(t4-t0, t1-t0, t2-t1, t3-t2, t4-t3)
        return float(reinf_loss.data), len(src_batch)
        
    if test_data is not None:
        test_src_data = [x for x,y in test_data]
        test_references = [y for x,y in test_data]
        def translate_test():
            translations_fn = output_files_dict["test_translation_output"] #save_prefix + ".test.out"
            control_src_fn = output_files_dict["test_src_output"] #save_prefix + ".test.src.out"
            return translate_to_file(encdec, eos_idx, test_src_data, mb_size, tgt_indexer, 
                   translations_fn, test_references = test_references, control_src_fn = control_src_fn,
                   src_indexer = src_indexer, gpu = gpu, nb_steps = 50, reverse_src = reverse_src, reverse_tgt = reverse_tgt,
                   s_unk_tag = s_unk_tag, t_unk_tag = t_unk_tag)
        def compute_test_loss():
            log.info("computing test loss")
            test_loss = compute_loss_all(encdec, test_data, eos_idx, mb_size, gpu = gpu,
                                         reverse_src = reverse_src, reverse_tgt = reverse_tgt)
            log.info("test loss: %f" % test_loss)
            return test_loss
    else:
        def translate_test():
            log.info("translate_test: No test data given")
            return None
        def compute_test_loss():
            log.info("compute_test_loss: No test data given")
            return None
            
    if dev_data is not None:
        dev_src_data = [x for x,y in dev_data]
        dev_references = [y for x,y in dev_data]
        def translate_dev():
            translations_fn = output_files_dict["dev_translation_output"] #save_prefix + ".test.out"
            control_src_fn = output_files_dict["dev_src_output"] #save_prefix + ".test.src.out"
            return translate_to_file(encdec, eos_idx, dev_src_data, mb_size, tgt_indexer, 
                   translations_fn, test_references = dev_references, control_src_fn = control_src_fn,
                   src_indexer = src_indexer, gpu = gpu, nb_steps = 50, reverse_src = reverse_src, reverse_tgt = reverse_tgt,
                   s_unk_tag = s_unk_tag, t_unk_tag = t_unk_tag)
        def compute_dev_loss():
            log.info("computing dev loss")
            dev_loss = compute_loss_all(encdec, dev_data, eos_idx, mb_size, gpu = gpu,
                                         reverse_src = reverse_src, reverse_tgt = reverse_tgt)
            log.info("dev loss: %f" % dev_loss)
            return dev_loss
    else:
        def translate_dev():
            log.info("translate_dev: No dev data given")
            return None
        def compute_dev_loss():
            log.info("compute_dev_loss: No dev data given")
            return None     

    if valid_data is not None:
        valid_src_data = [x for x,y in valid_data]
        valid_references = [y for x,y in valid_data]
        def translate_valid():
            translations_fn = output_files_dict["valid_translation_output"] #save_prefix + ".test.out"
            control_src_fn = output_files_dict["valid_src_output"] #save_prefix + ".test.src.out"
            return translate_to_file(encdec, eos_idx, valid_src_data, mb_size, tgt_indexer, 
                   translations_fn, test_references = valid_references, control_src_fn = control_src_fn,
                   src_indexer = src_indexer, gpu = gpu, nb_steps = 50, reverse_src = reverse_src, reverse_tgt = reverse_tgt,
                   s_unk_tag = s_unk_tag, t_unk_tag = t_unk_tag)
        def compute_valid_loss():
            log.info("computing valid loss")
            dev_loss = compute_loss_all(encdec, valid_data, eos_idx, mb_size, gpu = gpu,
                                         reverse_src = reverse_src, reverse_tgt = reverse_tgt)
            log.info("valid loss: %f" % dev_loss)
            return dev_loss
    else:
        def translate_valid():
            log.info("translate_valid: No valid data given")
            return None
        def compute_valid_loss():
            log.info("compute_valid_loss: No valid data given")
            return None  
    
    
    try:
        best_dev_bleu = 0
        best_dev_loss = None
        prev_time = time.clock()
        prev_i = None
        total_loss_this_interval = 0 
        total_nb_predictions_this_interval = 0
        for i in xrange(sys.maxint):
            if max_nb_iters is not None and max_nb_iters <= i:
                break
            print i,
            src_batch, tgt_batch, src_mask = mb_provider.next()
            if src_batch[0].data.shape[0] != mb_size:
                log.warn("got minibatch of size %i instead of %i"%(src_batch[0].data.shape[0], mb_size))
                
#             if lexical_probability_dictionary is not None:
#                 lexicon_matrix = utils.compute_lexicon_matrix(src_batch, lexical_probability_dictionary)
#                 if gpu is not None:
#                     lexicon_matrix = cuda.to_gpu(lexicon_matrix, gpu)
#             else:
#                 lexicon_matrix = None
                
#             if i%100 == 0:
#                 print "valid", 
#                 compute_valid()
            if not no_report_or_save:
                if i%sample_every == 0:
                    for v in src_batch + tgt_batch:
                        v.volatile = "on"
                    sample_once(encdec, src_batch, tgt_batch, src_mask, src_indexer, tgt_indexer, eos_idx,
                                max_nb = 20,
                                s_unk_tag = s_unk_tag, t_unk_tag = t_unk_tag)
                    for v in src_batch + tgt_batch:
                        v.volatile = "off"
                if i%report_every == 0:
                    current_time = time.clock()
                    if prev_i is not None:
                        iteration_interval = i-prev_i
                        avg_time = (current_time - prev_time) /(iteration_interval)
                        avg_training_loss = total_loss_this_interval / total_nb_predictions_this_interval
                        avg_sentence_size = float(total_nb_predictions_this_interval)/ (iteration_interval * mb_size)
                        
                    
                    else:
                        avg_time = 0
                        avg_training_loss = 0
                        avg_sentence_size = 0
                    prev_i = i
                    total_loss_this_interval = 0 
                    total_nb_predictions_this_interval = 0                
                    
                    print "avg time:", avg_time
                    print "avg training loss:", avg_training_loss
                    print "avg sentence size", avg_sentence_size
                    
                    bc_test = translate_test()
                    test_loss = compute_test_loss()
                    bc_dev = translate_dev()
                    dev_loss = compute_dev_loss()
                    bc_valid = translate_valid()
                    valid_loss = compute_valid_loss()
                    
                    
                    if dev_loss is not None and (best_dev_loss is None or dev_loss <= best_dev_loss):
                        best_dev_loss = dev_loss
                        log.info("saving best loss model %f" % best_dev_loss)
                        save_model("best_loss")
                        
                    if bc_test is not None:
                        
                        assert test_loss is not None
                        import sqlite3, datetime
                        db_path = output_files_dict["sqlite_db"]
                        log.info("saving test results to %s" %(db_path))
                        db_connection = sqlite3.connect(db_path)
                        db_cursor = db_connection.cursor()
                        db_cursor.execute('''CREATE TABLE IF NOT EXISTS exp_data 
        (date text, bleu_info text, iteration real, 
        loss real, bleu real, 
        dev_loss real, dev_bleu real, 
        valid_loss real, valid_bleu real,
        avg_time real, avg_training_loss real)''')
                        infos = (datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"), 
                                 repr(bc_test), i, float(test_loss), bc_test.bleu(), 
                                 float(dev_loss), bc_dev.bleu(), 
                                 float(valid_loss) if valid_loss is not None else None, bc_valid.bleu() if bc_valid is not None else None,
                                 avg_time, avg_training_loss)
                        db_cursor.execute("INSERT INTO exp_data VALUES (?,?,?,?,?,?,?,?,?,?,?)", infos)
                        db_connection.commit()
                        db_connection.close()
                        
                        if bc_dev.bleu() > best_dev_bleu:
                            best_dev_bleu = bc_dev.bleu()
                            log.info("saving best model %f" % best_dev_bleu)
                            save_model("best")
                    prev_time = time.clock()
                if i%save_ckpt_every == 0:       
                    save_model("ckpt")
                    fn_save_optimizer = output_files_dict["optimizer_ckpt"]
                    log.info("saving optimizer parameters to %s" % fn_save_optimizer)
                    serializers.save_npz(fn_save_optimizer, optimizer)
            
            if use_memory_optimization:
#                 if lexicon_matrix is not None:
#                     raise NotImplemented
                total_loss, total_nb_predictions = train_once_optim(src_batch, tgt_batch, src_mask)
            elif use_reinf:
                total_loss, total_nb_predictions = train_once_reinf(src_batch, tgt_batch, src_mask)
            else:                      
                total_loss, total_nb_predictions = train_once(src_batch, tgt_batch, src_mask)
#                 , 
#                                                               lexicon_matrix = lexicon_matrix)
            
            total_loss_this_interval += total_loss
            total_nb_predictions_this_interval += total_nb_predictions
    finally:
        if not do_not_save_data_for_resuming and not no_report_or_save:
            save_model("final")
            fn_save_optimizer = output_files_dict["optimizer_final"]
            log.info("saving optimizer parameters to %s" % fn_save_optimizer)
            serializers.save_npz(fn_save_optimizer, optimizer)
        