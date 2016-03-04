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

from utils import minibatch_provider
from evaluation import ( 
                  compute_loss_all, translate_to_file, sample_once)

logging.basicConfig()
log = logging.getLogger("rnns:training")
log.setLevel(logging.INFO)

def train_on_data(encdec, optimizer, training_data, output_files_dict,
                  src_voc, tgt_voc, eos_idx, mb_size = 80,
                  nb_of_batch_to_sort = 20,
                  test_data = None, dev_data = None, gpu = None, report_every = 200, randomized = False):
    
    mb_provider = minibatch_provider(training_data, eos_idx, mb_size, nb_of_batch_to_sort, gpu = gpu,
                                     randomized = randomized, sort_key = lambda x:len(x[0]))
    
#     mb_provider = minibatch_provider(training_data, eos_idx, mb_size, nb_of_batch_to_sort, gpu = gpu,
#                                      randomized = randomized, sort_key = lambda x:len(x[1]))
    
    
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
        
    def train_once(src_batch, tgt_batch, src_mask):
        t0 = time.clock()
        encdec.zerograds()
        t1 = time.clock()
        (total_loss, total_nb_predictions), attn = encdec(src_batch, tgt_batch, src_mask, raw_loss_info = True)
        loss = total_loss / total_nb_predictions
        t2 = time.clock()
        print "loss:", loss.data,
        loss.backward()
        t3 = time.clock()
        optimizer.update()
        t4 = time.clock()
        print " time %f zgrad:%f fwd:%f bwd:%f upd:%f"%(t4-t0, t1-t0, t2-t1, t3-t2, t4-t3)
        return float(total_loss.data), total_nb_predictions
        
#     def train_once_optim(src_batch, tgt_batch, src_mask):
#         t0 = time.clock()
#         encdec.zerograds()
#         t1 = time.clock()
#         total_loss, total_nb_predictions = encdec.compute_loss_and_backward(src_batch, tgt_batch, src_mask, raw_loss_info = True)
#         loss = total_loss / total_nb_predictions
#         t2 = time.clock()
#         print "loss:", loss,
#         t3 = time.clock()
#         optimizer.update()
#         t4 = time.clock()
#         print " time %f zgrad:%f fwd:%f bwd:%f upd:%f"%(t4-t0, t1-t0, t2-t1, t3-t2, t4-t3)
#         return total_loss, total_nb_predictions    
        
    if test_data is not None:
        test_src_data = [x for x,y in test_data]
        test_references = [y for x,y in test_data]
        def translate_test():
            translations_fn = output_files_dict["test_translation_output"] #save_prefix + ".test.out"
            control_src_fn = output_files_dict["test_src_output"] #save_prefix + ".test.src.out"
            return translate_to_file(encdec, eos_idx, test_src_data, mb_size, tgt_voc, 
                   translations_fn, test_references = test_references, control_src_fn = control_src_fn,
                   src_voc = src_voc, gpu = gpu, nb_steps = 50)
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
                   src_voc = src_voc, gpu = gpu, nb_steps = 50)
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
        best_dev_loss = None
        prev_time = time.clock()
        prev_i = None
        total_loss_this_interval = 0 
        total_nb_predictions_this_interval = 0
        for i in xrange(sys.maxint):
            print i,
            src_batch, tgt_batch, src_mask = mb_provider.next()
#             if i%100 == 0:
#                 print "valid", 
#                 compute_valid()
            if i%200 == 0:
                for v in src_batch + tgt_batch:
                    v.volatile = "on"
                sample_once(encdec, src_batch, tgt_batch, src_mask, src_voc, tgt_voc, eos_idx,
                            max_nb = 20)
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
                
                if best_dev_loss is None or dev_loss >= best_dev_loss:
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
    (date text, bleu_info text, iteration real, loss real, bleu real, dev_loss real, dev_bleu real, avg_time real, avg_training_loss real)''')
                    infos = (datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"), 
                             repr(bc_test), i, float(test_loss), bc_test.bleu(), float(dev_loss), bc_dev.bleu(), avg_time, avg_training_loss)
                    db_cursor.execute("INSERT INTO exp_data VALUES (?,?,?,?,?,?,?,?,?)", infos)
                    db_connection.commit()
                    db_connection.close()
                    
                    if bc_dev.bleu() > best_dev_bleu:
                        best_dev_bleu = bc_dev.bleu()
                        log.info("saving best model %f" % best_dev_bleu)
                        save_model("best")
                prev_time = time.clock()
            if i%1000 == 0:       
                save_model("ckpt")
                                        
            total_loss, total_nb_predictions = train_once(src_batch, tgt_batch, src_mask)
#             total_loss, total_nb_predictions = train_once_optim(src_batch, tgt_batch, src_mask)
            
            total_loss_this_interval += total_loss
            total_nb_predictions_this_interval += total_nb_predictions
    finally:
        save_model("final")