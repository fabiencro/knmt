#!/usr/bin/env python
"""training.py: training procedures."""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

from nmt_chainer.utilities import argument_parsing_tools
from chainer import serializers
import chainer.functions as F
import time

import logging
# import h5py

import chainer

from nmt_chainer.utilities.utils import minibatch_provider, minibatch_provider_curiculum, make_batch_src_tgt
from nmt_chainer.translation.evaluation import sample_once
from nmt_chainer.training_module.training_extensions import (ComputeLossExtension, 
                                                             ComputeBleuExtension, 
                                                             TrainingLossSummaryExtension,
                                                             SqliteLogExtension,
                                                             CheckpontSavingExtension)

from training_updater import LengthBasedSerialIterator, Updater, ScheduledIterator, UpdaterScheduledLearning

logging.basicConfig()
log = logging.getLogger("rnns:training")
log.setLevel(logging.INFO)

def train_on_data_chainer(encdec, optimizer, training_data, output_files_dict,
                          src_indexer, tgt_indexer, eos_idx,
                          config_training,
                          stop_trigger=None,
                          test_data=None, dev_data=None, valid_data=None,
                          ):

    output_dir = config_training.training_management.save_prefix
    mb_size = config_training.training.mb_size
    nb_of_batch_to_sort = config_training.training.nb_batch_to_sort
    gpu = config_training.training_management.gpu
    report_every = config_training.training_management.report_every
    randomized = config_training.training.randomized_data
    reverse_src = config_training.training.reverse_src
    reverse_tgt = config_training.training.reverse_tgt
    do_not_save_data_for_resuming = config_training.training_management.no_resume
    noise_on_prev_word = config_training.training.noise_on_prev_word
    curiculum_training = config_training.training.curiculum_training
    use_previous_prediction = config_training.training.use_previous_prediction
    no_report_or_save = config_training.training_management.no_report_or_save
    use_memory_optimization = config_training.training_management.use_memory_optimization
    sample_every = config_training.training_management.sample_every
    use_reinf = config_training.training.use_reinf
    save_ckpt_every = config_training.training_management.save_ckpt_every
    trainer_snapshot = config_training.training_management.load_trainer_snapshot
    save_initial_model_to = config_training.training_management.save_initial_model_to
    reshuffle_every_epoch = config_training.training_management.reshuffle_every_epoch
    
    sr_schedule_ratio = config_training.training_management.sr_schedule_ratio
    sr_schedule_cap = config_training.training_management.sr_schedule_cap
    sr_threshold = config_training.training_management.sr_threshold
    scheduled = sr_schedule_ratio is not None
    
    
    @chainer.training.make_extension()
    def sample_extension(trainer):
        encdec = trainer.updater.get_optimizer("main").target
        iterator = trainer.updater.get_iterator("main")
        mb_raw = iterator.peek()
        
        if scheduled:
            mb_raw = mb_raw[1]

        src_batch, tgt_batch, src_mask = make_batch_src_tgt(mb_raw, eos_idx=eos_idx, padding_idx=0, gpu=gpu, volatile="on", need_arg_sort=False)

        def s_unk_tag(num, utag):
            return "S_UNK_%i" % utag

        def t_unk_tag(num, utag):
            return "T_UNK_%i" % utag

        sample_once(encdec, src_batch, tgt_batch, src_mask, src_indexer, tgt_indexer, eos_idx,
                    max_nb=20,
                    s_unk_tag=s_unk_tag, t_unk_tag=t_unk_tag)

    
    if scheduled:
        iterator_training_data = LengthBasedSerialIterator(training_data, mb_size,
                                            nb_of_batch_to_sort=nb_of_batch_to_sort,
                                            sort_key=lambda x: len(x[1][0]),
                                            subiterator_type = ScheduledIterator,
                                            composed_batch = True,
                                            subiterator_keyword_args = {"sr_ratios": sr_schedule_ratio,
                                                                        "sr_cap":sr_schedule_cap})

    else:
        iterator_training_data = LengthBasedSerialIterator(training_data, mb_size,
                                            nb_of_batch_to_sort=nb_of_batch_to_sort,
                                            sort_key=lambda x: len(x[0]),
                                            repeat=True,
                                                       shuffle=reshuffle_every_epoch)
    
    if scheduled:
        def loss_func(src_batch, tgt_batch, src_mask):
    
            t0 = time.clock()
            (total_loss, total_nb_predictions), attn = encdec(src_batch, tgt_batch, src_mask, raw_loss_info=True,
                                                              noise_on_prev_word=noise_on_prev_word,
                                                              use_previous_prediction=use_previous_prediction,
                                                              mode="train", per_sentence = True)
#             print " loss/nb pred", total_loss.data, total_nb_predictions
            summed_total_loss = F.sum(total_loss)
            avg_loss = summed_total_loss / total_nb_predictions
    
            t1 = time.clock()
            chainer.reporter.report({"forward_time": t1 - t0})
    
            chainer.reporter.report({"mb_loss": summed_total_loss.data})
            chainer.reporter.report({"mb_nb_predictions": total_nb_predictions})
            chainer.reporter.report({"trg_loss": avg_loss.data})
            return total_loss, avg_loss     
    else:
        def loss_func(src_batch, tgt_batch, src_mask):
    
            t0 = time.clock()
            (total_loss, total_nb_predictions), attn = encdec(src_batch, tgt_batch, src_mask, raw_loss_info=True,
                                                              noise_on_prev_word=noise_on_prev_word,
                                                              use_previous_prediction=use_previous_prediction,
                                                              mode="train")
            avg_loss = total_loss / total_nb_predictions
    
            t1 = time.clock()
            chainer.reporter.report({"forward_time": t1 - t0})
    
            chainer.reporter.report({"mb_loss": total_loss.data})
            chainer.reporter.report({"mb_nb_predictions": total_nb_predictions})
            chainer.reporter.report({"trg_loss": avg_loss.data})
            return avg_loss


    if scheduled:
        def convert_mb(mb_raw, device):
            return make_batch_src_tgt(mb_raw, eos_idx=eos_idx, padding_idx=0, gpu=device, volatile="off", need_arg_sort=True)
    else:
        def convert_mb(mb_raw, device):
            return make_batch_src_tgt(mb_raw, eos_idx=eos_idx, padding_idx=0, gpu=device, volatile="off", need_arg_sort=False)


    if scheduled:
        updater = UpdaterScheduledLearning(iterator_training_data, optimizer,
                converter=convert_mb,
                device=gpu,
                      loss_per_example_func=loss_func,
                      need_to_convert_to_variables=False,
                      loss_threshold = sr_threshold)        
    else:
        updater = Updater(iterator_training_data, optimizer,
                converter=convert_mb,
                device=gpu,

                      loss_func=loss_func,
                      need_to_convert_to_variables=False)

    trainer = chainer.training.Trainer(updater, stop_trigger, out=output_dir)
#     trainer.extend(chainer.training.extensions.LogReport(trigger=(10, 'iteration')))
#     trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'iteration', 'trg_loss', "dev_loss", "dev_bleu"]),
#                    trigger = (1, "iteration"))

    if dev_data is not None and not no_report_or_save:
        dev_loss_extension = ComputeLossExtension(dev_data, eos_idx,
                                                  mb_size, gpu, reverse_src, reverse_tgt,
                                                  save_best_model_to=output_files_dict["model_best_loss"],
                                                  observation_name="dev_loss", config_training=config_training)
        trainer.extend(dev_loss_extension, trigger=(report_every, "iteration"))

        dev_bleu_extension = ComputeBleuExtension(dev_data, eos_idx, src_indexer, tgt_indexer,
                                                  output_files_dict["dev_translation_output"],
                                                  output_files_dict["dev_src_output"],
                                                  mb_size, gpu, reverse_src, reverse_tgt,
                                                  save_best_model_to=output_files_dict["model_best"],
                                                  observation_name="dev_bleu", config_training=config_training)

        trainer.extend(dev_bleu_extension, trigger=(report_every, "iteration"))

    if test_data is not None and not no_report_or_save:
        test_loss_extension = ComputeLossExtension(test_data, eos_idx,
                                                   mb_size, gpu, reverse_src, reverse_tgt,
                                                   observation_name="test_loss")
        trainer.extend(test_loss_extension, trigger=(report_every, "iteration"))

        test_bleu_extension = ComputeBleuExtension(test_data, eos_idx, src_indexer, tgt_indexer,
                                                   output_files_dict["test_translation_output"],
                                                   output_files_dict["test_src_output"],
                                                   mb_size, gpu, reverse_src, reverse_tgt,
                                                   observation_name="test_bleu")

        trainer.extend(test_bleu_extension, trigger=(report_every, "iteration"))

    if not no_report_or_save:
        trainer.extend(sample_extension, trigger=(sample_every, "iteration"))

        # trainer.extend(chainer.training.extensions.snapshot(), trigger = (save_ckpt_every, "iteration"))

        trainer.extend(CheckpontSavingExtension(output_files_dict["model_ckpt"], config_training), trigger=(save_ckpt_every, "iteration"))

        trainer.extend(SqliteLogExtension(db_path=output_files_dict["sqlite_db"]))

    trainer.extend(TrainingLossSummaryExtension(trigger=(report_every, "iteration")))

    if config_training.training_management.resume:
        if "model_parameters" not in config_training:
            log.error("cannot find model parameters in config file")
            raise ValueError(
                "Config file do not contain model_parameters section")
        if config_training.model_parameters.type == "snapshot":
            model_filename = config_training.model_parameters.filename
            log.info("resuming from trainer parameters %s" % model_filename)
            serializers.load_npz(model_filename, trainer)

    if trainer_snapshot is not None:
        log.info("loading trainer parameters from %s" % trainer_snapshot)
        serializers.load_npz(trainer_snapshot, trainer)

    try:
        if save_initial_model_to is not None:
            log.info("Saving initial parameters to %s" % save_initial_model_to)
            encdec = trainer.updater.get_optimizer("main").target
            serializers.save_npz(save_initial_model_to, encdec)

        trainer.run()
    except BaseException:
        if not no_report_or_save:
            final_snapshot_fn = output_files_dict["model_final"]
            log.info("Exception met. Trying to save current trainer state to file %s" % final_snapshot_fn)
            serializers.save_npz(final_snapshot_fn, trainer)
#             chainer.training.extensions.snapshot(filename = final_snapshot_fn)(trainer)
            config_session = config_training.copy(readonly=False)
            config_session.add_section("model_parameters", keep_at_bottom="metadata")
            config_session["model_parameters"]["filename"] = final_snapshot_fn
            config_session["model_parameters"]["type"] = "snapshot"
            config_session["model_parameters"]["description"] = "final"
            config_session["model_parameters"]["infos"] = argument_parsing_tools.OrderedNamespace()
            config_session["model_parameters"]["infos"]["iteration"] = trainer.updater.iteration
            config_session.set_metadata_modified_time()
            config_session.save_to(final_snapshot_fn + ".config")
# json.dump(config_session, open(final_snapshot_fn + ".config", "w"),
# indent=2, separators=(',', ': '))
            log.info("Saved trainer snapshot to file %s" % final_snapshot_fn)
        raise
