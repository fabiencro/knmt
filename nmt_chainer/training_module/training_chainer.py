#!/usr/bin/env python
"""training.py: training procedures."""
from nmt_chainer.utilities import argument_parsing_tools
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
import json

from nmt_chainer.utilities.utils import minibatch_provider, minibatch_provider_curiculum, make_batch_src_tgt
from nmt_chainer.translation.evaluation import (
    compute_loss_all, translate_to_file, sample_once)

import chainer.iterators
import chainer.dataset.iterator
import chainer.training
import chainer.training.extensions

logging.basicConfig()
log = logging.getLogger("rnns:training")
log.setLevel(logging.INFO)


def sent_complexity(sent):
    rank_least_common_word = max(sent)
    length = len(sent)
    return length * math.log(rank_least_common_word + 1)


def example_complexity(ex):
    return sent_complexity(ex[0]) + sent_complexity(ex[1])


import numpy


class SerialIteratorWithPeek(chainer.iterators.SerialIterator):

    def peek(self):
        """
        Return the next batch of data without updating its internal state.
        Several call to peek() should return the same result. A call to next()
        after a call to peek() will return the same result as the previous peek.
        """
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)
        if self._order is None:
            batch = self.dataset[i:i_end]
        else:
            batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    numpy.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch += list(self.dataset[:rest])
                    else:
                        batch += [self.dataset[index]
                                  for index in self._order[:rest]]
        return batch


class LengthBasedSerialIterator(chainer.dataset.iterator.Iterator):
    """
    This iterator will try to return batches with sequences of similar length.
    This is done by first extracting nb_of_batch_to_sort x batch_size sequences, sorting them by size,
    and then successively yielding nb_of_batch_to_sort batches of size batch_size.
    """

    def __init__(self, dataset, batch_size, nb_of_batch_to_sort=20, sort_key=lambda x: len(x[1]),
                 repeat=True, shuffle=True):

        self.sub_iterator = SerialIteratorWithPeek(dataset, batch_size * nb_of_batch_to_sort,
                                                   repeat=repeat, shuffle=shuffle)
        self.dataset = dataset
        self.index_in_sub_batch = 0
        self.sub_batch = None
        self.sort_key = sort_key
        self.batch_size = batch_size
        self.nb_of_batch_to_sort = nb_of_batch_to_sort
        self.repeat = repeat

    def update_sub_batch(self):
        self.sub_batch = list(self.sub_iterator.next())  # copy the result so that we can sort without side effects
        if self.repeat and len(self.sub_batch) != self.batch_size * self.nb_of_batch_to_sort:
            raise AssertionError
        self.sub_batch.sort(key=self.sort_key)
        self.index_in_sub_batch = 0

    def __next__(self):
        if self.sub_batch is None or self.index_in_sub_batch >= self.nb_of_batch_to_sort:
            assert self.sub_batch is None or self.index_in_sub_batch == self.nb_of_batch_to_sort
            self.update_sub_batch()

        minibatch = self.sub_batch[self.index_in_sub_batch * self.batch_size: (self.index_in_sub_batch + 1) * self.batch_size]

        self.index_in_sub_batch += 1

        return minibatch

    def peek(self):
        if self.sub_batch is None or self.index_in_sub_batch >= self.nb_of_batch_to_sort:
            assert self.sub_batch is None or self.index_in_sub_batch == self.nb_of_batch_to_sort
            sub_batch = list(self.sub_iterator.peek())  # copy the result so that we can sort without side effects
            sub_batch.sort(key=self.sort_key)
            index_in_sub_batch = 0
        else:
            sub_batch = self.sub_batch
            index_in_sub_batch = self.index_in_sub_batch
        minibatch = sub_batch[index_in_sub_batch * self.batch_size: (index_in_sub_batch + 1) * self.batch_size]
        return minibatch

    next = __next__

    # It is a bit complicated to keep an accurate value for epoch detail. In practice, the beginning of a sub_batch crossing epoch will have its
    # epoch_detail clipped to epoch
    @property
    def epoch_detail(self):
        remaining_sub_batch_lenth = (self.nb_of_batch_to_sort - self.index_in_sub_batch) * self.batch_size
        assert remaining_sub_batch_lenth >= 0
        epoch_discount = remaining_sub_batch_lenth / float(len(self.dataset))
        sub_epoch_detail = self.sub_iterator.epoch_detail
        epoch_detail = sub_epoch_detail - epoch_discount
        epoch_detail = max(epoch_detail, self.epoch)
        return epoch_detail

    # epoch and is_new_epoch are updated as soon as the end of the current
    # sub_batch has reached a new epoch.
    @property
    def epoch(self):
        return self.sub_iterator.epoch

    @property
    def is_new_epoch(self):
        if self.sub_iterator.is_new_epoch:
            assert self.sub_batch is None or self.index_in_sub_batch > 0
            if self.index_in_sub_batch == 1:
                return True
        return False

    # We do not serialize index_in_sub_batch. A deserialized iterator will start from the next sub_batch.
#     def serialize(self, serializer):
#         self.sub_iterator = serializer("sub_iterator", self.sub_iterator)
#         self.sub_iterator.serialize(serializer["sub_iterator"])


import six


def make_collection_of_variables(in_arrays, volatile="off"):
    if isinstance(in_arrays, tuple):
        in_vars = tuple(chainer.variable.Variable(x, volatile=volatile) for x in in_arrays)
    elif isinstance(in_arrays, dict):
        in_vars = {key: chainer.variable.Variable(x, volatile=volatile)
                   for key, x in six.iteritems(in_arrays)}
    else:
        in_vars = chainer.variable.Variable(in_arrays, volatile=volatile)
    return in_vars


class Updater(chainer.training.StandardUpdater):
    def __init__(self, iterator, optimizer, converter=chainer.dataset.convert.concat_examples,
                 device=None, loss_func=None, need_to_convert_to_variables=True):
        super(Updater, self).__init__(iterator, optimizer, converter=converter,
                                      device=device, loss_func=loss_func)
        self.need_to_convert_to_variables = need_to_convert_to_variables

    def update_core(self):
        t0 = time.clock()

        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if self.need_to_convert_to_variables:
            in_arrays = make_collection_of_variables(in_arrays)

        t1 = time.clock()

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)

        t2 = time.clock()
        update_duration = t2 - t0
        mb_preparation_duration = t1 - t0
        optimizer_update_cycle_duration = t2 - t1
        chainer.reporter.report({"update_duration": update_duration,
                                 "mb_preparation_duration": mb_preparation_duration,
                                 "optimizer_update_cycle_duration": optimizer_update_cycle_duration})


class ComputeLossExtension(chainer.training.Extension):
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, data, eos_idx,
                 mb_size, gpu, reverse_src, reverse_tgt,
                 save_best_model_to=None, observation_name="dev_loss", config_training=None):
        self.best_loss = None
        self.save_best_model_to = save_best_model_to
        self.observation_name = observation_name
        self.data = data
        self.eos_idx = eos_idx
        self.mb_size = mb_size
        self.gpu = gpu
        self.reverse_src = reverse_src
        self.reverse_tgt = reverse_tgt
        self.config_training = config_training

    def __call__(self, trainer):
        encdec = trainer.updater.get_optimizer("main").target
        log.info("computing %s" % self.observation_name)
        dev_loss = compute_loss_all(encdec, self.data, self.eos_idx, self.mb_size,
                                    gpu=self.gpu,
                                    reverse_src=self.reverse_src, reverse_tgt=self.reverse_tgt)
        log.info("%s: %f (current best: %r)" % (self.observation_name, dev_loss, self.best_loss))
        chainer.reporter.report({self.observation_name: dev_loss})

        if self.best_loss is None or self.best_loss > dev_loss:
            log.info("loss (%s) improvement: %r -> %r" % (self.observation_name,
                                                          self.best_loss, dev_loss))
            self.best_loss = dev_loss
            if self.save_best_model_to is not None:
                log.info("saving best loss (%s) model to %s" % (self.observation_name, self.save_best_model_to,))
                serializers.save_npz(self.save_best_model_to, encdec)
                if self.config_training is not None:
                    config_session = self.config_training.copy(readonly=False)
                    config_session.add_section("model_parameters", keep_at_bottom="metadata")
                    config_session["model_parameters"]["filename"] = self.save_best_model_to
                    config_session["model_parameters"]["type"] = "model"
                    config_session["model_parameters"]["description"] = "best_loss"
                    config_session["model_parameters"]["infos"] = argument_parsing_tools.OrderedNamespace()
                    config_session["model_parameters"]["infos"]["loss"] = float(dev_loss)
                    config_session["model_parameters"]["infos"]["iteration"] = trainer.updater.iteration
                    config_session.set_metadata_modified_time()
                    config_session.save_to(self.save_best_model_to + ".config")
#                     json.dump(config_session, open(self.save_best_model_to + ".config", "w"), indent=2, separators=(',', ': '))

    def serialize(self, serializer):
        self.best_loss = serializer("best_loss", self.best_loss)
        # Make sure that best_loss is at the right location.
        # After deserialization, the best_loss is
        # instanciated on the CPU instead of the GPU.
        if self.gpu is None:
            pass  # best_loss should be on the cpu memory anyway
#             if isinstance(self.best_loss, cupy.core.ndarray):
#                 self.best_loss = cupy.asnumpy(self.best_loss)
        else:
            import cupy
            if self.best_loss is not None and (isinstance(self.best_loss, numpy.ndarray) or self.best_loss.device.id != self.gpu):
                with cupy.cuda.Device(self.gpu):
                    self.best_loss = cupy.array(self.best_loss)


class ComputeBleuExtension(chainer.training.Extension):
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, data, eos_idx, src_indexer, tgt_indexer,
                 translations_fn, control_src_fn,
                 mb_size, gpu, reverse_src=False, reverse_tgt=False,
                 save_best_model_to=None, observation_name="dev_bleu",
                 nb_steps=50,
                 s_unk_tag=lambda num, utag: "S_UNK_%i" % utag,
                 t_unk_tag=lambda num, utag: "T_UNK_%i" % utag,
                 config_training=None):
        self.best_bleu = None
        self.save_best_model_to = save_best_model_to
        self.observation_name = observation_name
        self.data = data
        self.eos_idx = eos_idx
        self.mb_size = mb_size
        self.gpu = gpu
        self.reverse_src = reverse_src
        self.reverse_tgt = reverse_tgt
        self.s_unk_tag = s_unk_tag
        self.t_unk_tag = t_unk_tag

        self.src_indexer = src_indexer
        self.tgt_indexer = tgt_indexer
        self.nb_steps = nb_steps

        self.translations_fn = translations_fn
        self.control_src_fn = control_src_fn

        self.src_data = [x for x, y in data]
        self.references = [y for x, y in data]

        self.config_training = config_training

    def __call__(self, trainer):
        encdec = trainer.updater.get_optimizer("main").target
#         translations_fn = output_files_dict["dev_translation_output"] #save_prefix + ".test.out"
#         control_src_fn = output_files_dict["dev_src_output"] #save_prefix + ".test.src.out"
        bleu_stats = translate_to_file(encdec, self.eos_idx, self.src_data, self.mb_size,
                                       self.tgt_indexer,
                                       self.translations_fn, test_references=self.references,
                                       control_src_fn=self.control_src_fn,
                                       src_indexer=self.src_indexer, gpu=self.gpu, nb_steps=50,
                                       reverse_src=self.reverse_src, reverse_tgt=self.reverse_tgt,
                                       s_unk_tag=self.s_unk_tag, t_unk_tag=self.t_unk_tag)
        bleu = bleu_stats.bleu()
        chainer.reporter.report({self.observation_name: bleu,
                                 self.observation_name + "_details": repr(bleu)})

        if self.best_bleu is None or self.best_bleu < bleu:
            log.info("%s improvement: %r -> %r" % (self.observation_name, self.best_bleu, bleu))
            self.best_bleu = bleu
            if self.save_best_model_to is not None:
                log.info("saving best bleu (%s) model to %s" % (self.observation_name, self.save_best_model_to,))
                serializers.save_npz(self.save_best_model_to, encdec)
                if self.config_training is not None:
                    config_session = self.config_training.copy(readonly=False)
                    config_session.add_section("model_parameters", keep_at_bottom="metadata")
                    config_session["model_parameters"]["filename"] = self.save_best_model_to
                    config_session["model_parameters"]["type"] = "model"
                    config_session["model_parameters"]["description"] = "best_bleu"
                    config_session["model_parameters"]["infos"] = argument_parsing_tools.OrderedNamespace()
                    config_session["model_parameters"]["infos"]["bleu_stats"] = str(bleu_stats)
                    config_session["model_parameters"]["infos"]["iteration"] = trainer.updater.iteration
                    config_session.set_metadata_modified_time()
                    config_session.save_to(self.save_best_model_to + ".config")
# json.dump(config_session, open(self.save_best_model_to + ".config",
# "w"), indent=2, separators=(',', ': '))
        else:
            log.info("no bleu (%s) improvement: %f >= %f" % (self.observation_name, self.best_bleu, bleu))

    def serialize(self, serializer):
        self.best_bleu = serializer("best_bleu", self.best_bleu)


class TrainingLossSummaryExtension(chainer.training.Extension):
    priority = chainer.training.PRIORITY_EDITOR

    def __init__(self, trigger=(200, 'iteration')):
        self.update_trigger = chainer.training.trigger.get_trigger(trigger)
        self.reset()
#         self.previous_time = None

    def reset(self):
        self.total_loss = 0
        self.total_nb_predictions = 0
        self.total_update_time = 0
        self.nb_observations = 0

    def __call__(self, trainer):
        # accumulate the observations

        mb_avg_loss = float(trainer.observation["mb_loss"]) / trainer.observation["mb_nb_predictions"]
        log.info("E:%i I:%i L:%f U: %.4f = %.4f + %.4f F:%.4f" % (trainer.updater.epoch,
                 trainer.updater.iteration, mb_avg_loss,
                 trainer.observation["update_duration"],
                 trainer.observation["mb_preparation_duration"],
                 trainer.observation["optimizer_update_cycle_duration"],
                 trainer.observation["forward_time"]))

        self.total_loss += trainer.observation["mb_loss"]
        self.total_nb_predictions += trainer.observation["mb_nb_predictions"]
        self.total_update_time += trainer.observation["update_duration"]
        self.nb_observations += 1

        if self.update_trigger(trainer):
            # output the result
            avg_loss = float(self.total_loss) / self.total_nb_predictions
            avg_update_time = self.total_update_time / self.nb_observations
            chainer.reporter.report({"avg_training_loss": avg_loss})
            chainer.reporter.report({"avg_update_time": avg_update_time})
            self.reset()


class SqliteLogExtension(chainer.training.Extension):
    priority = chainer.training.PRIORITY_READER

    def __init__(self, db_path):
        self.db_path = db_path

    def __call__(self, trainer):
        if any((key in trainer.observation)
                for key in "dev_bleu dev_loss test_bleu test_loss avg_training_loss".split()):

            log.info("saving dev results to %s" % (self.db_path))

            import sqlite3
            import datetime
            db_connection = sqlite3.connect(self.db_path)
            db_cursor = db_connection.cursor()
            db_cursor.execute('''CREATE TABLE IF NOT EXISTS exp_data
(date text, bleu_info text, iteration real,
loss real, bleu real,
dev_loss real, dev_bleu real,
valid_loss real, valid_bleu real,
avg_time real, avg_training_loss real)''')

            dev_loss = trainer.observation.get("dev_loss", None)
            if dev_loss is not None:
                dev_loss = float(dev_loss)

            test_loss = trainer.observation.get("test_loss", None)
            if test_loss is not None:
                test_loss = float(test_loss)

            avg_training_loss = trainer.observation.get("avg_training_loss", None)
            if avg_training_loss is not None:
                avg_training_loss = float(avg_training_loss)

            infos = (datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"),
                     trainer.observation.get("test_bleu_details", None), trainer.updater.iteration,
                     test_loss,
                     trainer.observation.get("test_bleu", None),
                     dev_loss,
                     trainer.observation.get("dev_bleu", None),
                     None, None,
                     trainer.observation.get("avg_update_time", None), avg_training_loss)
            db_cursor.execute("INSERT INTO exp_data VALUES (?,?,?,?,?,?,?,?,?,?,?)", infos)
            db_connection.commit()
            db_connection.close()


class CheckpontSavingExtension(chainer.training.Extension):
    priority = chainer.training.PRIORITY_READER

    def __init__(self, save_to, config_training):
        self.save_to = save_to
        self.config_training = config_training

    def __call__(self, trainer):
        log.info("Saving current trainer state to file %s" % self.save_to)
        serializers.save_npz(self.save_to, trainer)
        config_session = self.config_training.copy(readonly=False)
        config_session.add_section("model_parameters", keep_at_bottom="metadata")
        config_session["model_parameters"]["filename"] = self.save_to
        config_session["model_parameters"]["type"] = "snapshot"
        config_session["model_parameters"]["description"] = "checkpoint"
        config_session["model_parameters"]["infos"] = argument_parsing_tools.OrderedNamespace()
        config_session["model_parameters"]["infos"]["iteration"] = trainer.updater.iteration
        config_session.set_metadata_modified_time()
        config_session.save_to(self.save_to + ".config")
# json.dump(config_session, open(self.save_to + ".config", "w"), indent=2,
# separators=(',', ': '))
        log.info("Saved trainer snapshot to file %s" % self.save_to)


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

    @chainer.training.make_extension()
    def sample_extension(trainer):
        encdec = trainer.updater.get_optimizer("main").target
        iterator = trainer.updater.get_iterator("main")
        mb_raw = iterator.peek()

        src_batch, tgt_batch, src_mask = make_batch_src_tgt(mb_raw, eos_idx=eos_idx, padding_idx=0, gpu=gpu, volatile="on", need_arg_sort=False)

        def s_unk_tag(num, utag):
            "S_UNK_%i" % utag

        def t_unk_tag(num, utag):
            "T_UNK_%i" % utag

        sample_once(encdec, src_batch, tgt_batch, src_mask, src_indexer, tgt_indexer, eos_idx,
                    max_nb=20,
                    s_unk_tag=s_unk_tag, t_unk_tag=t_unk_tag)

    iterator_training_data = LengthBasedSerialIterator(training_data, mb_size,
                                                       nb_of_batch_to_sort=nb_of_batch_to_sort,
                                                       sort_key=lambda x: len(x[0]),
                                                       repeat=True,
                                                       shuffle=reshuffle_every_epoch)

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

    def convert_mb(mb_raw, device):
        return make_batch_src_tgt(mb_raw, eos_idx=eos_idx, padding_idx=0, gpu=device, volatile="off", need_arg_sort=False)

    updater = Updater(iterator_training_data, optimizer,
                      converter=convert_mb,
                      # iterator_training_data = chainer.iterators.SerialIterator(training_data, mb_size,
                      # repeat = True,
                      # shuffle = reshuffle_every_epoch)
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
