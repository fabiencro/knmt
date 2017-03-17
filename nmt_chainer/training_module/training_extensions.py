"""training_extensions.py: chainer training extensions."""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import chainer
import chainer.training

import logging
from chainer import serializers
from nmt_chainer.utilities import argument_parsing_tools
import numpy

from nmt_chainer.translation.evaluation import (
    compute_loss_all, translate_to_file, sample_once)

logging.basicConfig()
log = logging.getLogger("rnns:trg_ext")
log.setLevel(logging.INFO)

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
        log.info("%s: %f (current best: %r)" %
     (self.observation_name, dev_loss, self.best_loss))
        chainer.reporter.report({self.observation_name: dev_loss})

        if self.best_loss is None or self.best_loss > dev_loss:
            log.info("loss (%s) improvement: %r -> %r" % (self.observation_name,
                                                          self.best_loss, dev_loss))
            self.best_loss = dev_loss
            if self.save_best_model_to is not None:
                log.info(
    "saving best loss (%s) model to %s" %
     (self.observation_name, self.save_best_model_to,))
                serializers.save_npz(self.save_best_model_to, encdec)
                if self.config_training is not None:
                    config_session = self.config_training.copy(readonly=False)
                    config_session.add_section(
    "model_parameters", keep_at_bottom="metadata")
                    config_session["model_parameters"]["filename"] = self.save_best_model_to
                    config_session["model_parameters"]["type"] = "model"
                    config_session["model_parameters"]["description"] = "best_loss"
                    config_session["model_parameters"]["infos"] = argument_parsing_tools.OrderedNamespace(
                        )
                    config_session["model_parameters"]["infos"]["loss"] = float(
                        dev_loss)
                    config_session["model_parameters"]["infos"]["iteration"] = trainer.updater.iteration
                    config_session.set_metadata_modified_time()
                    config_session.save_to(self.save_best_model_to + ".config")
# json.dump(config_session, open(self.save_best_model_to + ".config",
# "w"), indent=2, separators=(',', ': '))

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
            if self.best_loss is not None and (
    isinstance(
        self.best_loss,
         numpy.ndarray) or self.best_loss.device.id != self.gpu):
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
# control_src_fn = output_files_dict["dev_src_output"] #save_prefix +
# ".test.src.out"
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
            log.info("%s improvement: %r -> %r" %
     (self.observation_name, self.best_bleu, bleu))
            self.best_bleu = bleu
            if self.save_best_model_to is not None:
                log.info(
    "saving best bleu (%s) model to %s" %
     (self.observation_name, self.save_best_model_to,))
                serializers.save_npz(self.save_best_model_to, encdec)
                if self.config_training is not None:
                    config_session = self.config_training.copy(readonly=False)
                    config_session.add_section(
    "model_parameters", keep_at_bottom="metadata")
                    config_session["model_parameters"]["filename"] = self.save_best_model_to
                    config_session["model_parameters"]["type"] = "model"
                    config_session["model_parameters"]["description"] = "best_bleu"
                    config_session["model_parameters"]["infos"] = argument_parsing_tools.OrderedNamespace(
                        )
                    config_session["model_parameters"]["infos"]["bleu_stats"] = str(
                        bleu_stats)
                    config_session["model_parameters"]["infos"]["iteration"] = trainer.updater.iteration
                    config_session.set_metadata_modified_time()
                    config_session.save_to(self.save_best_model_to + ".config")
# json.dump(config_session, open(self.save_best_model_to + ".config",
# "w"), indent=2, separators=(',', ': '))
        else:
            log.info(
    "no bleu (%s) improvement: %f >= %f" %
     (self.observation_name, self.best_bleu, bleu))

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

        mb_avg_loss = float(
    trainer.observation["mb_loss"]) / trainer.observation["mb_nb_predictions"]
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

            avg_training_loss = trainer.observation.get(
                "avg_training_loss", None)
            if avg_training_loss is not None:
                avg_training_loss = float(avg_training_loss)

            infos = (datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"),
                     trainer.observation.get(
    "test_bleu_details", None), trainer.updater.iteration,
                     test_loss,
                     trainer.observation.get("test_bleu", None),
                     dev_loss,
                     trainer.observation.get("dev_bleu", None),
                     None, None,
                     trainer.observation.get("avg_update_time", None), avg_training_loss)
            db_cursor.execute(
    "INSERT INTO exp_data VALUES (?,?,?,?,?,?,?,?,?,?,?)", infos)
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
        config_session.add_section(
    "model_parameters",
     keep_at_bottom="metadata")
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

