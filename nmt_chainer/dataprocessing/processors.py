import nmt_chainer.external_libs.bpe.learn_bpe as learn_bpe
import nmt_chainer.external_libs.bpe.apply_bpe as apply_bpe
import collections
import operator
from collections import OrderedDict

from nmt_chainer.dataprocessing.indexer import Indexer

import logging
import json
import codecs
import itertools
import re
import copy

logging.basicConfig()
log = logging.getLogger("rnns:processors")
log.setLevel(logging.INFO)


def build_index_from_iterable(iterable, voc_limit=None):
    counts = collections.defaultdict(int)
    for num_ex, line in enumerate(iterable):
        for w in line:
            counts[w] += 1

    sorted_counts = sorted(
        counts.items(), key=operator.itemgetter(1), reverse=True)

    res = Indexer()

    for w, _ in sorted_counts[:voc_limit]:
        res.add_word(w, should_be_new=True)
    res.finalize()

    return res


class ApplyToMultiIterator(object):
    def __init__(self, iterable, function, can_iter=False):
        self.iterable = iterable
        self.function = function
        if can_iter:
            self.iterator = iter(self.iterable)

    def __iter__(self):
        return ApplyToMultiIterator(self.iterable, self.function, can_iter=True)

    def next(self):
        elem = self.iterator.next()
        return self.function(elem)


class ApplyToMultiIteratorPair(object):
    def __init__(self, iterable1, iterable2, function, can_iter=False):
        self.iterable1 = iterable1
        self.iterable2 = iterable2

        self.function = function
        if can_iter:
            self.iterator1 = iter(self.iterable1)
            self.iterator2 = iter(self.iterable2)

    def __iter__(self):
        return ApplyToMultiIteratorPair(self.iterable1, self.iterable2, self.function, can_iter=True)

    def next(self):
        elem1 = self.iterator1.next()
        elem2 = self.iterator2.next()
        return self.function(elem1, elem2)


class FileMultiIterator(object):
    def __init__(self, filename, max_nb_ex=None, can_iter=False):
        self.filename = filename
        self.max_nb_ex = max_nb_ex

        if can_iter:
            self.f = codecs.open(self.filename, encoding="utf8")
            self.nb_line_read = 0

    def next(self):
        if self.max_nb_ex is not None and self.nb_line_read >= self.max_nb_ex:
            raise StopIteration()
        line = self.f.readline()
        if len(line) == 0:
            raise StopIteration()
        else:
            #             print self.nb_line_read, line.strip()
            self.nb_line_read += 1
            return line.strip()

    def __iter__(self):
        return FileMultiIterator(self.filename, max_nb_ex=self.max_nb_ex, can_iter=True)


REGISTERED_PROCESSORS = []
NAME_TO_PROCESSOR = {}


def register_processor(processor):
    global REGISTERED_PROCESSORS
    global NAME_TO_PROCESSOR
    REGISTERED_PROCESSORS.append(processor)
    assert processor.processor_name() not in NAME_TO_PROCESSOR
    NAME_TO_PROCESSOR[processor.processor_name()] = processor


def get_processor_from_name(name):
    global NAME_TO_PROCESSOR
    return NAME_TO_PROCESSOR[name]


def registered_processor(cls):
    register_processor(cls)
    return cls


class PreProcessor(object):
    def __init__(self):
        self.is_initialized_ = False

    @classmethod
    def processor_name(cls):
        return cls.__name__

    def __str__(self):
        return "<Processor:%s>" % self.processor_name()

#     def make_new_stat(self):
#         return self.Stats()

    def initialize(self, iterable):
        self.is_initialized_ = True

    def is_initialized(self):
        return self.is_initialized_

    @classmethod
    def make_from_serializable(cls, obj):
        assert obj["type"] == "processor"
        processor_name = obj["name"]
        processor = get_processor_from_name(processor_name)
        return processor.make_from_serializable(obj)

    def to_serializable(self):
        raise NotImplemented()

    @classmethod
    def make_base_serializable_object(cls):
        obj = OrderedDict([
            ("type", "processor"),
            ("name", cls.processor_name()),
            ("rev", 1.0)
        ])
        return obj


class MonoProcessor(PreProcessor):
    def convert(self, sentence):
        raise NotImplemented()

    def apply_to_iterable(self, iterable):
        return ApplyToMultiIterator(iterable, lambda elem: self.convert(elem))
#         for sentence in iterable:
#             yield self.convert(sentence, stats = stats)

    def __add__(self, other):
        assert not isinstance(self, ProcessorChain)
        return ProcessorChain([self]) + other


class BiProcessor(PreProcessor):
    def convert(self, sentence1, sentence2):
        raise NotImplemented

    def deconvert(self, sentence1, sentence2):
        raise NotImplemented

    def src_processor(self):
        raise NotImplemented

    def tgt_processor(self):
        raise NotImplemented

    def apply_to_iterable(self, iterable1, iterable2):
        return ApplyToMultiIteratorPair(iterable1, iterable2, lambda elem1, elem2: self.convert(elem1, elem2))


@registered_processor
class ProcessorChain(MonoProcessor):
    def __init__(self, processor_list=[]):
        self.processor_list = processor_list

#     class Stats(object):
#         def __init__(self, stats_list):
#             self.stats_list = stats_list
#
#         def make_report(self):
#             result = []
#             for num, stats in enumerate(self.stats_list):
#                 if stats.__class__ is PreProcessor.Stats:
#                     continue
#                 result.append(stats.make_report())
#             return "\n".join(result)
#
#         def get_sub_stats(self, num):
#             return self.stats_list[num]

    def __str__(self):
        return "[%s]" % (" -> ".join([str(processor) for processor in self.processor_list]))

    def __add__(self, other):
        if isinstance(other, ProcessorChain):
            return ProcessorChain(self.processor_list + other.processor_list)
        else:
            return ProcessorChain(self.processor_list + [other])

    def make_new_stat(self):
        return self.Stats([p.make_new_stat() for p in self.processor_list])

    def initialize(self, iterable):
        for num_processor, processor in enumerate(self.processor_list):
            processor.initialize(iterable)
            if num_processor < len(self.processor_list) - 1:
                iterable = processor.apply_to_iterable(iterable)

    def deconvert(self, seq):
        for num_processor, processor in enumerate(self.processor_list[::-1]):
            seq = processor.deconvert(seq)
        return seq

    def convert(self, sentence, stats=None):
        for num_processor, processor in enumerate(self.processor_list):
            if stats is not None:
                this_stats = stats.get_sub_stats(num_processor)
            else:
                this_stats = None
#             print num_processor, sentence, processor, this_stats, this_stats.make_report()
            sentence = processor.convert(sentence, stats=this_stats)
        return sentence

    def is_initialized(self):
        all_initialized = all(processor.is_initialized()
                              for processor in self.processor_list)
        any_initialized = any(processor.is_initialized()
                              for processor in self.processor_list)
        if all_initialized != any_initialized:
            raise AssertionError(
                repr([processor.is_initialized() for processor in self.processor_list]))
        return all_initialized

    @staticmethod
    def processor_name():
        return "processor_chain"

    @classmethod
    def make_from_serializable(cls, obj):
        res = ProcessorChain()
        res.processor_list = [PreProcessor.make_from_serializable(
            subobj) for subobj in obj["processors_list"]]
        return res

    def to_serializable(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["processors_list"] = [processor.to_serializable()
                                  for processor in self.processor_list]
        return obj


@registered_processor
class BiProcessorChain(BiProcessor):
    def __init__(self):
        self.processors_list = []
        self.src_processor_ = ProcessorChain()
        self.tgt_processor_ = ProcessorChain()

    def __str__(self):
        return "[%s]" % (" -> ".join(["[%s]%s" % (channel, str(processor)) for (channel, processor) in self.processors_list]))

    def is_initialized(self):
        all_initialized = all(processor.is_initialized()
                              for channel, processor in self.processors_list)
        any_initialized = any(processor.is_initialized()
                              for channel, processor in self.processors_list)
        if all_initialized != any_initialized:
            raise AssertionError(repr(
                [processor.is_initialized() for channel, processor in self.processors_list]))
        return all_initialized

    def initialize(self, iterable1, iterable2):
        for num_processor, (channel, processor) in enumerate(
                self.processors_list):
            if channel == "src":
                processor.initialize(iterable1)
                if num_processor < len(self.processors_list) - 1:
                    iterable1 = processor.apply_to_iterable(iterable1)
            elif channel == "tgt":
                processor.initialize(iterable2)
                if num_processor < len(self.processors_list) - 1:
                    iterable2 = processor.apply_to_iterable(iterable2)
            elif channel == "all":
                processor.initialize(iterable1, iterable2)
                if num_processor < len(self.processors_list) - 1:
                    iterable1, iterable2 = processor.apply_to_iterable(iterable1, iterable2)

    def convert(self, sentence1, sentence2):
        for num_processor, (channel, processor) in enumerate(
                self.processors_list):
            if channel == "src":
                sentence1 = processor.convert(sentence1)
            elif channel == "tgt":
                sentence2 = processor.convert(sentence2)
            elif channel == "all":
                sentence1, sentence2 = processor.convert(sentence1, sentence2)
        return sentence1, sentence2

    def deconvert(self, sentence1, sentence2):
        for num_processor, (channel, processor) in enumerate(
                self.processors_list[::-1]):
            if channel == "src":
                sentence1 = processor.deconvert(sentence1)
            elif channel == "tgt":
                sentence2 = processor.deconvert(sentence2)
            elif channel == "all":
                sentence1, sentence2 = processor.deconvert(
                    sentence1, sentence2)
        return sentence1, sentence2

    def src_processor(self):
        return self.src_processor_

    def tgt_processor(self):
        return self.tgt_processor_

    def add_src_processor(self, processor):
        assert isinstance(processor, PreProcessor)
        self.processors_list.append(("src", processor))
        self.src_processor_ += processor

    def add_tgt_processor(self, processor):
        assert isinstance(processor, PreProcessor)
        self.processors_list.append(("tgt", processor))
        self.tgt_processor_ += processor

    def add_biprocessor(self, processor):
        assert isinstance(processor, BiProcessor)
        self.processors_list.append(("all", processor))
        src_processor = processor.src_processor()
        if src_processor is not None:
            self.src_processor_ += src_processor

        tgt_processor = processor.tgt_processor()
        if tgt_processor is not None:
            self.tgt_processor_ += tgt_processor

    @classmethod
    def make_from_serializable(cls, obj):
        res = BiProcessorChain()
        for ch, processor in [[channel, PreProcessor.make_from_serializable(subobj)] for channel, subobj in obj["processors_list"]]:
            if ch == "src":
                res.add_src_processor(processor)
            elif ch == "tgt":
                res.add_tgt_processor(processor)
            else:
                res.add_biprocessor(processor)
        return res

    def to_serializable(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["processors_list"] = [[channel, processor.to_serializable()]
                                  for channel, processor in self.processors_list]
        return obj


@registered_processor
class JointBPEBiProcessor(BiProcessor):
    def __init__(self, bpe_data_file, symbols=10000, min_frequency=2, separator="@@"):
        self.bpe_processor = BPEProcessing(bpe_data_file, symbols=symbols, min_frequency=min_frequency, separator=separator)

    def initialize(self, iterable1, iterable2):
        self.bpe_processor.initialize(itertools.chain(iterable1, iterable2))

    def convert(self, sentence1, sentence2):
        return self.bpe_processor.convert(sentence1), self.bpe_processor.convert(sentence2)

    def deconvert(self, sentence1, sentence2):
        return self.bpe_processor.deconvert(sentence1), self.bpe_processor.deconvert(sentence2)

    def src_processor(self):
        return self.bpe_processor

    def tgt_processor(self):
        return self.bpe_processor

    def is_initialized(self):
        return self.bpe_processor.is_initialized()

    @classmethod
    def make_from_serializable(cls, obj):
        bpe_processor = PreProcessor.make_from_serializable(obj["bpe_processor"])
        res = JointBPEBiProcessor(bpe_data_file="dummy")
        res.bpe_processor = bpe_processor
        return res

    def to_serializable(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["bpe_processor"] = self.bpe_processor.to_serializable()
        return obj


@registered_processor
class BPEProcessing(MonoProcessor):
    def __init__(self, bpe_data_file, symbols=10000, min_frequency=2, separator="@@"):
        self.bpe_data_file = bpe_data_file
        self.symbols = symbols
        self.min_frequency = min_frequency
        self.separator = separator
        self.is_initialized_ = False

    def load_bpe(self):
        log.info("loading BPE data from %s", self.bpe_data_file)
        with codecs.open(self.bpe_data_file, "r", encoding="utf8") as codes:
            self.bpe = apply_bpe.BPE(codes, self.separator)
        self.is_initialized_ = True

    def initialize(self, iterable):
        log.info("Creating BPE data and saving it to %s", self.bpe_data_file)
        with codecs.open(self.bpe_data_file, "w", encoding="utf8") as output:
            learn_bpe.learn_bpe_from_sentence_iterable(iterable, output=output,
                                                       symbols=self.symbols,
                                                       min_frequency=self.min_frequency,
                                                       verbose=False)
        self.load_bpe()

    def __str__(self):
        if self.is_initialized():
            return "bpe<%i-%s>" % (self.symbols, self.bpe_data_file)
        else:
            return "bpe<%i>" % (self.symbols)

    def convert(self, sentence, stats=None):
        assert self.is_initialized()
#         print sentence
#         print "->"
        converted = self.bpe.segment_splitted(sentence)
#         print converted
        return converted

    def deconvert(self, seq):
        res = []
        merge_to_previous = False
        for position, w in enumerate(seq):
            if len(w) >= len(self.separator) and w[-len(self.separator):] == self.separator and position != (len(seq) - 1):
                has_separator = True
                w = w[:-len(self.separator)]
            else:
                has_separator = False

            if merge_to_previous:
                res[-1] = res[-1] + w
            else:
                res.append(w)

            merge_to_previous = has_separator
        return res

    @staticmethod
    def processor_name():
        return "bpe"

    @classmethod
    def make_from_serializable(cls, obj):
        res = BPEProcessing(obj["bpe_data_file"], symbols=obj["symbols"],
                            min_frequency=obj["min_frequency"], separator=obj["separator"])
        res.load_bpe()
        return res

    def to_serializable(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["bpe_data_file"] = self.bpe_data_file
        obj["separator"] = self.separator
        obj["symbols"] = self.symbols
        obj["min_frequency"] = self.min_frequency
        return obj


def segment(line, type="word"):
    if type == "word":
        return line.split(" ")
    elif type == "word2char":
        return tuple("".join(line.split(" ")))
    elif type == "char":
        return tuple(line)
    else:
        raise NotImplemented


@registered_processor
class SimpleSegmenter(MonoProcessor):
    def __init__(self, type="word"):
        assert type in "word char word2char".split()
        self.type = type
        self.is_initialized_ = False

    def convert(self, sentence, stats=None):
        return segment(sentence, type=self.type)

    def deconvert(self, seq):
        if self.type == "word":
            return " ".join(seq)
        elif self.type == "char":
            return "".join(seq)
        elif self.type == "word2char":
            return "".join(seq)
        else:
            raise ValueError()

    @staticmethod
    def processor_name():
        return "simple_segmenter"

    def __str__(self):
        return "segmenter<%s>" % self.type

    @classmethod
    def make_from_serializable(cls, obj):
        res = SimpleSegmenter()
        res.type = obj["segmentation_type"]
        res.is_initialized_ = True
        return res

    def to_serializable(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["segmentation_type"] = self.type
        return obj


@registered_processor
class LatinScriptProcess(MonoProcessor):

    CAP_CHAR = u"\u203B"
    ALL_CAPS_CHAR = u"\u203C"
    SUFFIX_CHAR = u"\u203F"

    def __init__(self, mode="all_adjoint"):
        assert mode in "all_adjoint caps_isolate".split()
        self.mode = mode
        self.is_initialized_ = False

    def convert_caps(self, w):
        if self.CAP_CHAR in w or self.ALL_CAPS_CHAR in w:
            raise ValueError("Special char in word")
        assert len(w) > 0
        if w.istitle():
            return self.CAP_CHAR + w.lower()
        elif w.isupper():
            return self.ALL_CAPS_CHAR + w.lower()
        else:
            return w

    def deconvert_caps(self, w):
        if w.startswith(self.CAP_CHAR):
            assert not w[1:].istitle()
            return w[1:].title()  # w[1].upper() + w[2:]
        elif w.startswith(self.ALL_CAPS_CHAR):
            #             assert w[1:].islower()
            return w[1:].upper()
        else:
            return w

    def convert_caps_alt(self, w):
        if self.CAP_CHAR in w or self.ALL_CAPS_CHAR in w or " " in w:
            raise ValueError("Special char in word")
        assert len(w) > 0
        if w.istitle():
            return self.CAP_CHAR + " " + w.lower()
        elif w.isupper():
            return self.ALL_CAPS_CHAR + " " + w.lower()
        else:
            return w

    def deconvert_caps_alt_sentence(self, sentence):
        res = []
        next_is_cap = False
        next_is_all_caps = False
        for w in sentence.split(" "):
            if w == self.CAP_CHAR:
                next_is_cap = True
                continue
            if w == self.ALL_CAPS_CHAR:
                next_is_all_caps = True
                continue
            if next_is_all_caps:
                w = w.upper()
            elif next_is_cap:
                w = w.title()
            next_is_cap = False
            next_is_all_caps = False
            res.append(w)
        return " ".join(res)

    def convert_punct_word(self, w):
        if self.SUFFIX_CHAR in w:
            raise ValueError("Special char in word")
        if len(w) > 3 and w.endswith("..."):
            w = w[:-3] + " " + self.SUFFIX_CHAR + "..."
        else:
            if len(w) > 1:
                for punct in ".!?:,;\"%$'`)]":
                    if w.endswith(punct):
                        w = w[:-1] + " " + self.SUFFIX_CHAR + punct
                        break
        return w

    def convert_punct_inside(self, w):
        if self.SUFFIX_CHAR in w:
            raise ValueError("Special char in word")

        if w == "...":
            return w
        ends_with_triple_dots = False
        if len(w) > 3 and w.endswith("..."):
            ends_with_triple_dots = True
            w = w[:-3]

        if len(w) > 1:
            for punct in ".!?:,;\"%$'`)]-([&=+*<>_/\\^~#@|":
                splitted = w.split(punct)
                if len(splitted[0]) == 0:
                    splitted = [punct + splitted[1]] + splitted[2:]  # do not split if punc at the beginning
                w = (" " + self.SUFFIX_CHAR + punct).join(splitted)
        if ends_with_triple_dots:
            w = w + " " + self.SUFFIX_CHAR + "..."
        return w

    def deconvert_punct_sentence(self, sentence):
        res = []
        for w in sentence.split(" "):
            if len(res) > 0 and w.startswith(self.SUFFIX_CHAR):
                res[-1] = res[-1] + w[1:]
            else:
                res.append(w)
        return " ".join(res)

    def convert(self, sentence, stats=None):
        sentence = re.sub("\s+", " ", sentence)
        sentence = " ".join(self.convert_punct_inside(w)
                            for w in sentence.split(" "))
        if self.mode == "all_adjoint":
            sentence = " ".join(self.convert_caps(w)
                                for w in sentence.split(" "))
        elif self.mode == "caps_isolate":
            sentence = " ".join(self.convert_caps_alt(w)
                                for w in sentence.split(" "))
        return sentence

    def deconvert(self, sentence):
        if self.mode == "all_adjoint":
            sentence = " ".join(self.deconvert_caps(w)
                                for w in sentence.split(" "))
        elif self.mode == "caps_isolate":
            sentence = self.deconvert_caps_alt_sentence(sentence)
        sentence = self.deconvert_punct_sentence(sentence)
        return sentence

    @staticmethod
    def processor_name():
        return "latin_script_processor"

    def __str__(self):
        return "latin_script_processor"

    @classmethod
    def make_from_serializable(cls, obj):
        res = LatinScriptProcess()
        res.mode = obj.get("mode", "all_adjoint")
        res.is_initialized_ = True
        return res

    def to_serializable(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["mode"] = self.mode
        return obj


#     class IterableIterator(object):
#         def __init__(self, filename):
#             self.filename = filename
#         def make_iter(self):
#
#
#     def __iter__(self):
#         with codecs.open(self.filename, encoding="utf8") as f:
#             for num_line, line in enumerate(f):
# #                 print self.filename, num_line, self.max_nb_ex
#                 if self.max_nb_ex is not None and num_line >= self.max_nb_ex:
#                     return
#                 yield line.strip()

class IndexingPrePostProcessorBase(MonoProcessor):
    def __len__(self):
        raise NotImplemented

    def is_unk_idx(self, idx):
        raise NotImplemented

    def convert(self, sentence, stats=None):
        raise NotImplemented

    def convert_swallow(self, sentence, stats=None):
        raise NotImplemented

    def deconvert_swallow(self, seq, unk_tag="#UNK#", no_oov=True, eos_idx=None):
        raise NotImplemented

    def deconvert_post(self, seq):
        raise NotImplemented

    def deconvert(self, seq, unk_tag="#UNK#", no_oov=True, eos_idx=None):
        sentence = self.deconvert_swallow(
            seq, unk_tag=unk_tag, no_oov=no_oov, eos_idx=eos_idx)
        sentence = self.deconvert_post(sentence)
        return sentence

    class Stats(object):
        def make_report(self):
            return "nothing to report"

        def report_as_obj(self):
            return OrderedDict()

    def make_new_stat(self):
        return self.Stats()


@registered_processor
class BiIndexingPrePostProcessor(BiProcessor):
    def __init__(self, voc_limit1=None, voc_limit2=None):
        self.indexer1 = IndexingPrePostProcessor(voc_limit1)
        self.indexer2 = IndexingPrePostProcessor(voc_limit2)
        self.preprocessor = None
        self.is_initialized_ = False

    def convert(self, sentence1, sentence2, stat1=None, stat2=None):
        if self.preprocessor is not None:
            sentence1, sentence2 = self.preprocessor.convert(sentence1, sentence2)
        return self.indexer1.convert_swallow(sentence1, stat1), self.indexer2.convert_swallow(sentence2, stat2)

    def deconvert(self, sentence1, sentence2):
        sentence1, sentence2 = self.indexer1.deconvert_swallow(sentence1), self.indexer2.deconvert_swallow(sentence2)
        if self.preprocessor is not None:
            sentence1, sentence2 = self.preprocessor.deconvert(
                sentence1, sentence2)
        return sentence1, sentence2

    def __str__(self):
        return """BiIndexing:
    src: %s tgt: %s
    pp: %s""" % (self.indexer1, self.indexer2, self.preprocessor)

    def src_processor(self):
        return self.indexer1

    def tgt_processor(self):
        return self.indexer2

    def make_new_stat(self):
        return self.indexer1.make_new_stat(), self.indexer2.make_new_stat()

    def initialize(self, iterable1, iterable2):
        if self.preprocessor is not None:
            self.preprocessor.initialize(iterable1, iterable2)
            iterable_1_2 = self.preprocessor.apply_to_iterable(
                iterable1, iterable2)
            iterable1 = ApplyToMultiIterator(iterable_1_2, lambda x: x[0])
            iterable2 = ApplyToMultiIterator(iterable_1_2, lambda x: x[1])

        self.indexer1.initialize_swallow(iterable1)
        self.indexer2.initialize_swallow(iterable2)
        self.is_initialized_ = True

    def add_preprocessor(self, processor, can_be_initialized=False):
        if not can_be_initialized and self.is_initialized():
            raise ValueError("adding preprocessor to initialized biindexer")
        self.preprocessor = processor
        self.indexer1.add_preprocessor(processor.src_processor(), can_be_initialized=can_be_initialized)
        self.indexer2.add_preprocessor(processor.tgt_processor(), can_be_initialized=can_be_initialized)

    @classmethod
    def make_from_serializable(cls, obj):
        if isinstance(obj, list):
            if len(obj) != 2:
                raise ValueError("unrecognized preprocessor format")
            res = BiIndexingPrePostProcessor()
            obj_src, obj_tgt = obj
            indexer1 = IndexingPrePostProcessor.make_from_serializable(obj_src)
            indexer2 = IndexingPrePostProcessor.make_from_serializable(obj_tgt)

            res.indexer1 = indexer1
            res.indexer2 = indexer2

            if indexer1.preprocessor is not None or indexer2.preprocessor is not None:
                preprocessor = BiProcessorChain()
                if indexer1.preprocessor is not None:
                    preprocessor.add_src_processor(indexer1.preprocessor)
                    indexer1.preprocessor = None
                if indexer2.preprocessor is not None:
                    preprocessor.add_tgt_processor(indexer2.preprocessor)
                    indexer2.preprocessor = None
                res.add_preprocessor(preprocessor, can_be_initialized=True)
        else:
            res = BiIndexingPrePostProcessor()
            res.indexer1 = PreProcessor.make_from_serializable(
                obj["indexer_src"])
            res.indexer2 = PreProcessor.make_from_serializable(
                obj["indexer_tgt"])
            assert isinstance(res.indexer1, IndexingPrePostProcessorBase)
            assert isinstance(res.indexer2, IndexingPrePostProcessorBase)
            if "preprocessor" in obj:
                preprocessor = PreProcessor.make_from_serializable(obj["preprocessor"])
                res.add_preprocessor(preprocessor, can_be_initialized=True)
            else:
                res.preprocessor = None
            res.is_initialized_ = True
        return res

    def to_serializable(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["indexer_src"] = self.indexer1.to_serializable_swallow()
        obj["indexer_tgt"] = self.indexer2.to_serializable_swallow()
        if self.preprocessor is not None:
            obj["preprocessor"] = self.preprocessor.to_serializable()
        return obj


@registered_processor
class IndexingPrePostProcessor(IndexingPrePostProcessorBase):
    def __init__(self, voc_limit=None):
        self.voc_limit = voc_limit
#         self.save_index_to = save_index_to
        self.is_initialized_ = False
        self.preprocessor = None

    class Stats(object):
        def __init__(self):
            self.unk_cnt = 0
            self.token = 0
            self.nb_ex = 0

        def update(self, unk_cnt, token, nb_ex):
            #             print unk_cnt, token, nb_ex, self.unk_cnt, self.token, self.nb_ex
            self.unk_cnt += unk_cnt
            self.token += token
            self.nb_ex += nb_ex

        def make_report(self):
            report = "#tokens: %i   of which %i (%f%%) are unknown" % (self.token,
                                                                       self.unk_cnt,
                                                                       (self.unk_cnt * 100.0) / self.token if self.token != 0 else 0
                                                                       )
            return report

        def report_as_obj(self):
            return OrderedDict([
                ("nb_tokens", self.token),
                ("nb_unk", self.unk_cnt),
                ("unknown_percent", (self.unk_cnt * 100.0) / self.token if self.token != 0 else 0)
            ])

    def initialize_swallow(self, iterable):
        log.info("building dic")
        self.indexer = build_index_from_iterable(iterable, self.voc_limit)
        self.is_initialized_ = True

    def initialize(self, iterable):
        if self.preprocessor is not None:
            self.preprocessor.initialize(iterable)
            iterable = self.preprocessor.apply_to_iterable(iterable)
        self.initialize_swallow(iterable)
        self.indexer = build_index_from_iterable(iterable, self.voc_limit)

    def __str__(self):
        if not self.is_initialized():
            return "indexer<%i>pp[%s]" % (self.voc_limit, "" if self.preprocessor is None else str(self.preprocessor))
        else:
            return "indexer<%i - %i>pp[%s]" % (self.voc_limit, len(self.indexer), "" if self.preprocessor is None else str(self.preprocessor))

    def __len__(self):
        if not self.is_initialized():
            return 0
        else:
            return len(self.indexer)

    def is_initialized(self):
        return self.is_initialized_

    def convert(self, sentence, stats=None):
        if self.preprocessor is not None:
            sentence = self.preprocessor.convert(sentence)
        converted = self.convert_swallow(sentence, stats=stats)
        return converted

    def convert_swallow(self, sentence, stats=None):
        assert self.is_initialized()
        converted = self.indexer.convert(sentence)
        if stats is not None:
            unk_cnt = sum(self.indexer.is_unk_idx(w) for w in converted)
            stats.update(unk_cnt=unk_cnt, token=len(converted), nb_ex=1)
        return converted

    def apply_to_iterable(self, iterable, stats=None):
        raise AssertionError()

    def deconvert_swallow(self, seq, unk_tag="#UNK#", no_oov=True, eos_idx=None):
        return self.indexer.deconvert(seq, unk_tag=unk_tag, no_oov=no_oov, eos_idx=eos_idx)

    def deconvert_post(self, seq):
        if self.preprocessor is not None:
            seq = self.preprocessor.deconvert(seq)
        return seq

    def is_unk_idx(self, idx):
        return self.indexer.is_unk_idx(idx)

    @staticmethod
    def processor_name():
        return "indexer"

    @classmethod
    def make_serialized_form_compatible_with_newer_version(cls, obj):
        if Indexer.check_if_data_indexer(obj):
            new_obj = cls.make_base_serializable_object()
            new_obj["indexer"] = obj
            new_obj["voc_limit"] = len(obj)
            ss = SimpleSegmenter("word")
            ss.initialize(None)
            new_obj["preprocessor"] = ss.to_serializable()

        elif "processors_list" in obj:
            new_obj = obj["processors_list"][-1]
            if len(obj["processors_list"]) > 1:
                preproc_obj = copy.deepcopy(obj)
                preproc_obj["processors_list"] = preproc_obj["processors_list"][:-1]
                new_obj["preprocessor"] = preproc_obj
        else:
            new_obj = obj
        return new_obj

    @classmethod
    def make_from_serializable(cls, obj):
        obj = cls.make_serialized_form_compatible_with_newer_version(obj)
        res = IndexingPrePostProcessor(voc_limit=obj["voc_limit"])
        res.indexer = Indexer.make_from_serializable(obj["indexer"])
        if "preprocessor" in obj:
            res.preprocessor = PreProcessor.make_from_serializable(
                obj["preprocessor"])
        res.is_initialized_ = True
        return res

    def to_serializable(self):
        obj = self.to_serializable_swallow()
        if self.preprocessor is not None:
            obj["preprocessor"] = self.preprocessor.to_serializable()
        return obj

    def to_serializable_swallow(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["voc_limit"] = self.voc_limit
#         obj["voc_fn"] = self.save_index_to
        obj["indexer"] = self.indexer.to_serializable()
        return obj

    @classmethod
    def make_from_indexer(cls, indexer):
        res = IndexingPrePostProcessor(voc_limit=len(indexer))
        res.indexer = indexer
        res.is_initialized_ = True
        return res

    def add_preprocessor(self, processor, can_be_initialized=False):
        if not can_be_initialized and self.is_initialized():
            raise ValueError("adding procesor to initialized indexer")
        self.preprocessor = processor


def izip_must_equal(it1, it2):
    for s1, s2 in itertools.izip_longest(it1, it2, fillvalue=None):
        if s1 is None or s2 is None:
            raise ValueError("iterators have different sizes")
        yield s1, s2


def build_dataset_pp(src_fn, tgt_fn, bi_idx, max_nb_ex=None):
    #                   src_voc_limit=None, tgt_voc_limit=None, max_nb_ex=None, dic_src=None, dic_tgt=None,
    #                   tgt_segmentation_type="word", src_segmentation_type="word"):

    src = FileMultiIterator(src_fn, max_nb_ex=max_nb_ex)
    tgt = FileMultiIterator(tgt_fn, max_nb_ex=max_nb_ex)

    if not bi_idx.is_initialized():
        bi_idx.initialize(src, tgt)
#
#     if not src_pp.is_initialized():
#         log.info("building src_dic")
#         src_pp.initialize(src)
#
#     if not tgt_pp.is_initialized():
#         log.info("building tgt_dic")
#         tgt_pp.initialize(tgt)
#
#     print src_pp
#
#     print tgt_pp

    print bi_idx

    stats_src, stats_tgt = bi_idx.make_new_stat()

    log.info("start indexing")

    res = []

    for sentence_src, sentence_tgt in izip_must_equal(src, tgt):
        #         print len(sentence_tgt), len(sentence_src)
        seq_src, seq_tgt = bi_idx.convert(sentence_src, sentence_tgt, stats_src, stats_tgt)
        res.append((seq_src, seq_tgt))

    return res, stats_src, stats_tgt


def build_dataset_one_side_pp(src_fn, src_pp, max_nb_ex=None):

    src = FileMultiIterator(src_fn, max_nb_ex=max_nb_ex)
    print src_pp
    if not src_pp.is_initialized():
        log.info("building src_dic")
        src_pp.initialize(src)

    stats_src = src_pp.make_new_stat()
    log.info("start indexing")

    res = []

    for sentence_src in src:
        #         print len(sentence_tgt), len(sentence_src)
        seq_src = src_pp.convert(sentence_src, stats=stats_src)
        res.append(seq_src)

    return res, stats_src


# def load_pp_from_data(data):
#     if Indexer.check_if_data_indexer(data):
#         indexer = Indexer.make_from_serializable(data)
#         ipp = IndexingPrePostProcessor.make_from_indexer(indexer)
#         ss_pp = SimpleSegmenter("word")
#         ss_pp.initialize(None)
#         pp = ProcessorChain([ss_pp,
#                                  ipp])
#     else:
#         pp = PreProcessor.make_from_serializable(data)
#
#     return pp

def load_pp_pair_from_file(filename):
    bi_idx = BiIndexingPrePostProcessor.make_from_serializable(
        json.load(open(filename)))
    log.info("loading bilingual preprocessor from file %s", filename)
    log.info(str(bi_idx))
#     print bi_idx.src_processor()
#     print bi_idx.tgt_processor()
    return bi_idx


def save_pp_pair_to_file(bi_idx, filename):
    json.dump(bi_idx.to_serializable(), open(filename, "w"), indent=2, separators=(',', ': '))
