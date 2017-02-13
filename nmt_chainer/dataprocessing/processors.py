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
    def convert(self, sentence, stats = None):
        raise NotImplemented()
    
    class Stats(object):
        def make_report(self):
            return "Nothing to Report"
    
    @classmethod
    def processor_name(cls):
        raise NotImplemented()
    
    def __str__(self):
        return "<Processor:%s>"%self.processor_name()
    
    def __len__(self):
        raise NotImplemented()
    
    def apply_to_iterable(self, iterable, stats = None):
        for sentence in iterable:
            yield self.convert(sentence, stats = stats)
        
    def make_new_stat(self):
        return self.Stats()
        
    @classmethod
    def make_from_serializable(cls, obj):
        assert obj["type"] == "processor"
        processor_name = obj["name"]
        processor = get_processor_from_name(processor_name)
        return processor.make_from_serializable(obj)
        
    def to_serializable(self):
        raise NotImplemented()
        
    def make_base_serializable_object(self):
        assert self.is_initialized()
        obj = OrderedDict([
            ("type", "processor"),
            ("name", self.processor_name()),
            ("rev", 1.0)
            ])
        return obj
    
    def __add__(self, other):
        assert not isinstance(self, ProcessorChain)
        return ProcessorChain([self]) + other
        
@registered_processor        
class ProcessorChain(PreProcessor):
    def __init__(self, processor_list = []):
        self.processor_list = processor_list
     
    class Stats(object):
        def __init__(self, stats_list):
            self.stats_list = stats_list
            
        def make_report(self):
            result = []
            for num, stats in enumerate(self.stats_list):
                if stats.__class__ is PreProcessor.Stats:
                    continue
                result.append(stats.make_report())
            return "\n".join(result)
            
        def get_sub_stats(self, num):
            return self.stats_list[num]
    
    def __str__(self):
        return "[%s]"%(" -> ".join([str(processor) for processor in self.processor_list]))
    
    def __len__(self):
        return len(self.processor_list[-1])
    
    def __add__(self, other):
        if isinstance(other, ProcessorChain):
            return ProcessorChain(self.processor_list + other.processor_list)
        else:
            return ProcessorChain(self.processor_list + [other])
    
    def make_new_stat(self):
        return self.Stats([p.make_new_stat() for p in self.processor_list])     
   
    def initialize(self, iterable):
        for processor in self.processor_list:
            processor.initialize(iterable)
            iterable = processor.apply_to_iterable(iterable)
       
    def deconvert(self, seq, unk_tag="#UNK#", no_oov=True, eos_idx=None):
        for num_processor, processor in enumerate(self.processor_list[::-1]):
            if num_processor == 0:
                seq = processor.deconvert(seq, unk_tag=unk_tag, no_oov=no_oov, eos_idx=eos_idx)
            else:
                seq = processor.deconvert(seq)
        return seq
    
    def deconvert_swallow(self, seq, unk_tag="#UNK#", no_oov=True, eos_idx=None):
        return self.processor_list[-1].deconvert(seq, unk_tag=unk_tag, no_oov=no_oov, eos_idx=eos_idx)
            
    def deconvert_post(self, seq, unk_tag="#UNK#", no_oov=True, eos_idx=None):
        for num_processor, processor in enumerate(self.processor_list[::-1]):
            if num_processor == 0:
                pass
            else:
                seq = processor.deconvert(seq)
        return seq     
            
    def is_unk_idx(self, idx):
        return self.processor_list[-1].is_unk_idx(idx)
            
    def convert(self, sentence, stats = None):
        for num_processor, processor in enumerate(self.processor_list):
            if stats is not None:
                this_stats = stats.get_sub_stats(num_processor)
            else:
                this_stats = None
#             print num_processor, sentence, processor, this_stats, this_stats.make_report()
            sentence = processor.convert(sentence, stats = this_stats) 
        return sentence
    
    def is_initialized(self):
        all_initialized = all(processor.is_initialized() for processor in self.processor_list)
        any_initialized = any(processor.is_initialized() for processor in self.processor_list)
        if all_initialized != any_initialized:
            raise AssertionError(repr([processor.is_initialized() for processor in self.processor_list]))
        return all_initialized
    
    @staticmethod
    def processor_name():
        return "processor_chain"
    
    @classmethod
    def make_from_serializable(cls, obj):
        res = ProcessorChain()
        res.processor_list = [PreProcessor.make_from_serializable(subobj) for subobj in obj["processors_list"]]
        return res
    
    def to_serializable(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["processors_list"] = [processor.to_serializable() for processor in self.processor_list]
        return obj
        
@registered_processor
class BPEProcessing(PreProcessor):
    def __init__(self, bpe_data_file, symbols = 10000, min_frequency = 2, separator = "@@"):
        self.bpe_data_file = bpe_data_file
        self.symbols = symbols
        self.min_frequency = min_frequency
        self.separator = separator
        self.is_initialized_ = False
       
    def load_bpe(self):
        log.info("loading BPE data from %s", self.bpe_data_file)
        with codecs.open(self.bpe_data_file, "r", encoding = "utf8") as codes:
            self.bpe = apply_bpe.BPE(codes, self.separator)
        self.is_initialized_ = True
                
    def initialize(self, iterable):
        log.info("Creating BPE data and saving it to %s", self.bpe_data_file)
        with codecs.open(self.bpe_data_file, "w", encoding = "utf8") as output:
            learn_bpe.learn_bpe_from_sentence_iterable(iterable, output = output, 
                                                   symbols = self.symbols, 
                                                   min_frequency = self.min_frequency,
                                                   verbose = True)
        self.load_bpe()

    def __str__(self):
        if self.is_initialized():
            return "bpe<%i-%s>"%(self.voc_limit, self.bpe_data_file)
        else:
            return "bpe<%i>"%(self.symbols)

    def is_initialized(self):
        return self.is_initialized_
    
    def convert(self, sentence, stats = None):
        assert self.is_initialized()
        converted = self.bpe.segment_splitted(sentence)
        return converted
        
    def deconvert(self, seq):
        res = []
        merge_to_previous = False
        for position, w in enumerate(seq):
            if len(w) >= len(self.separator) and w[:-len(self.separator)] == self.separator and position != (len(seq) -1):
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
        assert len(obj) == 1
        res = BPEProcessing(*obj)
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


@registered_processor
class IndexingPrePostProcessor(PreProcessor):
    def __init__(self, voc_limit = None):
        self.voc_limit = voc_limit
#         self.save_index_to = save_index_to
        self.is_initialized_ = False
        
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
        
    def initialize(self, iterable):
        log.info("building dic")
        self.indexer = build_index_from_iterable(iterable, self.voc_limit)
#         if self.save_index_to is not None:
#             log.info("saving dic to %s", self.save_index_to)
#             self.indexer.to_serializable()
#             json.dump(self.indexer.to_serializable(),
#               open(self.save_index_to, "w"), indent=2, separators=(',', ': '))
        self.is_initialized_ = True
        
    def __str__(self):
        if not self.is_initialized():
            return "indexer<%i>"%(self.voc_limit)
        else:
            return "indexer<%i - %i>"%(self.voc_limit, len(self.indexer))
        
    def __len__(self):
        if not self.is_initialized():
            return 0
        else:
            return len(self.indexer)
        
    def is_initialized(self):
        return self.is_initialized_
        
    def convert(self, sentence, stats = None):
        assert self.is_initialized()
        converted = self.indexer.convert(sentence)
        if stats is not None:
            unk_cnt = sum(self.indexer.is_unk_idx(w) for w in converted)
            stats.update(unk_cnt = unk_cnt, token = len(converted), nb_ex = 1)
        return converted
        
    def deconvert(self, seq, unk_tag="#UNK#", no_oov=True, eos_idx=None):
        return self.indexer.deconvert(seq, unk_tag=unk_tag, no_oov=no_oov, eos_idx=eos_idx)
        
    def is_unk_idx(self, idx):
        return self.indexer.is_unk_idx(idx)
        
    @staticmethod
    def processor_name():
        return "indexer"
        
    @classmethod
    def make_from_serializable(cls, obj):
        res = IndexingPrePostProcessor(voc_limit = obj["voc_limit"])
        res.indexer = Indexer.make_from_serializable(obj["indexer"])
        res.is_initialized_ = True
        return res
    
    def to_serializable(self):
        assert self.is_initialized()
        obj = self.make_base_serializable_object()
        obj["voc_limit"] = self.voc_limit
#         obj["voc_fn"] = self.save_index_to
        obj["indexer"] = self.indexer.to_serializable()
        return obj
    
    @classmethod
    def make_from_indexer(cls, indexer):
        res = IndexingPrePostProcessor(voc_limit = len(indexer))
        res.indexer = indexer
        res.is_initialized_ = True
        return res
    

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
class SimpleSegmenter(PreProcessor):
    def __init__(self, type = "word"):
        assert type in "word char word2char".split()
        self.type = type
        self.is_initialized_ = False
        
    def initialize(self, iterable):
        self.is_initialized_ = True
            
    def convert(self, sentence, stats = None):
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
            
    def is_initialized(self):
        return self.is_initialized_
    
    @staticmethod
    def processor_name():
        return "simple_segmenter"
    
    def __str__(self):
        return "segmenter<%s>"%self.type
    
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
   

class FileMultiIterator(object):
    def __init__(self, filename, max_nb_ex = None):
        self.filename = filename
        self.max_nb_ex = max_nb_ex   
        
    def __iter__(self):
        with codecs.open(self.filename, encoding="utf8") as f:
            for num_line, line in enumerate(f):
#                 print self.filename, num_line, self.max_nb_ex
                if self.max_nb_ex is not None and num_line >= self.max_nb_ex:
                    return
                yield line.strip()
      

def izip_must_equal(it1, it2):
    for s1, s2 in itertools.izip_longest(it1, it2, fillvalue = None):
        if s1 is None or s2 is None:
            raise ValueError("iterators have different sizes")
        yield s1, s2
    

def build_dataset_pp(src_fn, tgt_fn, src_pp, tgt_pp, max_nb_ex=None):
#                   src_voc_limit=None, tgt_voc_limit=None, max_nb_ex=None, dic_src=None, dic_tgt=None,
#                   tgt_segmentation_type="word", src_segmentation_type="word"):

    src = FileMultiIterator(src_fn, max_nb_ex = max_nb_ex)
    tgt = FileMultiIterator(tgt_fn, max_nb_ex = max_nb_ex)
    
    if not src_pp.is_initialized():
        log.info("building src_dic")
        src_pp.initialize(src)

    if not tgt_pp.is_initialized():
        log.info("building tgt_dic")
        tgt_pp.initialize(tgt)

    print src_pp
    
    print tgt_pp

    stats_src = src_pp.make_new_stat()
    stats_tgt = tgt_pp.make_new_stat()
    
    log.info("start indexing")

    res = []

    for sentence_src, sentence_tgt in izip_must_equal(src, tgt):
#         print len(sentence_tgt), len(sentence_src)
        seq_src = src_pp.convert(sentence_src, stats = stats_src) 
        seq_tgt = tgt_pp.convert(sentence_tgt, stats = stats_tgt) 
        res.append((seq_src, seq_tgt))

    return res, stats_src, stats_tgt

def build_dataset_one_side_pp(src_fn, src_pp, max_nb_ex=None):

    src = FileMultiIterator(src_fn, max_nb_ex = max_nb_ex)
    
    if not src_pp.is_initialized():
        log.info("building src_dic")
        src_pp.initialize(src)

    stats_src = src_pp.make_new_stat()
    log.info("start indexing")

    res = []

    for sentence_src in src:
#         print len(sentence_tgt), len(sentence_src)
        seq_src = src_pp.convert(sentence_src, stats = stats_src) 
        res.append(seq_src)

    return res, stats_src


    
def load_pp_from_data(data):
    if Indexer.check_if_data_indexer(data):
        indexer = Indexer.make_from_serializable(data)
        ipp = IndexingPrePostProcessor.make_from_indexer(indexer)
        ss_pp = SimpleSegmenter("word")
        ss_pp.initialize(None)
        pp = ProcessorChain([ss_pp, 
                                 ipp])
    else:
        pp = PreProcessor.make_from_serializable(data)
        
    return pp
    
def load_pp_pair_from_file(filename):
    src_data, tgt_data = json.load(open(filename))
    return load_pp_from_data(src_data), load_pp_from_data(tgt_data)

def save_pp_pair_to_file(pp_src, pp_tgt, filename):
    json.dump([pp_src.to_serializable(), pp_tgt.to_serializable()], open(filename, "w"))
    
