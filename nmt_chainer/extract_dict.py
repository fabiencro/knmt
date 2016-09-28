# from _dbus_bindings import Int32
import logging
import codecs
import exceptions
from collections import defaultdict

"""extract_dict.py: Simple utility to extract a dictionary from aligned data"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

logging.basicConfig()
log = logging.getLogger("aparse")
log.setLevel(logging.INFO)

def read_one_parse_info_from_file_object(f):
    id_line = f.readline()
    if len(id_line) == 0:
        raise exceptions.EOFError()
    sharp, id_part, score_part = id_line.split()
    assert sharp == "#"
    id_tag, id_ = id_part.split("=")
    assert id_tag == "ID"
    score_tag, score = score_part.split("=")
    assert score_tag == "SCORE"
    score = float(score)
    sentence = []
    while 1:
        line = f.readline().strip()
        if len(line) == 0:
            return id_, sentence
        splitted_line = line.split("\t")
        num_pos = int(splitted_line[0])
        dpnd = int(splitted_line[1])
        word = splitted_line[2]
        assert num_pos == len(sentence)
        sentence.append(word)
        
def read_one_align_info_from_file_object(f):  
    id_line = f.readline()
    if len(id_line) == 0:
        raise exceptions.EOFError()
    id_line = id_line.strip()
    id_data = id_line.split()
    score = 0.0
    if len(id_data) == 3:
        sharp, id_, score = id_data
        score = float(score)
    else:
        sharp, id_ = id_data
    assert sharp == "#"
    align_line = f.readline().strip()
    alignment = []
    for link in align_line.split():
        left, right = link.split("-")
        if left == "" or right == "":
            continue
        left = [int(x) for x in left.split(",")]
        right = [int(x) for x in right.split(",")]
        alignment.append((left, right))
    return id_, score, alignment

def load_aligned_corpus(src_fn, tgt_fn, align_fn, skip_empty_align = True, invert_alignment_links = False):
    src = codecs.open(src_fn, encoding = "utf8")
    tgt = codecs.open(tgt_fn, encoding = "utf8")
    align_f = codecs.open(align_fn, encoding = "utf8")
    
    num_sentence = 0
    while 1:
        try:
            id_src, sentence_src = read_one_parse_info_from_file_object(src)
            id_tgt, sentence_tgt = read_one_parse_info_from_file_object(tgt)
            id_align, score_align, alignment = read_one_align_info_from_file_object(align_f)
        except exceptions.EOFError:
            return
        if skip_empty_align and len(alignment) == 0:
            log.warn("skipping empty alignment %i %s"%(num_sentence, id_align))
            continue
        assert id_src == id_tgt, "%s != %s @%i"%(id_src, id_tgt, num_sentence)
        assert id_src == id_align, "%s != %s @%i"%(id_src, id_align, num_sentence)
        
        if invert_alignment_links:
            inverted_alignment = [(right, left) for (left, right) in alignment]
            alignment = inverted_alignment
        
        yield sentence_src, sentence_tgt, alignment
        num_sentence += 1

def commandline():
    import argparse, operator, json
    parser = argparse.ArgumentParser()
    parser.add_argument("src_fn")
    parser.add_argument("tgt_fn")
    parser.add_argument("align_fn")
    parser.add_argument("dest")
    parser.add_argument("--max_n", type = int)
    parser.add_argument("--invert_alignment_links", default = False, action = "store_true")
    parser.add_argument("--generate_lexical_prob", default = False, action = "store_true")
    parser.add_argument("--count_null_translations", default = False, action = "store_true")
    
    args = parser.parse_args()
    corpus = load_aligned_corpus(args.src_fn, args.tgt_fn, args.align_fn, invert_alignment_links = args.invert_alignment_links)
    
    counter = defaultdict(lambda: defaultdict(int))
    for num, (sentence_src, sentence_tgt, alignment) in enumerate(corpus):
        if args.max_n and num > args.max_n:
            break
        if num%1000 == 0:
            log.info("%i sentences processed"% num)
        unaligned_pos_left = set(range(len(sentence_src))) 
        unaligned_pos_right = set(range(len(sentence_tgt))) 
        for left, right in alignment:
            for spos in left:
                if spos in unaligned_pos_left:
                    unaligned_pos_left.remove(spos)
                ws = sentence_src[spos]
                for tpos in right:
                    if tpos in unaligned_pos_right:
                        unaligned_pos_right.remove(tpos)
                    wt = sentence_tgt[tpos]
                    counter[ws][wt] += 1
                    
        if args.count_null_translations:
            for spos in unaligned_pos_left:
                ws = sentence_src[spos]
                counter[ws][None] += 1
            for tpos in unaligned_pos_right:
                wt = sentence_tgt[tpos]
                counter[None][wt] += 1
            
    if args.generate_lexical_prob:
        log.info("computing lexical probabilities")
        res = {}
        for ws in counter:
            if ws is None:
                continue
            res[ws] = {}
            sum_count = sum(counter[ws].values())
            for wt in counter[ws]:
                if wt is None:
                    continue
                res[ws][wt] = counter[ws][wt] / float(sum_count)
    else:
        log.info("selecting best")
        res = {}
        for ws in counter:
            translations = counter[ws].items()
            translations.sort(key = operator.itemgetter(1), reverse = True)
            res[ws] = translations[0][0]
    
    log.info("saving")
    json.dump(res, open(args.dest, "w"))
    
if __name__ == '__main__':
    commandline()
