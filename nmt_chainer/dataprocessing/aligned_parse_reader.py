#!/usr/bin/env python
"""aligned_parse_reader.py: read parsed and aligned files"""
__author__ = "Fabien Cromieres"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fabien.cromieres@gmail.com"
__status__ = "Development"

import logging
import codecs
import exceptions

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
    while True:
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
    sharp, id_, score = id_line.split()
    assert sharp == "#"
    score = float(score)
    align_line = f.readline().strip()
    alignment = []
    for link in align_line.split():
        left, right = link.split("-")
        left = [int(x) for x in left.split(",")]
        right = [int(x) for x in right.split(",")]
        alignment.append((left, right))
    return id_, score, alignment


def load_aligned_corpus(src_fn, tgt_fn, align_fn, skip_empty_align=True, invert_alignment_links=False):
    src = codecs.open(src_fn, encoding="utf8")
    tgt = codecs.open(tgt_fn, encoding="utf8")
    align_f = codecs.open(align_fn, encoding="utf8")

    num_sentence = 0
    while True:
        try:
            id_src, sentence_src = read_one_parse_info_from_file_object(src)
            id_tgt, sentence_tgt = read_one_parse_info_from_file_object(tgt)
            id_align, score_align, alignment = read_one_align_info_from_file_object(
                align_f)
        except exceptions.EOFError:
            return
        if skip_empty_align and len(alignment) == 0:
            log.warn("skipping empty alignment %i %s" % (num_sentence, id_align))
            continue
        assert id_src == id_tgt, "%s != %s @%i" % (id_src, id_tgt, num_sentence)
        assert id_src == id_align, "%s != %s @%i" % (id_src, id_align, num_sentence)

        if invert_alignment_links:
            inverted_alignment = [(right, left) for (left, right) in alignment]
            alignment = inverted_alignment

        yield sentence_src, sentence_tgt, alignment
        num_sentence += 1
