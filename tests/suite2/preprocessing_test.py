#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocessing_tests.py: Some preprocessing tests"""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import re

import nmt_chainer.dataprocessing.processors as processors


class TestLatinScriptProcess:

    def test_convert_deconvert_caps(self):
        pp = processors.LatinScriptProcess()

        experiments = {
            u"Hello": u"{0}hello".format(processors.LatinScriptProcess.CAP_CHAR),
            u"HELLO": u"{0}hello".format(processors.LatinScriptProcess.ALL_CAPS_CHAR),
            u"HeLLo": u"HeLLo"
        }

        for k, v in experiments.items():
            test = pp.convert_caps(k)
            assert test == v
            assert pp.deconvert_caps(test) == k

        try:
            pp.convert_caps(u"{0}hello".format(processors.LatinScriptProcess.CAP_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            pp.convert_caps(u"{0}hello".format(processors.LatinScriptProcess.ALL_CAPS_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

    def test_convert_deconvert_caps_alt(self):
        pp = processors.LatinScriptProcess()

        s1 = u"Hello"
        s2 = u"HELLO"
        s3 = u"HeLLo"
        t1 = pp.convert_caps_alt(s1)
        t2 = pp.convert_caps_alt(s2)
        t3 = pp.convert_caps_alt(s3)
        assert t1 == u"{0} hello".format(processors.LatinScriptProcess.CAP_CHAR)
        assert t2 == u"{0} hello".format(processors.LatinScriptProcess.ALL_CAPS_CHAR)
        assert t3 == u"HeLLo"
        assert pp.deconvert_caps_alt_sentence(t1) == s1
        assert pp.deconvert_caps_alt_sentence(t2) == s2
        assert pp.deconvert_caps_alt_sentence(t3) == s3
        assert pp.deconvert_caps_alt_sentence(u"{0} this is very {1} important".format(
            processors.LatinScriptProcess.CAP_CHAR,
            processors.LatinScriptProcess.ALL_CAPS_CHAR)) == "This is very IMPORTANT"

        try:
            pp.convert_caps_alt(t1)
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            pp.convert_caps_alt(t2)
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            pp.convert_caps_alt("This is a test")
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

    def test_convert_deconvert_punct_word(self):
        pp = processors.LatinScriptProcess()

        experiments = {
            u"x...": u"x {0}...".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"Wonderful!": u"Wonderful {0}!".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"Really?": u"Really {0}?".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"def:": u"def {0}:".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"a,": u"a {0},".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"b;": u"b {0};".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"6\"": u"6 {0}\"".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"100%": u"100 {0}%".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"1000000$": u"1000000 {0}$".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"Chris'": u"Chris {0}'".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"Back`": u"Back {0}`".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"Paren)": u"Paren {0})".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"Square]": u"Square {0}]".format(processors.LatinScriptProcess.SUFFIX_CHAR)
        }

        for k, v in experiments.items():
            test = pp.convert_punct_word(k)
            assert test == v
            assert pp.deconvert_punct_sentence(test) == k

        try:
            pp.convert_punct_word(u"{0}This".format(processors.LatinScriptProcess.SUFFIX_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

    def test_convert_deconvert_punct_inside(self):
        pp = processors.LatinScriptProcess()

        experiments = {
            u"...": u"...",
            u"maybe...": u"maybe {0}...".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u".test.star": u".test {0}.star".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"!test!star!wow": u"!test {0}!star {0}!wow".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"%test%star%wow...": u"%test {0}%star {0}%wow {0}...".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"test+one": u"test {0}+one".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"test?two": u"test {0}?two".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"test?three": u"test {0}?three".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"test'four": u"test {0}'four".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"test\"five": u"test {0}\"five".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"test%six!abc": u"test {0}%six {0}!abc".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"sentence_with!all:the;chars$that]we(want'to&test<if/it@is|possible.":
                u"sentence {0}_with {0}!all {0}:the {0};chars {0}$that {0}]we {0}(want {0}'to {0}&test {0}<if {0}/it {0}@is {0}|possible {0}.".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            u"sentence#with^all<the>other=chars*that\\we[want@to-test":
                u"sentence {0}#with {0}^all {0}<the {0}>other {0}=chars {0}*that {0}\\we {0}[want {0}@to {0}-test".format(processors.LatinScriptProcess.SUFFIX_CHAR)
        }

        for k, v in experiments.items():
            test = pp.convert_punct_inside(k)
            assert test == v
            assert pp.deconvert_punct_sentence(test) == k

        try:
            pp.convert_punct_inside(u"in{0}side".format(processors.LatinScriptProcess.SUFFIX_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

    def test_convert_deconvert_all_adjoint(self):
        pp = processors.LatinScriptProcess()

        experiments = {
            u"...": u"...",
            u"This  is     a    whitespace    test.": u"{0}this is a whitespace test {1}.".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR),
            u"Let's check IT out!!!": u"{0}let {1}'s check {2}it out {1}! {1}! {1}!".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR, processors.LatinScriptProcess.ALL_CAPS_CHAR),
            u"Éric est parti à l'école très tôt.": u"{0}éric est parti à l {1}'école très tôt {1}.".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR)
        }

        for k, v in experiments.items():
            test = pp.convert(k)
            assert test == v
            assert pp.deconvert(test) == re.sub('\s+', ' ', k)

        try:
            pp.convert(u"{0}hello".format(processors.LatinScriptProcess.CAP_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            pp.convert(u"{0}hello".format(processors.LatinScriptProcess.ALL_CAPS_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            pp.convert(u"in{0}side".format(processors.LatinScriptProcess.SUFFIX_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

    def test_convert_deconvert_caps_isolate(self):
        pp = processors.LatinScriptProcess(mode="caps_isolate")

        experiments = {
            u"...": u"...",
            u"This  is     a    whitespace    test.": u"{0} this is a whitespace test {1}.".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR),
            u"Let's check IT out!!!": u"{0} let {1}'s check {2} it out {1}! {1}! {1}!".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR, processors.LatinScriptProcess.ALL_CAPS_CHAR),
            u"Éric est parti à l'école très tôt.": u"{0} éric est parti à l {1}'école très tôt {1}.".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR)
        }

        for k, v in experiments.items():
            test = pp.convert(k)
            assert test == v
            assert pp.deconvert(test) == re.sub('\s+', ' ', k)

        try:
            pp.convert(u"{0}hello".format(processors.LatinScriptProcess.CAP_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            pp.convert(u"{0}hello".format(processors.LatinScriptProcess.ALL_CAPS_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            pp.convert(u"in{0}side".format(processors.LatinScriptProcess.SUFFIX_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"


class TestSimpleSegmenter:

    def test_convert_deconvert_word(self):
        pp = processors.SimpleSegmenter()

        experiments = {
            u"This is a test sentence.": ['This', 'is', 'a', 'test', 'sentence.'],
            u"   This    is  a test    sentence   with  white    spaces  ...":
                ['', '', '', 'This', '', '', '', 'is', '', 'a', 'test', '', '', '', 'sentence', '', '', 'with', '', 'white', '', '', '', 'spaces', '', '...'],
            u"What HAPPEN if there's funny chars like !@#$%?": ['What', 'HAPPEN', 'if', "there's", 'funny', 'chars', 'like', '!@#$%?']
        }

        for k, v in experiments.items():
            test = pp.convert(k)
            assert test == v
            assert pp.deconvert(test) == k

    def test_convert_deconvert_word2char(self):
        pp = processors.SimpleSegmenter(type="word2char")

        experiments = {
            u"This is a test sentence.": ('T', 'h', 'i', 's', 'i', 's', 'a', 't', 'e', 's', 't', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e', '.'),
            u"   This    is  a test    sentence   with  white    spaces  ...":
                ('T', 'h', 'i', 's', 'i', 's', 'a', 't', 'e', 's', 't', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e', 'w', 'i', 't', 'h', 'w', 'h', 'i', 't', 'e', 's', 'p', 'a', 'c', 'e', 's', '.', '.', '.'),
            u"What HAPPEN if there's funny chars like !@#$%?": ('W', 'h', 'a', 't', 'H', 'A', 'P', 'P', 'E', 'N', 'i', 'f', 't', 'h', 'e', 'r', 'e', "'", 's', 'f', 'u', 'n', 'n', 'y', 'c', 'h', 'a', 'r', 's', 'l', 'i', 'k', 'e', '!', '@', '#', '$', '%', '?')
        }

        for k, v in experiments.items():
            test = pp.convert(k)
            assert test == v
            assert pp.deconvert(test) == re.sub('\s+', '', k)

    def test_convert_deconvert_char(self):
        pp = processors.SimpleSegmenter(type="char")

        experiments = {
            u"This is a test sentence.": ('T', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 't', 'e', 's', 't', ' ', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e', '.'),
            u"   This    is  a test    sentence   with  white    spaces  ...":
                (' ', ' ', ' ', 'T', 'h', 'i', 's', ' ', ' ', ' ', ' ', 'i', 's', ' ', ' ', 'a', ' ', 't', 'e', 's', 't', ' ', ' ', ' ', ' ', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e', ' ', ' ', ' ', 'w', 'i', 't', 'h', ' ', ' ', 'w', 'h', 'i', 't', 'e', ' ', ' ', ' ', ' ', 's', 'p', 'a', 'c', 'e', 's', ' ', ' ', '.', '.', '.'),
            u"What HAPPEN if there's funny chars like !@#$%?":
                ('W', 'h', 'a', 't', ' ', 'H', 'A', 'P', 'P', 'E', 'N', ' ', 'i', 'f', ' ', 't', 'h', 'e', 'r', 'e', "'", 's', ' ', 'f', 'u', 'n', 'n', 'y', ' ', 'c', 'h', 'a', 'r', 's', ' ', 'l', 'i', 'k', 'e', ' ', '!', '@', '#', '$', '%', '?')
        }

        for k, v in experiments.items():
            test = pp.convert(k)
            assert test == v
            assert pp.deconvert(test) == k


class TestProcessorChain:

    def test_convert_deconvert(self):
        pp_1 = processors.LatinScriptProcess()
        pp_2 = processors.SimpleSegmenter(type="char")

        pp = processors.ProcessorChain([pp_1, pp_2])

        experiments = {
            u"Hello": (processors.LatinScriptProcess.CAP_CHAR, u'h', u'e', u'l', u'l', u'o'),
            u"HELLO": (processors.LatinScriptProcess.ALL_CAPS_CHAR, u'h', u'e', u'l', u'l', u'o'),
            u"HeLLo": (u'H', u'e', u'L', u'L', u'o'),
            u"sentence_with!all:the;chars$that]we(want'to&test<if/it@is|possible.":
                (u's', u'e', u'n', u't', u'e', u'n', u'c', u'e', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u'_', u'w', u'i', u't', u'h', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u'!', u'a', u'l', u'l', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u':', u't', u'h', u'e', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u';', u'c', u'h', u'a', u'r', u's', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u'$', u't', u'h', u'a', u't', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u']', u'w', u'e', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u'(', u'w', u'a', u'n', u't', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u"'", u't', u'o', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u'&', u't', u'e', u's', u't', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u'<', u'i', u'f', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u'/', u'i', u't', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u'@', u'i', u's', u' ',
                 processors.LatinScriptProcess.SUFFIX_CHAR, u'|', u'p', u'o', u's', u's', u'i', u'b',
                 u'l', u'e', u' ', processors.LatinScriptProcess.SUFFIX_CHAR, u'.')
        }

        for k, v in experiments.items():
            test = pp.convert(k)
            assert test == v
            assert pp.deconvert(test) == k

        try:
            pp.convert(u"{0}hello".format(processors.LatinScriptProcess.CAP_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            pp.convert(u"{0}hello".format(processors.LatinScriptProcess.ALL_CAPS_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            pp.convert(u"in{0}side".format(processors.LatinScriptProcess.SUFFIX_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"


class TestBiProcessChain:

    def test_convert_deconvert_with_simple_processors(self):
        pp = processors.BiProcessorChain()

        src_pp = processors.LatinScriptProcess(mode="all_adjoint")
        pp.add_src_processor(src_pp)

        tgt_pp = processors.LatinScriptProcess(mode="all_adjoint")
        pp.add_tgt_processor(tgt_pp)

        experiments = {
            (u"Hello", u"HELLO", 
             u"{0}hello".format(processors.LatinScriptProcess.CAP_CHAR), u"{0}hello".format(processors.LatinScriptProcess.ALL_CAPS_CHAR)),
            (u"x...", u"Wonderful!",
             u"x {0}...".format(processors.LatinScriptProcess.SUFFIX_CHAR),
             u"{0}wonderful {1}!".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR)),
            (u"test?three", u"test'four",
             u"test {0}?three".format(processors.LatinScriptProcess.SUFFIX_CHAR), 
             u"test {0}'four".format(processors.LatinScriptProcess.SUFFIX_CHAR)),
            (u"sentence_with!all:the;chars$that]we(want'to&test<if/it@is|possible.",
             u"sentence#with^all<the>other=chars*that\\we[want@to-test",
             u"sentence {0}_with {0}!all {0}:the {0};chars {0}$that {0}]we {0}(want {0}'to {0}&test {0}<if {0}/it {0}@is {0}|possible {0}.".format(processors.LatinScriptProcess.SUFFIX_CHAR),
             u"sentence {0}#with {0}^all {0}<the {0}>other {0}=chars {0}*that {0}\\we {0}[want {0}@to {0}-test".format(processors.LatinScriptProcess.SUFFIX_CHAR))

        }

        for idx, exp_data in enumerate(experiments):
            sentence_src = exp_data[0]
            sentence_tgt = exp_data[1]
            sentence_src_expected_res = exp_data[2]
            sentence_tgt_expected_res = exp_data[3]
            sentence_src_res, sentence_tgt_res = pp.convert(sentence_src, sentence_tgt)
            # print u"Exp {0}".format(idx)
            # print u"sentence_src={0}".format(sentence_src)
            # print u"sentence_tgt={0}".format(sentence_tgt)
            # print u"sentence_src_res={0}".format(sentence_src_res)
            # print u"sentence_tgt_res={0}".format(sentence_tgt_res)
            assert sentence_src_res == sentence_src_expected_res 
            assert sentence_tgt_res == sentence_tgt_expected_res
            
            sentence_src_res, sentence_tgt_res = pp.deconvert(sentence_src_res, sentence_tgt_res)
            assert sentence_src_res == sentence_src
            assert sentence_tgt_res == sentence_tgt

        valid_sentence = u"..."
        invalid_sentence = u"in{0}side".format(processors.LatinScriptProcess.SUFFIX_CHAR)

        try:
            sentence_src_res, sentence_tgt_res = pp.convert(invalid_sentence, valid_sentence)
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

        try:
            sentence_src_res, sentence_tgt_res = pp.convert(valid_sentence, invalid_sentence)
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

