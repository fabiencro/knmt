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


class TestPreprocessing:

    def test_LatinScriptProcess_caps(self):
        pp = processors.LatinScriptProcess()

        experiments = {
            "Hello": u"{0}hello".format(processors.LatinScriptProcess.CAP_CHAR),
            "HELLO": u"{0}hello".format(processors.LatinScriptProcess.ALL_CAPS_CHAR),
            "HeLLo": u"HeLLo"
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

    def test_LatinScriptProcess_caps_alt(self):
        pp = processors.LatinScriptProcess()

        s1 = "Hello"
        s2 = "HELLO"
        s3 = "HeLLo"
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

    def test_LatinScriptProcess_punct_word(self):
        pp = processors.LatinScriptProcess()

        experiments = {
            "x...": u"x {0}...".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "Wonderful!": u"Wonderful {0}!".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "Really?": u"Really {0}?".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "def:": u"def {0}:".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "a,": u"a {0},".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "b;": u"b {0};".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "6\"": u"6 {0}\"".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "100%": u"100 {0}%".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "1000000$": u"1000000 {0}$".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "Chris'": u"Chris {0}'".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "Back`": u"Back {0}`".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "Paren)": u"Paren {0})".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "Square]": u"Square {0}]".format(processors.LatinScriptProcess.SUFFIX_CHAR)

        }

        for k, v in experiments.items():
            test = pp.convert_punct_word(k)
            assert test == v
            assert pp.deconvert_punct_sentence(test) == k

        try:
            pp.convert_punct_word(u"{0}This".format(processors.LatinScriptProcess.SUFFIX_CHAR))
        except Exception, ex:
            assert type(ex) == ValueError and str(ex) == "Special char in word"

    def test_LatinScriptProcess_punct_inside(self):
        pp = processors.LatinScriptProcess()

        experiments = {
            "...": u"...",
            "maybe...": u"maybe {0}...".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            ".test.star": u".test {0}.star".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "!test!star!wow": u"!test {0}!star {0}!wow".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "%test%star%wow...": u"%test {0}%star {0}%wow {0}...".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "test+one": u"test {0}+one".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "test?two": u"test {0}?two".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "test?three": u"test {0}?three".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "test'four": u"test {0}'four".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "test\"five": u"test {0}\"five".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "test%six!abc": u"test {0}%six {0}!abc".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "sentence_with!all:the;chars$that]we(want'to&test<if/it@is|possible.":
                u"sentence {0}_with {0}!all {0}:the {0};chars {0}$that {0}]we {0}(want {0}'to {0}&test {0}<if {0}/it {0}@is {0}|possible {0}.".format(processors.LatinScriptProcess.SUFFIX_CHAR),
            "sentence#with^all<the>other=chars*that\\we[want@to-test":
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

    def test_LatinScriptProcess_convert_deconvert_all_adjoint(self):
        pp = processors.LatinScriptProcess()

        experiments = {
            "...": u"...",
            "This  is     a    whitespace    test.": u"{0}this is a whitespace test {1}.".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR),
            "Let's check IT out!!!": u"{0}let {1}'s check {2}it out {1}! {1}! {1}!".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR, processors.LatinScriptProcess.ALL_CAPS_CHAR),
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

    def test_LatinScriptProcess_convert_deconvert_caps_isolate(self):
        pp = processors.LatinScriptProcess(mode="caps_isolate")

        experiments = {
            "...": u"...",
            "This  is     a    whitespace    test.": u"{0} this is a whitespace test {1}.".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR),
            "Let's check IT out!!!": u"{0} let {1}'s check {2} it out {1}! {1}! {1}!".format(processors.LatinScriptProcess.CAP_CHAR, processors.LatinScriptProcess.SUFFIX_CHAR, processors.LatinScriptProcess.ALL_CAPS_CHAR),
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
