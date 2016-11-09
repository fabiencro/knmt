#!/usr/bin/env python
"""prepostprocessor.py: prepostprocess the input data"""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"

import collections
import logging
import codecs
import json
import operator
import os.path
import gzip
import io
from subword_nmt.apply_bpe import *
from subword_nmt.learn_bpe import *

from utils import ensure_path

if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

logging.basicConfig()
log = logging.getLogger("rnns:prepostprocess")
log.setLevel(logging.INFO)

class PrePostProcessor:
	def apply_preprocessing(self):
		raise "Not Implemented"

	def apply_postprocessing(self):
		raise "Not Implemented"

	def load_model(self):
		raise "Not Implemented"

class BPEPrePostProcessor(PrePostProcessor):
	def __init__(self):
		log.info("Initializing the prepostprocessor")

	def apply_postprocessing(self, input_data):
		log.info("Applying post-processing to %s" % input_data)
		return input_data.replace(self.bpe.separator+ " ", "").strip()
		
	def apply_preprocessing(self, input_data):
		log.info("Applying pre-processing to %s" % input_data)
		return self.bpe.segment(input_data).strip()
	
	def load_model(self, model_path):
		log.info("Loading existing BPE model from %s" % model_path)
		self.bpe = BPE(io.open(model_path))