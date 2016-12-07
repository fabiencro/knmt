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
import re

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
	def __init__(self, target_language = None):
		self.target_language = target_language
		self.prefix = "<2" + target_language + "> " if target_language else ""
		log.info("Initializing the prepostprocessor")

	def apply_postprocessing(self, input_data_file, output_data_file):
		log.info("Applying post-processing to %s" % input_data_file)
		output_data_file = io.open(output_data_file, 'w', encoding="utf-8")
		for input_data in io.open(input_data_file, encoding="utf-8"):
			if self.target_language:
				input_data = re.sub(r'^<2..> ', '', input_data)
			output_data_file.write(input_data.replace(self.bpe.separator+ " ", "").strip() + "\n")
		output_data_file.flush()
		output_data_file.close()
		log.info("post-processing done")

		
	def apply_preprocessing(self, input_data_file, output_data_file):
		log.info("Applying pre-processing to %s" % input_data_file)
		output_data_file = io.open(output_data_file, 'w', encoding="utf-8")
		for input_data in io.open(input_data_file, encoding="utf-8"):
			output_data_file.write(self.prefix + self.bpe.segment(input_data).strip() + "\n")
		output_data_file.flush()
		output_data_file.close()
		log.info("pre-processing done")
		
	def load_model(self, model_path):
		log.info("Loading existing BPE model from %s" % model_path)
		self.model_path = model_path
		self.bpe = BPE(io.open(model_path))