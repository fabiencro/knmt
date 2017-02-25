#!/usr/bin/env python
"""remove_dev_test_from_corpus.py: Read in a monolingual corpus that has csentences split int osegments over contiguous lines. This program reconstructs sentences based on strong punctuation. Hopefully!"""
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
import re
import random
from collections import defaultdict

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from utils import ensure_path

if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

logging.basicConfig()
log = logging.getLogger("rnns:reconstructonstrongpunctuation")
log.setLevel(logging.INFO)


class ReconstructSentencesOnStrongPunctuation:
	def __init__(self, args):
		log.info("Initializing the Data Processing Pipeline.")
		
		self.args = args
		self.infile = args.infile
		self.outfile = args.outfile
		self.verbose = args.verbose
		
		log.info("Initialization complete.")
	
	def load_file(self, infile):
		full_corpus = []
		counter = 0
		for line in infile:
			full_corpus.append(line.strip())
			counter += 1
		return full_corpus, counter

	def load_data(self):
		log.info("Loading monolingual corpus.")
		self.infile, lines = self.load_file(io.open(self.infile, encoding="utf-8"))
		log.info("Loaded %d lines." % lines)
		log.info("Opening reconstructed corpus file.")
		self.outfile = io.open(self.outfile, 'w', encoding="utf-8")

	def reconstruct_sentences(self):
		counter = 0
		sentence = ""
		for line in self.infile:
			split_indices = [m.end(0) for m in re.finditer("[0-9A-Za-z]+[\.\!\?][ \t]*", line)]
			if len(split_indices) == 0:
				split_indices = [m.end(0) for m in re.finditer("[0-9A-Za-z]+[\.\!\?]\"[ \t]*", line)]
				if len(split_indices) == 0:
					sentence += line
			else:
				curr_index = 0
				for index in split_indices:
					sentence += line[curr_index:index]
					self.outfile.write(sentence + "\n")
					counter += 1
					sentence = ""
					curr_index = index
				sentence = line[curr_index:]
		self.outfile.write(sentence + "\n")
		self.outfile.flush()
		self.outfile.close()
		log.info("Wrote %d lines." % counter)

	def run_pipeline(self):
		log.info("Loading data.")
		self.load_data()
		log.info("Data loaded.")
		log.info("Reconstructing lines on strong punctuation.")
		self.reconstruct_sentences()
		log.info("Processing is complete.")



if __name__ == '__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser(description="Reconstruct sentences on strong punctuation.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"infile", help="The input corpus file.")
	parser.add_argument(
		"outfile", help="The output corpus file with sentences restructured on strong punctuation.")
	parser.add_argument(
		"--verbose", default = False, action = "store_true", help="More details.")
	args = parser.parse_args()
		
	dpp = ReconstructSentencesOnStrongPunctuation(args)

	dpp.run_pipeline()