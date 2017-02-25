#!/usr/bin/env python
"""remove_dev_test_from_corpus.py: Read in 2 files and write lines from the first file that are not present in another file into another file. The purpose is to remove target side of dev and test set from a monolingual corpus."""
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
log = logging.getLogger("rnns:removedevtestfrommono")
log.setLevel(logging.INFO)


class RemoveDevTestFromMonolingual:
	def __init__(self, args):
		log.info("Initializing the Data Filtering Pipeline.")
		
		self.args = args
		self.corpus = args.corpus
		self.devtestcorpus = args.devtestcorpus
		self.filteredcorpus = args.filteredcorpus
		self.verbose = args.verbose
		
		log.info("Initialization complete.")
	
	def load_file(self, infile):
		full_corpus = set()
		counter = 0
		for line in infile:
			full_corpus.add(line.strip())
			counter += 1
		return full_corpus, counter

	def load_data(self):
		log.info("Loading monolingual corpus.")
		self.corpus, lines = self.load_file(io.open(self.corpus, encoding="utf-8"))
		log.info("Loaded %d lines." % lines)
		log.info("Loading devtest corpus.")
		self.devtestcorpus, lines = self.load_file(io.open(self.devtestcorpus, encoding="utf-8"))
		log.info("Loaded %d lines." % lines)
		log.info("Opening filtered corpus file.")
		self.filteredcorpus = io.open(self.filteredcorpus, 'w', encoding="utf-8")

	def filter_data(self):
		counter = 0
		overlap_stats = defaultdict(int)
		for monoline in self.corpus:
			if monoline not in self.devtestcorpus:
				self.filteredcorpus.write(monoline + "\n")
				counter += 1
			elif self.verbose:
				log.info("Detected overlap: %s." % monoline)
				overlap_stats[monoline] += 1	
		self.filteredcorpus.flush()
		self.filteredcorpus.close()
		log.info("Wrote %d lines." % counter)
		if self.verbose:
			log.info("Some overlap stats. Format: (String,Overlap Frequency).")
			for datum in sorted(overlap_stats.iteritems(), key=lambda (k,v): v, reverse=False)[:100]:
				log.info("%s" % str(datum))

	def run_pipeline(self):
		log.info("Loading data.")
		self.load_data()
		log.info("Data loaded.")
		log.info("Removing overlapping lines.")
		self.filter_data()
		log.info("Filtering is complete.")



if __name__ == '__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser(description="Remove Dev and Test set sentences from Monolingual corpora.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"corpus", help="The input monolingual corpus file.")
	parser.add_argument(
		"devtestcorpus", help="The input dev and test sentences file.")
	parser.add_argument(
		"filteredcorpus", help="The filtered corpus output file.")
	parser.add_argument(
		"--verbose", default = False, action = "store_true", help="More details about overlap stats.")
	args = parser.parse_args()
		
	obj = RemoveDevTestFromMonolingual(args)

	obj.run_pipeline()