#!/usr/bin/env python
"""remove_dev_test_from_bitext.py: Read in 3 files and write lines from the first and second file that are not present in the third file into another file. The purpose is to remove target side of dev and test set from a bilingual corpus."""
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
log = logging.getLogger("rnns:removedevtestfrombitext")
log.setLevel(logging.INFO)


class RemoveDevTestFromBitext:
	def __init__(self, args):
		log.info("Initializing the Data Filtering Pipeline.")
		
		self.args = args
		self.src_corpus = args.src_corpus
		self.tgt_corpus = args.tgt_corpus
		self.devtestcorpus = args.devtestcorpus
		self.filteredsrccorpus = args.filteredsrccorpus
		self.filteredtgtcorpus = args.filteredtgtcorpus
		self.verbose = args.verbose
		
		log.info("Initialization complete.")
	
	def load_file(self, infile1, infile2 = None):
		full_corpus = None
		counter = 0
		if infile2 is None:
			full_corpus = set()
			for line in infile1:
				full_corpus.add(line.strip())
				counter += 1
		else:
			full_corpus = {}
			for line1, line2 in zip(infile1, infile2):
				full_corpus[line2.strip()] = line1.strip()
				counter += 1
		return full_corpus, counter

	def load_data(self):
		log.info("Loading bilingual corpus.")
		self.corpus, lines = self.load_file(io.open(self.src_corpus, encoding="utf-8"), io.open(self.tgt_corpus, encoding="utf-8"))
		log.info("Loaded %d lines." % lines)
		log.info("Loading devtest corpus.")
		self.devtestcorpus, lines = self.load_file(io.open(self.devtestcorpus, encoding="utf-8"))
		log.info("Loaded %d lines." % lines)
		log.info("Opening filtered corpus file.")
		self.filteredsrccorpus = io.open(self.filteredsrccorpus, 'w', encoding="utf-8")
		self.filteredtgtcorpus = io.open(self.filteredtgtcorpus, 'w', encoding="utf-8")

	def filter_data(self):
		counter = 0
		overlap_stats = defaultdict(int)
		for tgt_line, src_line in self.corpus.iteritems():
			if tgt_line not in self.devtestcorpus:
				self.filteredsrccorpus.write(src_line + "\n")
				self.filteredtgtcorpus.write(tgt_line + "\n")
				counter += 1
			elif self.verbose:
				log.info("Detected overlap: %s, %s." % (src_line, tgt_line))
				overlap_stats[src_line+", "+tgt_line] += 1	
		self.filteredsrccorpus.flush()
		self.filteredsrccorpus.close()
		self.filteredtgtcorpus.flush()
		self.filteredtgtcorpus.close()
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
	parser = argparse.ArgumentParser(description="Remove Dev and Test set sentences from Bilingual corpora.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"src_corpus", help="The input source corpus file.")
	parser.add_argument(
		"tgt_corpus", help="The input target corpus file.")
	parser.add_argument(
		"devtestcorpus", help="The input dev and test sentences file.")
	parser.add_argument(
		"filteredsrccorpus", help="The filtered source corpus output file.")
	parser.add_argument(
		"filteredtgtcorpus", help="The filtered target corpus output file.")
	parser.add_argument(
		"--verbose", default = False, action = "store_true", help="More details about overlap stats.")
	args = parser.parse_args()
		
	obj = RemoveDevTestFromBitext(args)

	obj.run_pipeline()