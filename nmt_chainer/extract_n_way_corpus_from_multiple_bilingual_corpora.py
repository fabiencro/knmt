#!/usr/bin/env python
"""extract_n_way_corpus_from_multiple_bilingual_corpora.py: Reads in source and target files representing multiple bilingual parallel corpora and keeps source sentences that have the same target sentence. This essentially generates an N way corpus."""
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
log = logging.getLogger("rnns:extractnwaycorpus")
log.setLevel(logging.INFO)


class ExtractNWayCorpus:
	def __init__(self, args):
		log.info("Initializing the Data Filtering Pipeline.")
		
		self.args = args
		self.src_corpora = args.src_corpora
		self.tgt_corpora = args.tgt_corpora
		self.extension = args.extension
		self.src_corpora_filtered = [io.open('/'.join(corpus_path.split("/")[:-1]) + "/" + self.extension + "." + corpus_path.split("/")[-1], 'w', encoding="utf-8") for corpus_path in self.src_corpora]
		self.tgt_corpora_filtered = [io.open('/'.join(corpus_path.split("/")[:-1]) + "/" + self.extension + "." + corpus_path.split("/")[-1], 'w', encoding="utf-8") for corpus_path in self.tgt_corpora]
		self.verbose = args.verbose
		
		log.info("Initialization complete.")
	
	def load_file(self, infile1, infile2):
		counter = 0
		full_corpus = {}
		for line1, line2 in zip(infile1, infile2):
			full_corpus[line2.strip().replace("   ", " ").replace("  ", " ")] = line1.strip().replace("   ", " ").replace("  ", " ")
			counter += 1
		return full_corpus, counter

	def load_data(self):
		log.info("Loading bilingual corpora.")
		self.corpora = []
		for i in xrange(len(self.src_corpora)):
			corpus, lines = self.load_file(io.open(self.src_corpora[i], encoding="utf-8"), io.open(self.tgt_corpora[i], encoding="utf-8"))
			self.corpora.append(corpus)
			log.info("Loaded %d lines." % lines)

	def filter_data(self):
		common_sentences = set(self.corpora[0].keys())
		for additional_corpus in self.corpora[1:]:
			log.info("Common sentence count so far is: %d" % len(common_sentences))
			common_sentences = common_sentences.intersection(set(additional_corpus.keys()))

		log.info("Common sentence count is: %d" % len(common_sentences))
		log.info("Writing all corpora.")
		log.info("Writing from (%s,%s) to (%s,%s)." % (self.src_corpora,self.tgt_corpora,self.src_corpora_filtered,self.tgt_corpora_filtered))
		
		for tgt_sentence in common_sentences:
			for i in xrange(len(self.corpora)):
				corpus = self.corpora[i]
				if corpus.has_key(tgt_sentence):
					self.tgt_corpora_filtered[i].write(tgt_sentence + "\n")
					self.src_corpora_filtered[i].write(corpus[tgt_sentence] + "\n")
			self.src_corpora_filtered[i].flush()
			self.tgt_corpora_filtered[i].flush()
			
		for i in xrange(len(self.corpora)):
			self.src_corpora_filtered[i].close()
			self.tgt_corpora_filtered[i].close()

	def run_pipeline(self):
		log.info("Loading data.")
		self.load_data()
		log.info("Data loaded.")
		log.info("Writing N-way corpus.")
		self.filter_data()
		log.info("Done.")



if __name__ == '__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser(description="Remove Dev and Test set sentences from Bilingual corpora.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"--src_corpora", nargs = "+", help="The input source corpus file.")
	parser.add_argument(
		"--tgt_corpora", nargs = "+", help="The input target corpus file.")
	parser.add_argument(
		"--extension", help="The filtered source corpus output file.")
	parser.add_argument(
		"--verbose", default = False, action = "store_true", help="More details about overlap stats.")
	args = parser.parse_args()
		
	enwc = ExtractNWayCorpus(args)

	enwc.run_pipeline()