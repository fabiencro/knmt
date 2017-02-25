#!/usr/bin/env python
"""create_dummy_source_corpus.py: Creates a source corpus of N lines with a dummy token both of which are specified via an argument."""
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
import time
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
log = logging.getLogger("rnns:createdummycorpus")
log.setLevel(logging.INFO)


class CreateDummyCorpus:
	def __init__(self, args):
		log.info("Initializing the Data Generation Pipeline.")
		
		self.args = args
		self.num_lines = int(args.num_lines)
		self.token = args.token
		self.outcorpus = args.outcorpus
		self.verbose = args.verbose
		
		log.info("Dummy token is %s." % self.token)

		log.info("Initialization complete.")
	
	def generate_corpus(self):
		self.outcorpus = io.open(self.outcorpus, 'w', encoding="utf-8")
		for i in xrange(self.num_lines):
			self.outcorpus.write(unicode(self.token+"\n"))
			if not (i+1)%100000 and self.verbose:
				log.info("Generated %d lines." % (i+1))
		self.outcorpus.flush()
		self.outcorpus.close()

	def run_pipeline(self):
		log.info("Generating corpus.")
		self.generate_corpus()
		log.info("Done.")



if __name__ == '__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser(description="Remove Dev and Test set sentences from Monolingual corpora.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"num_lines", help="The number of lines to generate.")
	parser.add_argument(
		"token", help="The dummy token to append.")
	parser.add_argument(
		"outcorpus", help="The generated corpus output file.")
	parser.add_argument(
		"--verbose", default = False, action = "store_true", help="More details.")
	args = parser.parse_args()
	start = time.time()
	dpp = CreateDummyCorpus(args)

	dpp.run_pipeline()
	end = time.time()
	log.info("Took a total of %d seconds." % (end-start))