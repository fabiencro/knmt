#!/usr/bin/env python
"""generate_multilingual_bpe_data.py: Accept multiple parallel corpora and generate bpe segmented data. The vocabulary for multiple languages will be shared and the models for source and target side can either be kept separate or shared (TODO:prajdabre)."""
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
from subword_nmt.apply_bpe import *
from subword_nmt.learn_bpe import *

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from utils import ensure_path

if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

logging.basicConfig()
log = logging.getLogger("rnns:prepostprocess")
log.setLevel(logging.INFO)


class DataPreparationPipeline:
	def __init__(self, args):
		log.info("Initializing the Data Preprocessing Pipeline")
		self.args = args
		self.save_prefix = args.save_prefix
		
		self.srcs = [i.split(":")[0] for i in args.train_lang_corpora_pairs]
		self.tgts = [i.split(":")[1] for i in args.train_lang_corpora_pairs]
		self.src_corpora = [i.split(":")[2] for i in args.train_lang_corpora_pairs]
		self.tgt_corpora = [i.split(":")[3] for i in args.train_lang_corpora_pairs]
		self.mlnmt_src_file = io.open(self.save_prefix + "/mlnmt.train.src", "w", encoding = "utf-8")
		self.mlnmt_tgt_file = io.open(self.save_prefix + "/mlnmt.train.tgt", "w", encoding = "utf-8")
		assert len(self.srcs) == len(self.tgts) == len(self.src_corpora) == len(self.tgt_corpora)
		self.balance_vocab_counts = False
		if args.balance_vocab_counts:
			self.balance_vocab_counts = True
			log.info("The vocab counts will be balanced to ensure equal importance to all languages")

		self.balance_corpora = False
		if args.balance_corpora:
			self.balance_corpora = True
			log.info("The corpora for the language pairs with lesser data will be oversampled to ensure that the model focuses equally on all the language pairs.")

		self.is_multi_target = False
		if len(set(self.tgts)) > 1:
			self.is_multi_target = True
			log.info("Multiple target languages detected. Appending the <2xx> token to the beginning of the target side sentences to condition the NMT model to know which target language it should translate to.")
		
		self.merge_operations = args.num_bpe_merge_operations
		self.min_frequency = 2
		self.verbose = True
		log.info("Initialization complete.")
	
	def prepare_data(self):
		log.info("Processing training data")
		self.preprocess_training_data()

		if self.args.dev_lang_corpora_pairs:
			self.srcs = [i.split(":")[0] for i in self.args.dev_lang_corpora_pairs]
			self.tgts = [i.split(":")[1] for i in self.args.dev_lang_corpora_pairs]
			self.src_corpora = [i.split(":")[2] for i in self.args.dev_lang_corpora_pairs]
			self.tgt_corpora = [i.split(":")[3] for i in self.args.dev_lang_corpora_pairs]
			self.mlnmt_src_file = io.open(self.save_prefix + "/mlnmt.dev.src", "w", encoding = "utf-8")
			self.mlnmt_tgt_file = io.open(self.save_prefix + "/mlnmt.dev.tgt", "w", encoding = "utf-8")
			assert len(self.srcs) == len(self.tgts) == len(self.src_corpora) == len(self.tgt_corpora)
			log.info("Preprocessing dev data")
			self.preprocess_eval_data()
			self.mlnmt_src_file = io.open(self.save_prefix + "/mlnmt.dev.src.raw", "w", encoding = "utf-8")
			self.mlnmt_tgt_file = io.open(self.save_prefix + "/mlnmt.dev.tgt.raw", "w", encoding = "utf-8")
			self.generate_multilingual_data_raw()
			

		if self.args.test_lang_corpora_pairs:
			self.srcs = [i.split(":")[0] for i in self.args.test_lang_corpora_pairs]
			self.tgts = [i.split(":")[1] for i in self.args.test_lang_corpora_pairs]
			self.src_corpora = [i.split(":")[2] for i in self.args.test_lang_corpora_pairs]
			self.tgt_corpora = [i.split(":")[3] for i in self.args.test_lang_corpora_pairs]
			self.mlnmt_src_file = io.open(self.save_prefix + "/mlnmt.test.src", "w", encoding = "utf-8")
			self.mlnmt_tgt_file = io.open(self.save_prefix + "/mlnmt.test.tgt", "w", encoding = "utf-8")
			assert len(self.srcs) == len(self.tgts) == len(self.src_corpora) == len(self.tgt_corpora)
			log.info("Preprocessing test data")
			self.preprocess_eval_data()
			self.mlnmt_src_file = io.open(self.save_prefix + "/mlnmt.test.src.raw", "w", encoding = "utf-8")
			self.mlnmt_tgt_file = io.open(self.save_prefix + "/mlnmt.test.tgt.raw", "w", encoding = "utf-8")
			self.generate_multilingual_data_raw()

	def preprocess_eval_data(self):
		self.segment_corpora()
		self.generate_multilingual_data(False)
		
	def preprocess_training_data(self):
		log.info("Running the pipeline")
		self.generate_vocabularies()
		if self.balance_vocab_counts:
			self.balance_vocabularies()
		self.merge_vocabularies()
		self.learn_bpe_models()
		self.segment_corpora()
		self.generate_multilingual_data()
		log.info("Pipeline has finished running")

	def generate_vocabularies(self):
		log.info("Generating vocabularies")
		self.all_vocs_src = {}
		self.all_vocs_src_sizes = {}
		self.largest_voc_size_src = 0
		self.all_vocs_tgt = {}
		self.all_vocs_tgt_sizes = {}
		self.largest_voc_size_tgt = 0
		
		self.all_corpora_sizes = {}
		self.largest_corpus_size = 0
		
		for i in range(len(self.srcs)):
			vocab_src, vocab_src_size, corpus_src_size = get_vocabulary_and_totals(io.open(self.src_corpora[i], encoding="utf-8"))
			vocab_tgt, vocab_tgt_size, corpus_tgt_size = get_vocabulary_and_totals(io.open(self.tgt_corpora[i], encoding="utf-8"))
			
			assert corpus_src_size == corpus_tgt_size

			self.all_corpora_sizes[self.src_corpora[i] + self.tgt_corpora[i]] = corpus_src_size

			if self.all_vocs_src.has_key(self.srcs[i]):
				self.all_vocs_src[self.srcs[i]] += vocab_src
				self.all_vocs_src_sizes[self.srcs[i]] += vocab_src_size
			else:
				self.all_vocs_src[self.srcs[i]] = vocab_src
				self.all_vocs_src_sizes[self.srcs[i]] = vocab_src_size

			if self.all_vocs_tgt.has_key(self.tgts[i]):
				self.all_vocs_tgt[self.tgts[i]] += vocab_tgt
				self.all_vocs_tgt_sizes[self.tgts[i]] += vocab_tgt_size
			else:
				self.all_vocs_tgt[self.tgts[i]] = vocab_tgt
				self.all_vocs_tgt_sizes[self.tgts[i]] = vocab_tgt_size
			
			if self.largest_voc_size_src < self.all_vocs_src_sizes[self.srcs[i]]:
				self.largest_voc_size_src = self.all_vocs_src_sizes[self.srcs[i]]
			if self.largest_voc_size_tgt < self.all_vocs_tgt_sizes[self.tgts[i]]:
				self.largest_voc_size_tgt = self.all_vocs_tgt_sizes[self.tgts[i]]

			if self.largest_corpus_size < corpus_src_size:
				self.largest_corpus_size = corpus_src_size

		log.info("Vocabularies generated")

	def balance_vocabularies(self):
		log.info("Adjusting vocabulary counts to match the largest corpus")
		for i in range(len(self.srcs)):
			src_voc = self.all_vocs_src[self.srcs[i]]
			src_voc_size = self.all_vocs_src_sizes[self.srcs[i]]
			if src_voc_size != self.largest_voc_size_src:
				for word in src_voc:
					src_voc[word] *= 1.0*self.largest_voc_size_src/src_voc_size
			tgt_voc = self.all_vocs_tgt[self.tgts[i]]
			tgt_voc_size = self.all_vocs_tgt_sizes[self.tgts[i]]
			if tgt_voc_size != self.largest_voc_size_tgt:
				for word in tgt_voc:
					tgt_voc[word] *= 1.0*self.largest_voc_size_tgt/tgt_voc_size
		log.info("Vocabulary counts adjusted")
	
	def merge_vocabularies(self):
		log.info("Merging all vocabularies")
		log.info("This merged vocabulary will be used to train the BPE model")
		self.merged_voc_src = reduce(lambda x,y: x+y, [self.all_vocs_src[vocs] for vocs in self.all_vocs_src])
		self.merged_voc_tgt = reduce(lambda x,y: x+y, [self.all_vocs_tgt[vocs] for vocs in self.all_vocs_tgt])
		log.info("Merging complete")

	def learn_bpe_models(self):
		log.info("Learning BPE model for source")
		log.info("Number of merges to be done: %d" % self.merge_operations)
		self.learn_model(self.merged_voc_src, self.save_prefix + "/bpe_model.src")
		log.info("Learning BPE model for target")
		log.info("Number of merges to be done: %d" % self.merge_operations)
		self.learn_model(self.merged_voc_tgt, self.save_prefix + "/bpe_model.tgt")

	def learn_model(self, vocab, model_path):
		model_path = io.open(model_path, 'w', encoding="utf-8")
		vocab = dict([(tuple(x)+('</w>',) ,y) for (x,y) in vocab.items()])
		sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
		stats, indices = get_pair_statistics(sorted_vocab)
		big_stats = copy.deepcopy(stats)
		# threshold is inspired by Zipfian assumption, but should only affect speed
		threshold = max(stats.values()) / 10
		for i in range(self.merge_operations):
			if stats:
				most_frequent = max(stats, key=stats.get)

			# we probably missed the best pair because of pruning; go back to full statistics
			if not stats or (i and stats[most_frequent] < threshold):
				prune_stats(stats, big_stats, threshold)
				stats = copy.deepcopy(big_stats)
				most_frequent = max(stats, key=stats.get)
				# threshold is inspired by Zipfian assumption, but should only affect speed
				threshold = stats[most_frequent] * i/(i+10000.0)
				prune_stats(stats, big_stats, threshold)

			if stats[most_frequent] < self.min_frequency:
				sys.stderr.write('no pair has frequency >= {0}. Stopping\n'.format(self.min_frequency))
				break

			if self.verbose:
				sys.stderr.write(unicode('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent])))
			model_path.write(unicode('{0} {1}\n'.format(*most_frequent)))
			changes = replace_pair(most_frequent, sorted_vocab, indices)
			update_pair_statistics(most_frequent, changes, stats, indices)
			stats[most_frequent] = 0
			if not i % 100:
				prune_stats(stats, big_stats, threshold)

	def segment_corpora(self):
		log.info("Segmenting all the source side corpora using the unified BPE model")
		self.load_model(self.save_prefix + "/bpe_model.src")
		for i in range(len(self.srcs)):
			log.info("Segmenting %s " % self.src_corpora[i])
			infile = io.open(self.src_corpora[i], encoding="utf-8")
			outfile = io.open(self.src_corpora[i] + ".segmented", 'w', encoding="utf-8")
			for inline in infile:
				segmented = self.bpe.segment(inline).strip() + "\n"
				outfile.write(segmented)
		infile.close()
		outfile.close()
		log.info("Segmentation of all source side corpora is done")
		log.info("Segmenting all the target side corpora using the unified BPE model")
		self.load_model(self.save_prefix + "/bpe_model.tgt")
		for i in range(len(self.srcs)):
			log.info("Segmenting %s " % self.tgt_corpora[i])
			prefix = ""
			if self.is_multi_target:
				prefix = "<2" + self.tgts[i] + "> "
			infile = io.open(self.tgt_corpora[i], encoding="utf-8")
			outfile = io.open(self.tgt_corpora[i] + ".segmented", 'w', encoding="utf-8")
			for inline in infile:
				segmented = prefix + self.bpe.segment(inline).strip() + "\n"
				outfile.write(segmented)
			infile.close()
			outfile.close()
		log.info("Segmentation of all target side corpora is done")

	def generate_multilingual_data(self, is_train = True):
		log.info("Generating data for MLNMT")
		mlnmt_src_file = self.mlnmt_src_file
		mlnmt_tgt_file = self.mlnmt_tgt_file
		log.info("Writing merged source and target files at %s and %s respectively" % (mlnmt_src_file, mlnmt_tgt_file))
		for i in range(len(self.srcs)):
			log.info("Processing: %s and %s" % (self.src_corpora[i] + ".segmented", self.tgt_corpora[i] + ".segmented"))
			src_file = io.open(self.src_corpora[i] + ".segmented", encoding="utf-8")
			tgt_file = io.open(self.tgt_corpora[i] + ".segmented", encoding="utf-8")
			src_lang = self.srcs[i]
			tgt_lang = self.tgts[i]
			oversample_ratio = 1.0
			if is_train:
				num_lines = self.all_corpora_sizes[self.src_corpora[i] + self.tgt_corpora[i]]
				if num_lines != self.largest_corpus_size:
					oversample_ratio = 1.0 * self.largest_corpus_size / num_lines
			self.oversample_corpus_and_write(src_file, tgt_file, src_lang, tgt_lang, mlnmt_src_file, mlnmt_tgt_file, oversample_ratio)
			mlnmt_src_file.flush()
			mlnmt_tgt_file.flush()
			log.info("Done")
		mlnmt_src_file.close()
		mlnmt_tgt_file.close()
		log.info("MLNMT data has been generated")

	def generate_multilingual_data_raw(self):
		log.info("Merging raw, unsegmented evaluation data for references")
		mlnmt_src_file = self.mlnmt_src_file
		mlnmt_tgt_file = self.mlnmt_tgt_file
		log.info("Writing merged source and target raw files at %s and %s respectively" % (mlnmt_src_file, mlnmt_tgt_file))
		for i in range(len(self.srcs)):
			log.info("Processing: %s and %s" % (self.src_corpora[i], self.tgt_corpora[i]))
			src_file = io.open(self.src_corpora[i], encoding="utf-8")
			tgt_file = io.open(self.tgt_corpora[i], encoding="utf-8")
			src_lang = self.srcs[i]
			tgt_lang = self.tgts[i]
			oversample_ratio = 1.0
			self.oversample_corpus_and_write(src_file, tgt_file, src_lang, tgt_lang, mlnmt_src_file, mlnmt_tgt_file, oversample_ratio)
			mlnmt_src_file.flush()
			mlnmt_tgt_file.flush()
			log.info("Done")
		mlnmt_src_file.close()
		mlnmt_tgt_file.close()
		log.info("Raw MLNMT data has been generated")

	def oversample_corpus_and_write(self, src_file, tgt_file, src_lang, tgt_lang, mlnmt_src_file, mlnmt_tgt_file, oversample_ratio):
		for src_line, tgt_line in zip(src_file, tgt_file):
			self.oversample_sentence_pair_and_write(src_line, tgt_line, src_lang, tgt_lang, mlnmt_src_file, mlnmt_tgt_file, oversample_ratio)
			

	def oversample_sentence_pair_and_write(self, src_line, tgt_line, src_lang, tgt_lang, mlnmt_src_file, mlnmt_tgt_file, oversample_ratio):
		while(oversample_ratio > 1):
			mlnmt_src_file.write(src_line)
			mlnmt_tgt_file.write(tgt_line)
			oversample_ratio -= 1
		random_number = random.random()
		if oversample_ratio >= random_number:
			mlnmt_src_file.write(src_line)
			mlnmt_tgt_file.write(tgt_line)

	
	def load_model(self, model_path):
		log.info("Loading existing BPE model from %s" % model_path)
		self.bpe = BPE(io.open(model_path))

if __name__ == '__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser(description="Prepare training data.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"--save_prefix", help="created files will be saved with this prefix")
	parser.add_argument(
		"--train_lang_corpora_pairs", nargs='+', type=str, help="list of colon separated quadruplet of language pairs and their corpora. ex: ja:zh:/tmp/corpus-ja-zh.ja:/tmp/corpus-ja-zh.zh en:ja:/tmp/corpus-en-ja.en:/tmp/corpus-en-ja.ja")
	parser.add_argument(
		"--dev_lang_corpora_pairs", nargs='+', type=str, help="list of colon separated quadruplet of language pairs and their corpora. ex: ja:zh:/tmp/corpus-ja-zh.ja:/tmp/corpus-ja-zh.zh en:ja:/tmp/corpus-en-ja.en:/tmp/corpus-en-ja.ja")
	parser.add_argument(
		"--test_lang_corpora_pairs", nargs='+', type=str, help="list of colon separated quadruplet of language pairs and their corpora. ex: ja:zh:/tmp/corpus-ja-zh.ja:/tmp/corpus-ja-zh.zh en:ja:/tmp/corpus-en-ja.en:/tmp/corpus-en-ja.ja")
	parser.add_argument("--balance_vocab_counts", default=None, help="Before learning the BPE model do we want to adjust the count information? This might be needed if one language pair has more data than the other. For now the balancing will be done based on the ratio of the total word count for the text with the maximum total number of words to the total word count for the current text.")
	parser.add_argument("--balance_corpora", default=None, help="After learning the BPE model and segmenting the data do we want to oversample the smaller corpora? This might be needed if one language pair has more data than the other. For now the balancing will be done based on the ratio of the total line count for the text with the maximum total number of lines to the total line count for the current text.")
	parser.add_argument("--num_bpe_merge_operations", type=int, default=32000, help="Number of merge operations that the BPE model should perform on the training vocabulary to learn the BPE codes.")
	args = parser.parse_args()

	dpp = DataPreparationPipeline(args)

	dpp.prepare_data()