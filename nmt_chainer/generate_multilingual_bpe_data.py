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
log = logging.getLogger("rnns:prepostprocess")
log.setLevel(logging.INFO)


class DataPreparationPipeline:
	def __init__(self, args):
		log.info("Initializing the Data Preprocessing Pipeline")
		self.args = args
		self.save_prefix = args.save_prefix
		self.merge_operations = args.num_bpe_merge_operations
		self.is_multisource = args.is_multisource
		self.drop_source_sentences = args.drop_source_sentences
		self.pretrained_src_bpe_model = args.pretrained_src_bpe_model
		self.pretrained_tgt_bpe_model = args.pretrained_tgt_bpe_model
		self.min_frequency = 2
		self.verbose = True
		self.joint_bpe_model = args.joint_bpe_model
		self.max_src_vocab = args.max_src_vocab
		self.max_tgt_vocab = args.max_tgt_vocab
		self.mlnmt_src_vocab = set()
		self.mlnmt_tgt_vocab = set()
		self.srcs = [i.split(":")[0] for i in args.train_lang_corpora_pairs]
		self.tgts = [i.split(":")[1] for i in args.train_lang_corpora_pairs]
		self.src_corpora = [i.split(":")[2] for i in args.train_lang_corpora_pairs]
		self.tgt_corpora = [i.split(":")[3] for i in args.train_lang_corpora_pairs]
		self.mlnmt_src_file = io.open(self.save_prefix + "/mlnmt.train.src", "w", encoding = "utf-8")
		self.mlnmt_tgt_file = io.open(self.save_prefix + "/mlnmt.train.tgt", "w", encoding = "utf-8")
		assert len(self.srcs) == len(self.tgts) == len(self.src_corpora) == len(self.tgt_corpora)
		self.balance_vocab_counts = args.balance_vocab_counts
		if args.balance_vocab_counts:
			log.info("The vocab counts will be balanced to ensure equal importance to all languages")

		self.balance_corpora = args.balance_corpora
		if args.balance_corpora:
			log.info("The corpora for the language pairs with lesser data will be oversampled to ensure that the model focuses equally on all the language pairs.")

		self.is_multi_target = False
		if len(set(self.tgts)) > 1:
			self.is_multi_target = True
			log.info("Multiple target languages detected. Appending the <2xx> token to the beginning of the source side sentences to condition the NMT model to know which target language it should translate to.")
		
		log.info("Initialization complete.")
	
	def prepare_data(self):
		log.info("Preprocessing training data")
		self.preprocess_training_data()
		self.mlnmt_src_file = io.open(self.save_prefix + "/mlnmt.train.src.raw", "w", encoding = "utf-8")
		self.mlnmt_tgt_file = io.open(self.save_prefix + "/mlnmt.train.tgt.raw", "w", encoding = "utf-8")
		self.generate_multilingual_data_raw()

		if self.args.dev_lang_corpora_pairs:
			self.srcs = [i.split(":")[0] for i in self.args.dev_lang_corpora_pairs]
			self.tgts = [i.split(":")[1] for i in self.args.dev_lang_corpora_pairs]
			self.src_corpora = [i.split(":")[2] for i in self.args.dev_lang_corpora_pairs]
			self.tgt_corpora = [i.split(":")[3] for i in self.args.dev_lang_corpora_pairs]
			self.mlnmt_src_file = io.open(self.save_prefix + "/mlnmt.dev.src", "w", encoding = "utf-8")
			self.mlnmt_tgt_file = io.open(self.save_prefix + "/mlnmt.dev.tgt", "w", encoding = "utf-8")
			assert len(self.srcs) == len(self.tgts) == len(self.src_corpora) == len(self.tgt_corpora)
			log.info("Preprocessing dev data")
			self.preprocess_eval_data("dev")
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
			self.preprocess_eval_data("test")
			self.mlnmt_src_file = io.open(self.save_prefix + "/mlnmt.test.src.raw", "w", encoding = "utf-8")
			self.mlnmt_tgt_file = io.open(self.save_prefix + "/mlnmt.test.tgt.raw", "w", encoding = "utf-8")
			self.generate_multilingual_data_raw()

	def preprocess_eval_data(self, train_dev_test = "dev"):
		self.drop_source_sentences = False
		self.segment_corpora(train_dev_test)
		self.generate_multilingual_data(train_dev_test)
		
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
		
		self.all_vocs_src_tgt = {}
		self.all_vocs_src_tgt_sizes = {}
		self.largest_voc_size_src_tgt = 0
		

		self.all_corpora_sizes = {}
		self.largest_corpus_size = 0
		
		for i in range(len(self.srcs)):
			vocab_src, vocab_src_size, corpus_src_size = get_vocabulary_and_totals(io.open(self.src_corpora[i], encoding="utf-8"))
			vocab_tgt, vocab_tgt_size, corpus_tgt_size = get_vocabulary_and_totals(io.open(self.tgt_corpora[i], encoding="utf-8"))
			
			assert corpus_src_size == corpus_tgt_size

			self.all_corpora_sizes[self.src_corpora[i] + self.tgt_corpora[i]] = corpus_src_size

			if self.joint_bpe_model:
				if self.all_vocs_src_tgt.has_key(self.srcs[i]):
					self.all_vocs_src_tgt[self.srcs[i]] += vocab_src
					self.all_vocs_src_tgt_sizes[self.srcs[i]] += vocab_src_size
				else:
					self.all_vocs_src_tgt[self.srcs[i]] = vocab_src
					self.all_vocs_src_tgt_sizes[self.srcs[i]] = vocab_src_size

				if self.all_vocs_src_tgt.has_key(self.tgts[i]):
					self.all_vocs_src_tgt[self.tgts[i]] += vocab_tgt
					self.all_vocs_src_tgt_sizes[self.tgts[i]] += vocab_tgt_size
				else:
					self.all_vocs_src_tgt[self.tgts[i]] = vocab_tgt
					self.all_vocs_src_tgt_sizes[self.tgts[i]] = vocab_tgt_size
				
				if self.largest_voc_size_src_tgt < max(self.all_vocs_src_tgt_sizes[self.srcs[i]], self.all_vocs_src_tgt_sizes[self.tgts[i]]):
					self.largest_voc_size_src_tgt = max(self.all_vocs_src_tgt_sizes[self.srcs[i]], self.all_vocs_src_tgt_sizes[self.tgts[i]])
			else:				
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
			if self.joint_bpe_model:
				src_voc = self.all_vocs_src_tgt[self.srcs[i]]
				src_voc_size = self.all_vocs_src_tgt_sizes[self.srcs[i]]
				if src_voc_size != self.largest_voc_size_src_tgt:
					for word in src_voc:
						src_voc[word] *= 1.0*self.largest_voc_size_src_tgt/src_voc_size
					self.all_vocs_src_tgt_sizes[self.srcs[i]] = self.largest_voc_size_src_tgt
				tgt_voc = self.all_vocs_src_tgt[self.tgts[i]]
				tgt_voc_size = self.all_vocs_src_tgt_sizes[self.tgts[i]]
				if tgt_voc_size != self.largest_voc_size_src_tgt:
					for word in tgt_voc:
						tgt_voc[word] *= 1.0*self.largest_voc_size_src_tgt/tgt_voc_size
					self.all_vocs_src_tgt_sizes[self.tgts[i]] = self.largest_voc_size_src_tgt
			else:
				src_voc = self.all_vocs_src[self.srcs[i]]
				src_voc_size = self.all_vocs_src_sizes[self.srcs[i]]
				if src_voc_size != self.largest_voc_size_src:
					for word in src_voc:
						src_voc[word] *= 1.0*self.largest_voc_size_src/src_voc_size
					self.all_vocs_src_sizes[self.srcs[i]] = self.largest_voc_size_src
				tgt_voc = self.all_vocs_tgt[self.tgts[i]]
				tgt_voc_size = self.all_vocs_tgt_sizes[self.tgts[i]]
				if tgt_voc_size != self.largest_voc_size_tgt:
					for word in tgt_voc:
						tgt_voc[word] *= 1.0*self.largest_voc_size_tgt/tgt_voc_size
					self.all_vocs_tgt_sizes[self.tgts[i]] = self.largest_voc_size_tgt
		log.info("Vocabulary counts adjusted")
	
	def merge_vocabularies(self):
		log.info("Merging all vocabularies")
		log.info("This merged vocabulary will be used to train the BPE model")
		if self.joint_bpe_model:
			log.info("Merging the source and target vocab into a single collection")
			self.merged_voc_src_tgt = reduce(lambda x,y: x+y, [self.all_vocs_src_tgt[vocs] for vocs in self.all_vocs_src_tgt])
		else:
			log.info("Merging the source and target vocab into a their respective collections")
			self.merged_voc_src = reduce(lambda x,y: x+y, [self.all_vocs_src[vocs] for vocs in self.all_vocs_src])
			self.merged_voc_tgt = reduce(lambda x,y: x+y, [self.all_vocs_tgt[vocs] for vocs in self.all_vocs_tgt])
		log.info("Merging complete")

	def replicate_bpe_model(self, in_model, model_path):
		log.info("Replicating the BPE model.")
		model_path = io.open(model_path, 'w', encoding="utf-8")
		in_model = io.open(in_model, encoding="utf-8")
		for line in in_model:
			model_path.write(line)
		model_path.flush()
		model_path.close()
		in_model.close()

	def learn_bpe_models(self):
		adjustment = 0
		if self.is_multi_target:
			adjustment = len(set(self.tgts))
		if self.joint_bpe_model:
			if self.pretrained_src_bpe_model or self.pretrained_tgt_bpe_model:
				log.info("Using pretrained BPE model for source and target. Replicating as copies.")
				self.replicate_bpe_model(self.pretrained_src_bpe_model, self.save_prefix + "/bpe_model.src")	
				self.replicate_bpe_model(self.pretrained_src_bpe_model, self.save_prefix + "/bpe_model.tgt")
				log.info("Replication completed")
			else:
				log.info("Learning joint BPE model for source and target")
				if self.max_src_vocab or self.max_tgt_vocab:
					self.max_tgt_vocab = self.max_src_vocab if not self.max_tgt_vocab else self.max_tgt_vocab
					self.max_src_vocab = self.max_tgt_vocab if not self.max_src_vocab else self.max_src_vocab
					log.info("The number of merge operations will be determined based on how many merge rules it takes to reach the goal of a joint vocabulary of size %d for source and target. Note that %d vocab symbols will have to be accounted to accommodate the <2xx> tokens in case of multiple target languages." % (self.max_src_vocab, adjustment))
					self.learn_model(self.merged_voc_src_tgt, self.save_prefix + "/bpe_model.src", self.max_src_vocab - adjustment)
				else:
					log.info("Number of merges to be done: %d" % self.merge_operations)
					self.learn_model(self.merged_voc_src_tgt, self.save_prefix + "/bpe_model.src")
				log.info("Model learned. Now replicating it for target side")
				self.replicate_bpe_model(self.save_prefix + "/bpe_model.src", self.save_prefix + "/bpe_model.tgt")
				log.info("Replication completed")
		else:
			if self.pretrained_src_bpe_model:
				log.info("Using pretrained BPE model for source. Replicating as a copy.")
				self.replicate_bpe_model(self.pretrained_src_bpe_model, self.save_prefix + "/bpe_model.src")
				log.info("Replication completed")
			else:
				log.info("Learning BPE model for source")
				if self.max_src_vocab:
					log.info("The number of merge operations will be determined based on how many merge rules it takes to reach the goal of a vocabulary of size %d for source. Note that %d vocab symbols will have to be accounted to accommodate the <2xx> tokens in case of multiple target languages." % (self.max_src_vocab, adjustment))
					self.learn_model(self.merged_voc_src, self.save_prefix + "/bpe_model.src", self.max_src_vocab - adjustment)
				else:
					log.info("Number of merges to be done: %d" % self.merge_operations)
					self.learn_model(self.merged_voc_src, self.save_prefix + "/bpe_model.src")
			
			if self.pretrained_tgt_bpe_model:
				log.info("Using pretrained BPE model for target. Replicating as a copy.")
				self.replicate_bpe_model(self.pretrained_tgt_bpe_model, self.save_prefix + "/bpe_model.tgt")
				log.info("Replication completed")
			else:
				log.info("Learning BPE model for target")
				if self.max_tgt_vocab:
					log.info("The number of merge operations will be determined based on how many merge rules it takes to reach the goal of a vocabulary of size %d for target" % self.max_tgt_vocab)
					self.learn_model(self.merged_voc_tgt, self.save_prefix + "/bpe_model.tgt", self.max_tgt_vocab)
				else:
					log.info("Number of merges to be done: %d" % self.merge_operations)
					self.learn_model(self.merged_voc_tgt, self.save_prefix + "/bpe_model.tgt")

	def learn_model(self, vocab, model_path, max_vocab = None):
		if max_vocab:
			self.merge_operations = sys.maxint
		model_path = io.open(model_path, 'w', encoding="utf-8")
		dict_list = [(tuple(x)+('</w>',) ,round(y)) for (x,y) in vocab.items()]
		random.shuffle(dict_list)
		vocab = dict(dict_list)
		sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
		stats, indices, symbol_list = get_pair_statistics(sorted_vocab)
		big_stats = copy.deepcopy(stats)
		curr_vocab_size = 0
		#print symbol_list
		# threshold is inspired by Zipfian assumption, but should only affect speed
		curr_vocab_size = len(symbol_list)
		log.info("The current vocab size is %d. " % curr_vocab_size)
		threshold = max(stats.values()) / 10
		for i in xrange(self.merge_operations):
			if max_vocab:
				curr_vocab_size = len(symbol_list)
				log.info("The current vocab size is %d. " % curr_vocab_size)
				if curr_vocab_size > max_vocab:
					log.info("Exceeded the permissible limit of %d vocab size. No more BPE merges needed. Maximal model learned." % max_vocab)
					break
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
			update_pair_statistics(most_frequent, changes, stats, indices, symbol_list)
			stats[most_frequent] = 0
			if not i % 100:
				prune_stats(stats, big_stats, threshold)

	def segment_corpora(self, train_dev_test = "train"):
		log.info("Segmenting all the source side corpora using the source side BPE model")
		self.load_model(self.save_prefix + "/bpe_model.src", self.save_prefix + "/bpe_model.tgt")
		for i in range(len(self.srcs)):
			log.info("Segmenting %s " % self.src_corpora[i])
			infile = io.open(self.src_corpora[i], encoding="utf-8")
			outfile = io.open(self.save_prefix + "/" + self.srcs[i] + "-" + self.tgts[i] + "." + self.srcs[i] + "." +train_dev_test + ".segmented", 'w', encoding="utf-8")
			prefix = ""
			if self.is_multi_target:
				prefix = "<2" + self.tgts[i] + "> "
			for inline in infile:
				segmented = prefix + self.bpe_src.segment(inline.strip()).strip() + "\n"
				if train_dev_test == "train":
					for word in segmented.strip().split(" "):
						self.mlnmt_src_vocab.add(word)
				outfile.write(segmented)
		infile.close()
		outfile.close()
		log.info("Segmentation of all source side corpora is done")
		log.info("Segmenting all the target side corpora using the target side BPE model")
		#self.load_model()
		for i in range(len(self.srcs)):
			log.info("Segmenting %s " % self.tgt_corpora[i])
			infile = io.open(self.tgt_corpora[i], encoding="utf-8")
			outfile = io.open(self.save_prefix + "/" + self.srcs[i] + "-" + self.tgts[i] + "." + self.tgts[i] + "." +train_dev_test + ".segmented", 'w', encoding="utf-8")
			for inline in infile:
				segmented = self.bpe_tgt.segment(inline.strip()).strip() + "\n"
				if train_dev_test == "train":
					for word in segmented.strip().split(" "):
						self.mlnmt_tgt_vocab.add(word)
				outfile.write(segmented)
			infile.close()
			outfile.close()
		log.info("Segmentation of all target side corpora is done")

	def generate_multilingual_data(self, train_dev_test = "train"):
		log.info("Generating data for MLNMT")
		mlnmt_src_file = self.mlnmt_src_file
		mlnmt_tgt_file = self.mlnmt_tgt_file
		if self.is_multisource:
			assert len(set(self.tgts)) == 1
			a = self.all_corpora_sizes.values()
			#print a
			assert len(set(a)) == 1
			#assert all(e == a[0] for e in a)
			log.info("Generating multisource data and writing to %s and %s" % (mlnmt_src_file, mlnmt_tgt_file))
			total_lines = a[0]
			all_source_files = [io.open(self.save_prefix + "/" + self.srcs[i] + "-" + self.tgts[i] + "." + self.srcs[i] + "." +train_dev_test + ".segmented", encoding="utf-8") for i in range(len(self.srcs))]
			tgt_file = io.open(self.save_prefix + "/" + self.srcs[0] + "-" + self.tgts[0] + "." + self.tgts[0] + "." +train_dev_test + ".segmented", encoding="utf-8")
			for j in xrange(total_lines):
				all_source_sents = [all_source_files[i].readline().strip() for i in range(len(self.srcs))]
				final_src_sents = self.generate_sentence_dropped_multisource_sentences(all_source_sents)
				final_tgt_sent = tgt_file.readline().strip() + "\n"
				for final_src_sent in final_src_sents:
					if len(final_src_sent.strip()) != 0:
						mlnmt_src_file.write(final_src_sent)
						mlnmt_tgt_file.write(final_tgt_sent)
				mlnmt_src_file.flush()
				mlnmt_tgt_file.flush()
			mlnmt_src_file.close()
			mlnmt_tgt_file.close()
			log.info("Done")
		else:
			log.info("Writing merged source and target files at %s and %s respectively" % (mlnmt_src_file, mlnmt_tgt_file))
			for i in range(len(self.srcs)):
				log.info("Processing: %s and %s" % (self.save_prefix + "/" + self.srcs[i] + "-" + self.tgts[i] + "." + self.srcs[i] + "." +train_dev_test + ".segmented", self.save_prefix + "/" + self.srcs[i] + "-" + self.tgts[i] + "." + self.tgts[i] + "." +train_dev_test + ".segmented"))
				src_file = io.open(self.save_prefix + "/" + self.srcs[i] + "-" + self.tgts[i] + "." + self.srcs[i] + "." +train_dev_test + ".segmented", encoding="utf-8")
				tgt_file = io.open(self.save_prefix + "/" + self.srcs[i] + "-" + self.tgts[i] + "." + self.tgts[i] + "." +train_dev_test + ".segmented", encoding="utf-8")
				src_lang = self.srcs[i]
				tgt_lang = self.tgts[i]
				oversample_ratio = 1.0
				if train_dev_test == "train":
					num_lines = self.all_corpora_sizes[self.src_corpora[i] + self.tgt_corpora[i]]
					if num_lines != self.largest_corpus_size:
						oversample_ratio = 1.0 * self.largest_corpus_size / num_lines
				log.info("Oversampling ratio is %f" % oversample_ratio)
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
		if self.is_multisource:
			assert len(set(self.tgts)) == 1
			a = self.all_corpora_sizes.values()
			assert len(set(a)) == 1
			#assert all(e == a[0] for e in a)
			log.info("Generating multisource data and writing to %s and %s" % (mlnmt_src_file, mlnmt_tgt_file))
			total_lines = a[0]
			all_source_files = [io.open(self.src_corpora[i], encoding="utf-8") for i in range(len(self.srcs))]
			tgt_file = io.open(self.tgt_corpora[i], encoding="utf-8")
			for j in xrange(total_lines):
				all_source_sents = [all_source_files[i].readline().strip() for i in range(len(self.srcs))]
				final_src_sents = self.generate_sentence_dropped_multisource_sentences(all_source_sents)
				final_tgt_sent = tgt_file.readline().strip() + "\n"
				for final_src_sent in final_src_sents:
					if len(final_src_sent.strip()) != 0:
						mlnmt_src_file.write(final_src_sent)
						mlnmt_tgt_file.write(final_tgt_sent)
				mlnmt_src_file.flush()
				mlnmt_tgt_file.flush()
			mlnmt_src_file.close()
			mlnmt_tgt_file.close()
			log.info("Done")
		else:
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

	def generate_sentence_dropped_multisource_sentences(self, all_source_sents):
		final_src_sents = [" ".join(all_source_sents) + "\n"]
		if self.drop_source_sentences:
			max_drops = len(all_source_sents)-1
			for i in range(max_drops):
				drop_index = random.randint(0, len(all_source_sents)-1)
				all_source_sents.pop(drop_index)
				final_src_sents.append(" ".join(all_source_sents) + "\n")
		return final_src_sents


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

	
	def load_model(self, src_model_path, tgt_model_path):
		log.info("Loading existing BPE models from %s and %s" % (src_model_path, tgt_model_path))
		self.bpe_src = None
		self.bpe_tgt = None
		self.bpe_src = BPE(io.open(src_model_path))
		self.bpe_tgt = BPE(io.open(tgt_model_path))

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
	parser.add_argument("--balance_vocab_counts", default=False, action = "store_true", help="Before learning the BPE model do we want to adjust the count information? This might be needed if one language pair has more data than the other. For now the balancing will be done based on the ratio of the total word count for the text with the maximum total number of words to the total word count for the current text.")
	parser.add_argument("--balance_corpora", default=False, action = "store_true", help="After learning the BPE model and segmenting the data do we want to oversample the smaller corpora? This might be needed if one language pair has more data than the other. For now the balancing will be done based on the ratio of the total line count for the text with the maximum total number of lines to the total line count for the current text.")
	parser.add_argument("--num_bpe_merge_operations", type=int, default=90000, help="Number of merge operations that the BPE model should perform on the training vocabulary to learn the BPE codes.")
	parser.add_argument("--joint_bpe_model", default=False, action = "store_true", help="Do you want to learn a single BPE model for both the source and target side corpora so that the merge operations are consistent on both sides? This might be useful in the cases where the languages share cognates.")
	parser.add_argument("--max_src_vocab", type=int, default=None, help="Maximum number of source side symbols desired. This will automatically limit the number of BPE merge operations by stopping when this limit has been reached.  When the joint_bpe_model flag has been specified the limit will be decided by the value of this flag.")
	parser.add_argument("--max_tgt_vocab", type=int, default=None, help="Maximum number of target side symbols desired. This will automatically limit the number of BPE merge operations by stopping when this limit has been reached. When the joint_bpe_model flag has been specified the limit will be decided by the value of the max_src_vocab flag.")
	parser.add_argument("--is_multisource", default=False, action = "store_true", help="Do you want to generate multisource data? This assumes that the target side of each sentence is the same.")
	parser.add_argument("--drop_source_sentences", default=False, action = "store_true", help="Do you want to generate multisource data with randomly dropped source sentences? Is to ensure that the NMT system becomes robust enough to deal with situations where multiple source sentences might not exist.")
	parser.add_argument(
		"--pretrained_src_bpe_model", help="Use a pretrained BPE model for source text segmentation")
	parser.add_argument(
		"--pretrained_tgt_bpe_model", help="Use a pretrained BPE model for target text segmentation")
	args = parser.parse_args()
		
	dpp = DataPreparationPipeline(args)

	dpp.prepare_data()