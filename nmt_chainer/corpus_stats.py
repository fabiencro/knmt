#!/usr/bin/env python
"""corpus_stats.py: Get corpus stats"""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"

import sys, os
from collections import defaultdict

def get_length_distribution(histogram_data, max_len):
	new_histogram_data = defaultdict(int)
	for length in histogram_data:
		for curr_len in range(length + 5 - length%5, max_len+5, 5):
			new_histogram_data[curr_len] += histogram_data[length]
	return new_histogram_data


def get_stats(input_file):
	input_file = open(input_file)
	histogram_data = defaultdict(int)
	total_lines = 0
	total_words = 0
	for line in input_file:
		words = len(line.strip().split(" "))
		histogram_data[words] += 1
		total_lines += 1
		total_words += words
	print "Total lines: ", total_lines
	max_len = -1000
	for length in sorted(histogram_data):
		#print histogram_data[length], " lines of length: ", length
		if max_len < length:
			max_len = length
	print "Average line length in corpus is ", total_words/total_lines 
	histogram_data = get_length_distribution(histogram_data, max_len)
	for length in sorted(histogram_data):
		if length <= 100:
			print histogram_data[length], " lines of length upto: ", length, " and percentage of lines covered is ", ((100.0*histogram_data[length])/total_lines)
	print "10 percent of the corpus is roughly ", total_lines/10, " lines"
	
def commandline():
    
    import argparse
    parser = argparse.ArgumentParser(description= "Get corpus stats", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file", help = "input file")

    args = parser.parse_args()

    get_stats(args.input_file)

if __name__ == '__main__':
    commandline() 
