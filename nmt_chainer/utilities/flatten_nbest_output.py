"""flatten_nbest_output.py: Extract the first sentence of a translation result file that used --nbest."""

import fileinput
import re

p = re.compile("^(\d+)\|\|\|(.*)\|\|\|(.*)$")

sentence_idx = -1
for line in fileinput.input():
    m = p.match(line)
    if m:
        line_sentence_idx = m.group(1)
        line_sentence_text = m.group(2)
        line_sentence_scores = m.group(3)
        if line_sentence_idx != sentence_idx:
            print line_sentence_text
            sentence_idx = line_sentence_idx
