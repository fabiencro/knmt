#!/usr/bin/env python
"""client.py: Client that can issue requests to KNMT Server."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import socket
import os.path
import pkg_resources
import re
from xml.sax.saxutils import escape

class Client:

    def __init__(self, server_ip, server_port):
        self.ip = server_ip
        self.port = server_port

    def query(self, sentence, article_id = 1, beam_width = 30, nb_steps = 50, nb_steps_ratio = 1.5, 
	      prob_space_combination = False, normalize_unicode_unk = True, remove_unk = False, attempt_to_relocate_unk_source = False, 
              sentence_id = 1):

	with open(pkg_resources.resource_filename("nmt_chainer", "templates/query.xml")) as f:
	    query_lines = f.readlines()
	
	query = "".join(query_lines)	
	query = re.sub(r"\bARTICLE_ID\b", str(article_id), query)
	query = re.sub(r"\bBEAM_WIDTH\b", str(beam_width), query)
	query = re.sub(r"\bNB_STEPS\b", str(nb_steps), query)
	query = re.sub(r"\bNB_STEPS_RATIO\b", str(nb_steps_ratio), query)
	query = re.sub(r"\bPROB_SPACE_COMBINATION\b", str(prob_space_combination), query)
	query = re.sub(r"\bNORMALIZE_UNICODE_UNK\b", str(normalize_unicode_unk), query)
	query = re.sub(r"\bREMOVE_UNK\b", str(remove_unk), query)
	query = re.sub(r"\bATTEMPT_TO_RELOCATE_UNK_SOURCE\b", str(attempt_to_relocate_unk_source), query)
	query = re.sub(r"\bSENTENCE_ID\b", str(sentence_id), query)
	query = re.sub(r"\bSENTENCE\b", escape(sentence), query)

	s = socket.socket()
	s.connect((self.ip, self.port))
	s.send(query)

	try:
	    resp = ''
	    while True:
		data = s.recv(1024)
		resp += data
		if not data:
		    break
	    return resp
	finally:
	    s.close()
