#!/usr/bin/env python
"""client.py: Client that can issue requests to KNMT Server."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import socket
import os.path
import re
from xml.sax.saxutils import escape


class Client:

    def __init__(self, server_ip, server_port):
        self.ip = server_ip
        self.port = server_port

    def submit_request(self, request):
        s = None
        try:
            s = socket.socket()
            s.connect((self.ip, self.port))
            s.send(request)

            resp = ''
            while True:
                data = s.recv(1024)
                resp += data
                if not data:
                    break
            return resp
        finally:
            if s:
                s.close()

    def cancel(self):
        query = """<?xml version="1.0" encoding="utf-8"?><cancel_translation/>"""
        return self.submit_request(query)

    def query(self, 
              sentence, 
              article_id=1, 
              beam_width=30, 
              nb_steps=50, 
              nb_steps_ratio=1.5,
              beam_pruning_margin='none',  
              beam_score_length_normalization='none',
              beam_score_length_normalization_strength=0.2,
              post_score_length_normalization='simple',
              post_score_length_normalization_strength=0.2,
              beam_score_coverage_penalty='none',
              beam_score_coverage_penalty_strength=0.2,
              post_score_coverage_penalty='none',
              post_score_coverage_penalty_strength=0.2,
              prob_space_combination=False, 
              normalize_unicode_unk=True, 
              remove_unk=False, 
              attempt_to_relocate_unk_source=False,
              force_finish=False,
              sentence_id=1, 
              attn_graph_width=0, 
              attn_graph_height=0):

        query = """<?xml version="1.0" encoding="utf-8"?>
<article id="{0}"
    beam_width="{1}"
    nb_steps="{2}"
    nb_steps_ratio="{3}"
    beam_pruning_margin="{4}"
    beam_score_length_normalization="{5}"
    beam_score_length_normalization_strength="{6}"
    post_score_length_normalization="{7}"
    post_score_length_normalization_strength="{8}"
    beam_score_coverage_penalty="{9}"
    beam_score_coverage_penalty_strength="{10}"
    post_score_coverage_penalty="{11}"
    post_score_coverage_penalty_strength="{12}"
    prob_space_combination="{13}"
    normalize_unicode_unk="{14}"
    remove_unk="{15}"
    attempt_to_relocate_unk_source="{16}"
    force_finish="{17}"
    attn_graph_width="{18}"
    attn_graph_height="{19}">
    <sentence id="{20}">
        <i_sentence>{21}</i_sentence>
    </sentence>
</article>"""

        query = query.format(article_id, 
                             beam_width, 
                             nb_steps, 
                             nb_steps_ratio, 
                             beam_pruning_margin,
                             beam_score_length_normalization,
                             beam_score_length_normalization_strength,
                             post_score_length_normalization,
                             post_score_length_normalization_strength,
                             beam_score_coverage_penalty,
                             beam_score_coverage_penalty_strength,
                             post_score_coverage_penalty,
                             post_score_coverage_penalty_strength,                    
                             str(prob_space_combination).lower(),
                             str(normalize_unicode_unk).lower(), 
                             str(remove_unk).lower(), 
                             str(attempt_to_relocate_unk_source).lower(), 
                             str(force_finish).lower(), 
                             str(attn_graph_width),
                             str(attn_graph_height),
                             sentence_id, 
                             escape(sentence))

        return self.submit_request(query)

    def get_log_file(self, filename, page=1):
        query = """<?xml version="1.0" encoding="utf-8"?><get_log_file filename="{0}" page="{1}"/>"""
        return self.submit_request(query.format(filename, page))

    def get_log_files(self):
        query = """<?xml version="1.0" encoding="utf-8"?><get_log_files/>"""
        return self.submit_request(query)
