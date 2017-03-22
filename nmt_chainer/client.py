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

    def query(self, sentence, article_id=1, beam_width=30, nb_steps=50, nb_steps_ratio=1.5,
              prob_space_combination=False, normalize_unicode_unk=True, remove_unk=False, attempt_to_relocate_unk_source=False,
              sentence_id=1):

        query = """<?xml version="1.0" encoding="utf-8"?>
<article id="{0}"
    beam_width="{1}"
    nb_steps="{2}"
    nb_steps_ratio="{3}"
    prob_space_combination="{4}"
    normalize_unicode_unk="{5}"
    remove_unk="{6}"
    attempt_to_relocate_unk_source="{7}">
    <sentence id="{8}">
        <i_sentence>{9}</i_sentence>
    </sentence>
</article>"""

        query = query.format(article_id, beam_width, nb_steps, nb_steps_ratio, prob_space_combination,
                             normalize_unicode_unk, remove_unk, attempt_to_relocate_unk_source, sentence_id, escape(sentence))
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
