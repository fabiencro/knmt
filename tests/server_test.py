#!/usr/bin/env python
"""server_test.py: Test the server"""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import json

from nmt_chainer.translation.client import Client
import nmt_chainer.translation.server as server

import os.path
import psutil
import pytest
import signal
import subprocess
import time


class TestServer:

    def test_simple_query_to_server(self, tmpdir, gpu):
        """
        Test if the server can start and answers a simple translation query.
        """
        test_data_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "tests_data")
        segmenter_command = "echo '%s' | bin/z2h.pl | bin/tokenizer.perl"
        segmenter_format = "plain"
        config_file = os.path.join(str(test_data_dir), "models/result_invariability.train.train.config")
        model_file = os.path.join(str(test_data_dir), "models/result_invariability.train.model.best.npz")
        args_server = '--server 127.0.0.1:45766 --mode beam_search --segmenter_command="{0}" --segmenter_format {1} {2} {3}'.format(segmenter_command,
                                                                                                                                    segmenter_format, config_file, model_file)
        if gpu is not None:
            args_server += '--gpu {0}'.format(gpu)

        server_process = subprocess.Popen(
            ["python -m nmt_chainer eval {0}".format(args_server)], shell=True)
        try:
            print "Server PID={0}".format(server_process.pid)

            # Wait 10 seconds to make sure that the server has started properly.
            time.sleep(10)

            client = Client('127.0.0.1', 45766)
            resp = client.query("les lunettes sont rouges")
            print "resp={0}".format(resp)
            resp_json = json.loads(resp)
        finally:
            parent = psutil.Process(server_process.pid)
            children = parent.children(recursive=True)
            for process in children:
                process.send_signal(signal.SIGTERM)
            server_process.terminate()

        assert(resp_json['out'] == "die Brille sind rot\n")
