#!/usr/bin/env python
"""multiserver.py: Translation server that works asynchronously and dispatches requests to multiple translation servers."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import datetime
import json
import logging
import sys
 
import time
from time import sleep
import timeit
import socket
import threading
import SocketServer

import Queue

logging.basicConfig()
log = logging.getLogger("rnns:server")
log.setLevel(logging.INFO)

class Worker(threading.Thread):

    def __init__(self, name, src_lang, tgt_lang, manager):
        threading.Thread.__init__(self)
        self.name = name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.manager = manager

    def run(self):
        while True:
            key = "{0}-{1}".format(self.src_lang, self.tgt_lang)
            translation_request = self.manager.translation_request_queues[key].get(True)
            start_request = timeit.default_timer()
            log.info(timestamped_msg("Thead {0}: Handling request: {1}:{2}".format(self.name, translation_request['session_id'], translation_request['article_id'])))
            sleep(60)
            log.info("Request processed in {0} s. by {1} [{2}:{3}]".format(timeit.default_timer() - start_request, self.name, translation_request['session_id'], translation_request['article_id']))

class Manager(object):

    def __init__(self, translation_server_config):
        self.clients = {}
        self.translation_request_queues = {}
        self.workers = {}
        for key in translation_server_config:
            src_lang, tgt_lang = key.split("-")
            self.translation_request_queues[key] = Queue.Queue()
            workerz = []
            self.workers[key] = workerz
            for idx, server in enumerate(translation_server_config[key]):
                worker = Worker("Translater-{0}({1}-{2})".format(idx, src_lang, tgt_lang), src_lang, tgt_lang, self)
                workerz.append(worker)
                worker.start()
        log.info("qs={0}".format(self.translation_request_queues))
        for k, q in list(self.translation_request_queues.items()):
            q.join()


class Server(SocketServer.ThreadingMixIn, SocketServer.TCPServer):

    daemon_threads = True
    allow_reuse_address = True

    def make_request_handler(self, manager):
        class ServerRequestHandler(SocketServer.BaseRequestHandler, object):

            def __init__(self, *args, **kwargs):
                self.manager = manager
                super(ServerRequestHandler, self).__init__(*args, **kwargs)

            def handle(self):
                start_request = timeit.default_timer()
                log.info(timestamped_msg("Handling request..."))
                str_data = self.request.recv(4096)
                log.info("data={0}".format(str_data))
                json_data = json.loads(str_data)
                
                key = "{0}-{1}".format(json_data['src_lang'], json_data['tgt_lang'])
                if json_data['type'] == 'translate':
                    self.manager.translation_request_queues[key].put(json_data)
                    response = {'msg': 'Translation request has been added to queue.'}
                elif json_data['type'] == 'debug':
                    log.info(self.manager.translation_request_queues)
                    trans_queues = {}
                    for k, q in list(self.manager.translation_request_queues.items()):
                        log.info("k={0} q={1}".format(k, q))
                        trans_queues[k] = list(q.queue)
                    response = {
                        'translation_request_queue': trans_queues,
                        'clients': dict(self.manager.clients)
                    }
                else:
                    response = {'msg': 'Unknown request type. Request has been ignored.'}
                    pass
                log.info("q={0} sz={1}".format(self.manager.translation_request_queues[key], self.manager.translation_request_queues[key].qsize()))
                log.info("Request processed in {0} s. by {1}".format(
                    timeit.default_timer() - start_request, threading.current_thread().name))
                response = json.dumps(response)
                self.request.sendall(response)

        return ServerRequestHandler

    def __init__(self, server_address, translation_server_config):
        self.manager = Manager(translation_server_config)
        handler_class = self.make_request_handler(self.manager)
        SocketServer.TCPServer.__init__(self, server_address, handler_class)

def timestamped_msg(msg):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    return "{0}: {1}".format(timestamp, msg)

 
def do_start_server(config_file):
    config = json.load(open(config_file))
    log.info("config={0}".format(config))
    log.info("host={0} port={1}".format(config['host'], config['port']))
    log.info("servers={0}".format(config['servers']))
    server = Server((config['host'], int(config['port'])), config['servers'])
    ip, port = server.server_address
    log.info(timestamped_msg("Start listening for requests on {0}:{1}...".format( socket.gethostname(), port)))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()

    sys.exit(0)
