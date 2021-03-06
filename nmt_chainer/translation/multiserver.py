#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""multiserver.py: Translation server that works asynchronously and dispatches requests to multiple translation servers."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import json
import logging
import logging.config
import sys
 
import time
from time import sleep
import timeit
import socket
import threading
import socketserver
import subprocess
from collections import deque

from urllib.parse import urlencode
import queue
import os
from os import listdir
from os.path import isfile, join, dirname, basename

from nmt_chainer.translation.client import Client

PAGE_SIZE = 5000

class RequestQueue(queue.Queue):

    def __init__(self, log):
        self.log = log
        queue.Queue.__init__(self)

    def redistribute_requests(self):
        self.log.debug("redistribute_requests begin")
        self.mutex.acquire()
        temp = {}
        self.log.debug('begin')
        while self._qsize() > 0:
            item = self._get()
            #self.log.debug("item={0}".format(item))
            #self.log.debug("{0} {1}".format(item['session_id'], item['msg_id']))
            #self.log.debug("{0} {1}".format(item['session_id'], item['article_id']))
            key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
            self.log.debug("{0} {1}".format(key, item['sentence_number']))

            if not key in temp:
                temp[key] = deque()
            temp[key].append(item)
        self.log.debug('end')
        keys = list(temp.keys())[:]
        self.log.debug('keys={0}'.format(keys))
        
        self.log.debug('begin2')
        while keys:
            for k in keys:
                item = temp[k].popleft()
                key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
                self.log.debug("{0} {1}".format(key, item['sentence_number']))
                #self.log.debug("item={0}".format(item))
                self._put(item)
            keys = [k for k in keys if temp[k]]
        self.log.debug('end2')
        self.mutex.release()
        self.log.debug("redistribute_requests end")

    def cancel_requests_from(self, client_id):
        self.log.debug("cancel_requests_from {0} begin".format(client_id)) 
        self.mutex.acquire()
        temp = deque()
        while self._qsize() > 0:
            item = self._get()
            item_key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
            if 'sentence_number' in item:
                self.log.debug("{0} {1}".format(item_key, item['sentence_number']))
            elif 'type' in item:
                self.log.debug("{0} {1}".format(item_key, item['type']))

            if item_key != client_id:
                temp.append(item)
        while len(temp) > 0:
            item = temp.pop()
            self._put(item)
        self.mutex.release()
        self.log.debug("cancel_requests_from {0} end".format(client_id)) 

    def content(self):
        # Beware, if the base class changes, this will break.
        self.mutex.acquire()
        queue_copy = list(self.queue)
        self.mutex.release()
        return queue_copy

    def refresh_keepalive_timer(self, client_id, timestamp):
        self.mutex.acquire()
        for trans_req in self.queue:
            key = "{0}-{1}".format(trans_req['session_id'], trans_req['client_tab_id']) 
            if key == client_id:
                trans_req['keepalive'] = timestamp
        self.mutex.release()

    def remove_expired_requests(self):
        now = time.time()
        self.mutex.acquire()
        self.queue = deque([x for x in self.queue if now - x['keepalive'] < 10])
        self.mutex.release()

class QueueCleaner(threading.Thread):

    def __init__(self, queues, delay=10):
        threading.Thread.__init__(self)
        self.queues = queues
        self.delay = delay

    def run(self):
        while True:
            time.sleep(self.delay)
            for k in self.queues:
                self.queues[k].remove_expired_requests()

class Worker(threading.Thread):

    def __init__(self, name, category, host, port, manager, log):
        threading.Thread.__init__(self)
        self.name = name
        self.category = category
        self.lang_pair, self.model = category.split("_")
        self.src_lang, self.tgt_lang = self.lang_pair.split("-")
        self.host = host
        self.port = port
        self.manager = manager
        self.log = log
        self.current_client_key = None

    def stop(self):
        client = Client(self.host, self.port)
        try:
            resp = client.cancel()
        except BaseException as err:
            self.log.info("An error has occurred when the client named '{0}' performed a CANCEL query for the '{1}' category: '{2}'".format(self.name, self.category, err))

    def run(self):
        while True:
            translation_request = self.manager.translation_request_queues[self.category].get(True)
            self.log.debug("Request for worker={0}".format(translation_request))
            self.current_client_key = "{0}-{1}".format(translation_request['session_id'], translation_request['client_tab_id']) 
            start_request = timeit.default_timer()
            self.log.debug("Thread {0}: Handling request: {1}:{2}".format(self.name, self.current_client_key, translation_request['article_id']))

            self.log.debug("SENTENCE for worker {0}:{1}: {2}".format(self.host, self.port, translation_request['sentence']))
            self.manager.add_active_translation(self.category, translation_request)
            client = Client(self.host, self.port)
            try:
                resp = client.query(translation_request['sentence'], 
                    article_id=translation_request['article_id'],
                    beam_width=translation_request['beam_width'] if 'beam_width' in translation_request else 30,
                    nb_steps=translation_request['nb_steps'] if 'nb_steps' in translation_request else 50,
                    nb_steps_ratio=translation_request['nb_steps_ratio'] if 'nb_steps_ratio' in translation_request else 1.2,
                    beam_pruning_margin=translation_request['beam_pruning_margin'] if 'beam_pruning_margin' in translation_request else 'none',
                    beam_score_length_normalization=translation_request['beam_score_length_normalization'] if 'beam_score_length_normalization' in translation_request else 'none',
                    beam_score_length_normalization_strength=translation_request['beam_score_length_normalization_strength'] if 'beam_score_length_normalization_strength' in translation_request else 0.2,
                    post_score_length_normalization=translation_request['post_score_length_normalization'] if 'post_score_length_normalization' in translation_request else 'simple',
                    post_score_length_normalization_strength=translation_request['post_score_length_normalization_strength'] if 'post_score_length_normalization_strength' in translation_request else 0.2,
                    beam_score_coverage_penalty=translation_request['beam_score_coverage_penalty'] if 'beam_score_coverage_penalty' in translation_request else 'none',
                    beam_score_coverage_penalty_strength=translation_request['beam_score_coverage_penalty_strength'] if 'beam_score_coverage_penalty_strength' in translation_request else 0.2,
                    post_score_coverage_penalty=translation_request['post_score_coverage_penalty'] if 'post_score_coverage_penalty' in translation_request else 'none',
                    post_score_coverage_penalty_strength=translation_request['post_score_coverage_penalty_strength'] if 'post_score_coverage_penalty_strength' in translation_request else 0.2,
                    prob_space_combination=translation_request['prob_space_combination'] if 'prob_space_combination' in translation_request else 'false',
                    normalize_unicode_unk=translation_request['normalize_unicode_unk'] if 'normalize_unicode_unk' in translation_request else 'true',
                    remove_unk=translation_request['remove_unk'] if 'remove_unk' in translation_request else 'false',
                    attempt_to_relocate_unk_source=translation_request['attempt_to_relocate_unk_source'] if 'attempt_to_relocate_unk_source' in translation_request else 'false',
                    force_finish=translation_request['force_finish'] if 'force_finish' in translation_request else 'false',
                    sentence_id=translation_request['sentence_number'],
                    attn_graph_width=translation_request['attn_graph_width'] if 'attn_graph_width' in translation_request else 0,
                    attn_graph_height=translation_request['attn_graph_height'] if 'attn_graph_height' in translation_request else 0)
                json_resp = json.loads(resp)
                if 'error' in json_resp:
                    self.log.info("An error has occurred: {0}\n".format(json_resp['error']))
                    if 'stacktrace' in json_resp:
                        for item in json_resp['stacktrace']:
                            self.log.info(item)
                else:
                    self.log.debug("TRANSLATION: {0}".format(json_resp['out']))
                    if not self.current_client_key in self.manager.client_cancellations or not self.manager.client_cancellations[self.current_client_key]:
                        response_queue = self.manager.client_responses[self.current_client_key]
                        self.log.debug("adding translation to queue {0} [key:{1}]".format(repr(response_queue), self.current_client_key))
                        response_queue.put(json_resp)
            except BaseException as err:
                self.log.info("An error has occurred when the client named '{0}' performed a TRANSLATE query for the '{1}' category: '{2}'".format(self.name, self.category, err))
            finally:
                self.manager.remove_active_translation(self.category, translation_request)

            self.log.debug("Request processed in {0} s. by {1} [{2}:{3}]".format(timeit.default_timer() - start_request, self.name, self.current_client_key, translation_request['article_id']))

class Manager(object):

    def __init__(self, translation_server_config, text_to_sentences_splitter, log):
        self.translation_server_config = translation_server_config
        self.text_to_sentences_splitter = text_to_sentences_splitter
        self.log = log
        self.client_responses = {}
        self.client_cancellations = {}
        self.translation_request_queues = {}
        self.active_translations = {}
        self.workers = {}
        self.mutex = threading.Lock()
        for server_category in translation_server_config:
            self.translation_request_queues[server_category] = RequestQueue(self.log)
            self.active_translations[server_category] = []
            workerz = []
            self.workers[server_category] = workerz
            for idx, server in enumerate(translation_server_config[server_category]):
                worker = Worker("Translater-{0}({1})".format(idx, server), server_category, server['host'], server['port'], self, self.log)
                workerz.append(worker)
                worker.start()
        self.translation_request_queue_cleaner = QueueCleaner(self.translation_request_queues)
        self.translation_request_queue_cleaner.start()
        for k, q in list(self.translation_request_queues.items()):
            q.join()

    def add_active_translation(self, category, req):
        self.mutex.acquire()
        self.active_translations[category].append(req)
        self.mutex.release()

    def remove_active_translation(self, category, req):
        self.mutex.acquire()
        self.active_translations[category].remove(req)
        self.mutex.release()

    def get_stats(self, category):
        clients = set()
        all_requests = self.translation_request_queues[category].content() + self.active_translations[category]
        for item in all_requests:
            if 'session_id' in item and 'client_tab_id' in item:
                client_key = "{0}-{1}".format(item['session_id'], item['client_tab_id'])
                clients.add(client_key)
        return {'clients': len(clients), 'requests': len(all_requests)};

    def poll(self, category, client_id):
        if category and category in self.translation_request_queues:
            self.translation_request_queues[category].refresh_keepalive_timer(client_id, time.time())

        resp = {};
        if category is None:
            resp['servers'] = list(self.translation_server_config.keys())
        elif category in self.workers:
            resp['workload'] = {
                'workers': len(self.workers[category])
            };
            resp['workload'].update(self.get_stats(category))
            req_per_worker = resp['workload']['requests'] / resp['workload']['workers']
            resp['workload']['factor'] = req_per_worker
            if resp['workload']['clients'] > resp['workload']['workers']:
                resp['workload']['factor'] += (resp['workload']['clients'] - resp['workload']['workers']) * req_per_worker
            self.log.debug("req/work={0} r={1} c={2} w={3} f={4}".format(req_per_worker, resp['workload']['requests'], resp['workload']['clients'], resp['workload']['workers'], resp['workload']['factor']))

        responses = []
        #print(f"client_id:{client_id} in:{client_id in self.client_responses}")
        if client_id in self.client_responses:
            #print(f"clresp:{repr(self.client_responses[client_id])} empty:{self.client_responses[client_id].empty()}")
            while not self.client_responses[client_id].empty():
                responses.append(self.client_responses[client_id].get(False))
        resp['responses'] = responses;

        return resp

    def cancel(self, category, client_id):
        self.client_cancellations[client_id] = True
        self.translation_request_queues[category].cancel_requests_from(client_id)
        for worker in self.workers[category]:
            if worker.current_client_key == client_id:
                worker.stop()

    def get_log_files(self):
        log_file_table = {}

        all_log_files = []
        for handler in self.log.root.handlers:
            if hasattr(handler, 'baseFilename'):
                log_dir = os.path.dirname(handler.baseFilename)
                log_base_fn = os.path.basename(handler.baseFilename)
                log_files = [f for f in listdir(log_dir) if isfile(join(log_dir, f)) and f.startswith(log_base_fn)]
                all_log_files += log_files
        log_file_table['multiserver'] = all_log_files

        for server_category in self.translation_server_config:
            for idx, server in enumerate(self.translation_server_config[server_category]):
                client = Client(server['host'], server['port'])
                try:
                    resp = client.get_log_files()
                    resp_json = json.loads(resp)
                    if server_category in log_file_table:
                        log_file_table[server_category] += resp_json['log_files']
                    else:
                        log_file_table[server_category] = resp_json['log_files']
                except BaseException as err:
                    self.log.info("An error has occurred when the multiserver performed a GET_LOG_FILES query for the '{0}' category: '{1}'".format(server_category, err))
        return log_file_table

    def get_log_file_content(self, requested_file, page=1):
        page_count = 1
        log_file_content = ''

        for handler in self.log.root.handlers:
            if hasattr(handler, 'baseFilename'):
                log_dir = os.path.dirname(handler.baseFilename)
                log_base_fn = os.path.basename(handler.baseFilename)
                actual_log_file = "{0}/{1}".format(log_dir, requested_file)
                if log_base_fn in requested_file and os.path.isfile(actual_log_file):
                    line_in_page = 0
                    start = (page - 1) * PAGE_SIZE
                    stop = start + PAGE_SIZE
                    with open(actual_log_file, 'r') as f:
                        for line, str_line in enumerate(f):
                            if line >= start and line < stop:
                                log_file_content += str_line
                            line_in_page += 1
                            if line_in_page == PAGE_SIZE:
                                page_count += 1
                                line_in_page = 0
                    return {
                        'content': log_file_content, 
                        'page': page,
                        'pageCount': page_count
                    }
        
        for server_category in self.translation_server_config:
            for idx, server in enumerate(self.translation_server_config[server_category]):
                client = Client(server['host'], server['port'])
                try:
                    resp = client.get_log_file(requested_file, page)
                    resp_json = json.loads(resp)
                    if resp_json['status'] == 'OK' and 'content' in resp_json:
                        return {
                            'content': resp_json['content'], 
                            'page': resp_json['page'],
                            'pageCount': resp_json['pageCount']
                        }
                except BaseException as err:
                    self.log.info("An error has occurred when the multiserver performed a GET_LOG_FILE query for the '{0}' category: '{1}'".format(server_category, err))

        return {
            'content': log_file_content, 
            'page': page,
            'pageCount': page_count
        }

EOM = "==== EOM ===="

class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):

    daemon_threads = True
    allow_reuse_address = True

    def make_request_handler(self, manager, log):
        class ServerRequestHandler(socketserver.BaseRequestHandler, object):

            def __init__(self, *args, **kwargs):
                self.manager = manager
                self.log = log
                super(ServerRequestHandler, self).__init__(*args, **kwargs)

            def handle(self):

                start_request = timeit.default_timer()
                self.log.debug("Handling request....")

                # Read until EOM delimiter is met. 
                total_data = []
                data = ''
                while True:
                    data = self.request.recv(4096).decode('utf-8')
                    if EOM in data:
                        total_data.append(data[:data.find(EOM)])
                        break
                    total_data.append(data)
                    if len(total_data)>1:
                        #check if EOM was split
                        last_pair = total_data[-2] + total_data[-1]
                        if EOM in last_pair:
                            total_data[-2] = last_pair[:last_pair.find(EOM)]
                            total_data.pop()
                            break
                str_data = ''.join(total_data)

                self.log.debug("Request to server={0}".format(str_data))
                response = {}
                json_data = None
                try:
                    json_data = json.loads(str_data)
                except Exception as e:
                    self.log.info("Invalid JSON data. Request ignored.")

                if json_data:    
                    if json_data['type'] == 'get_log_files':
                        response = self.manager.get_log_files()
                    elif json_data['type'] == 'get_log_file':
                        response = self.manager.get_log_file_content(json_data['file'][1:], int(json_data['page']))
                    else:
                        lang_pair = None
                        if json_data['src_lang'] and json_data['tgt_lang']:
                            lang_pair = "{0}-{1}".format(json_data['src_lang'], json_data['tgt_lang'])
                        model = None
                        if json_data['model']:
                            model = json_data['model']
                        category = None
                        if lang_pair and model:
                            category = "{0}_{1}".format(lang_pair, model)
                        key = "{0}-{1}".format(json_data['session_id'], json_data['client_tab_id']) 
                        self.log.debug("lang_pair={0} model={1} category={2}".format(lang_pair, model, category))
                        if json_data['type'] == 'translate':
                            self.log.info("TRANSLATE from {0}".format(key))
                            splitter_cmd = self.manager.text_to_sentences_splitter % json_data['text'].replace("'", "'\\''")
                            splitter_cmd = splitter_cmd.replace("$lang_source", json_data['src_lang'])
                            splitter_cmd = splitter_cmd.replace("$lang_target", json_data['tgt_lang'])
                            self.log.info("splitter_cmd=%s" % splitter_cmd)
                            start_cmd = timeit.default_timer()
                            splitter_output = subprocess.check_output(splitter_cmd, shell=True).decode('utf-8')
                            self.log.info(
                                "Splitter cmd processed in {} s.".format(
                                    timeit.default_timer() - start_cmd))
                            self.log.info("splitter_output=%s" % splitter_output)
                            sentences = splitter_output.splitlines()

                            self.manager.client_cancellations[key] = False 
                            
                            sentence_count = len(sentences)
                            if key in self.manager.client_responses:
                                response_queue = self.manager.client_responses[key]
                            else:
                                response_queue = queue.Queue()
                                self.manager.client_responses[key] = response_queue

                            json_resp_str = '{"articleId": '
                            json_resp_str += str(int(time.time())) 
                            json_resp_str += ', "sentences":['
                            delim = ""
                            for sentence in sentences:
                                json_resp_str += '{0}"{1}"'.format(delim, sentence.replace('"', '\\\"'))
                                delim = ","
                            json_resp_str += "]}"
                            json_resp = json.loads(json_resp_str)
                            response_queue.put(json_resp)

                            self.log.debug("sentenceCount={0}".format(len(sentences)))
                            for index, sentence in enumerate(json_resp['sentences']):
                                translate_sentence_request = dict(json_data)
                                del translate_sentence_request['text']
                                translate_sentence_request['sentence'] = sentence
                                translate_sentence_request['sentence_number'] = index
                                translate_sentence_request['lang_source'] = translate_sentence_request['src_lang']
                                translate_sentence_request['lang_target'] = translate_sentence_request['tgt_lang']
                                translate_sentence_request['keepalive'] = time.time()
                                self.manager.translation_request_queues[category].put(translate_sentence_request)
                            self.manager.translation_request_queues[category].redistribute_requests()
                            response = {'msg': 'All sentences have been queued.'}
                        elif json_data['type'] == 'poll':
                            self.log.debug("POLL from {0}".format(key))   
                            response = self.manager.poll(category, key)
                        elif json_data['type'] == 'cancelTranslation':
                            self.log.info("CANCEL from {0}".format(key))
                            self.manager.cancel(category, key)
                            response = {'msg': 'Translation request has been cancelled.'}
                        else:
                            response = {'msg': 'Unknown request type. Request has been ignored.'}

                    self.log.debug("Request processed in {0} s. by {1}".format(timeit.default_timer() - start_request, threading.current_thread().name))
                    response = json.dumps(response)
                    self.request.sendall(response.encode('utf-8'))

        return ServerRequestHandler

    def __init__(self, server_address, translation_server_config, text_to_sentences_splitter, log):
        self.manager = Manager(translation_server_config, text_to_sentences_splitter, log)
        handler_class = self.make_request_handler(self.manager, log)
        socketserver.TCPServer.__init__(self, server_address, handler_class)

def do_start_server(config_file, log_config):
    if log_config:
        logging.config.fileConfig(log_config)
    log = logging.getLogger("default")

    config = json.load(open(config_file))
    server = Server((config['host'], int(config['port'])), config['servers'], config['text_to_sentences_splitter'], log)
    ip, port = server.server_address
    log.info("Start listening for requests on {0}:{1}...".format( socket.gethostname(), port))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()

    sys.exit(0)
