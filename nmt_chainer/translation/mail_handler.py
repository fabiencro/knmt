#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""mail_handler.py: Daemon that monitors an IMAP account for messages containing translation requests."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import argparse
import base64
import codecs
import daemon
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import gettext
import gzip
import http.client
import imaplib
import io
import json
import locale
import logging
import logging.config
import os.path
import re
import smtplib
from socket import gaierror
import socket
import subprocess
import tempfile
import time
import urllib.parse

from nmt_chainer._version import __version__ as knmt_version
from nmt_chainer.translation.client import Client


list_resp_pattern = re.compile('(.*?) "(.*)" (.*)')
subject_pattern = re.compile('([a-zA-Z][a-zA-Z])_([a-zA-Z][a-zA-Z])')


def split_text_into_paragraphes(text, src_lang, tgt_lang):
    if src_lang in ['ja', 'zh'] and '\u3000' in text:
        paragraphes = text.replace('\n', '').split('\u3000')
        paragraphes = filter(lambda p: len(p) > 0, paragraphes)
        return paragraphes

    paragraphes = text.split("\n")
    paragraphes = filter(lambda p: len(p) > 0, paragraphes)
    return paragraphes


# Taken from https://gist.githubusercontent.com/miohtama/5389146/raw/03362b9aa72f9d3a4e68fb99af40a0ef160e0ec5/gistfile1.py
# and adjusted a little bit for Python 3.
def get_decoded_email_body(message_body):
    """ Decode email body.
    Detect character set if the header is not set.
    We try to get text/plain, but if there is not one then fallback to text/html.
    :param message_body: Raw 7-bit message body input e.g. from imaplib. Double encoded in quoted-printable and latin-1
    :return: Message body as unicode string
    """

    msg = email.message_from_bytes(message_body)

    text = ""
    if msg.is_multipart():
        html = None
        for part in msg.get_payload():
            if part.get_content_type() not in ['text/plain', 'text/html']:
                continue

            if part.get_content_charset() is None:
                # We cannot know the character set, so return decoded "something"
                text = part.get_payload(decode=True)
                continue

            charset = part.get_content_charset()
            if part.get_content_type() == 'text/plain':
                text = part.get_payload(None, True)

            if part.get_content_type() == 'text/html':
                html = part.get_payload(None, True)

        if text is not None:
            return text.strip().decode(charset)
        else:
            return html.strip().decode(charset)
    else:
        text = msg.get_payload(None, True)
        return text.strip().decode(msg.get_content_charset())


class MailHandler:

    def __init__(self, config, log_config):
        self.config_file = config
        self.log_config_file = log_config
        self._requests = []

    def _send_mail(self, to, subject, text):
        config = json.load(open(self.config_file))
        resp_msg = MIMEMultipart()
        resp_msg['From'] = config['smtp']['from']

        # Never send a message to a daemon user.
        if to is not None and "daemon" in to.lower():
            return

        resp_msg['To'] = to
        cc = None
        bcc = None
        if 'cc' in config['smtp']:
            resp_msg['Cc'] = config['smtp']['cc']
            cc = config['smtp']['cc'].split(",")
        if 'bcc' in config['smtp']:
            resp_msg['Bcc'] = config['smtp']['bcc']
            bcc = config['smtp']['bcc'].split(",")

        dests = []
        if to:
            dests.append(to)
        if cc:
            dests += cc
        if bcc:
            dests += bcc

        if len(dests) == 0:
            return

        resp_msg['Subject'] = subject
        resp_msg.attach(MIMEText(text, 'plain', 'utf-8'))

        smtp_server = smtplib.SMTP(config['smtp']['host'], config['smtp']['port'])
        smtp_server.starttls()
        smtp_server.login(config['smtp']['user'], config['smtp']['password'])
        text = resp_msg.as_string().encode('ascii')
        smtp_server.sendmail(config['smtp']['from'], dests, text)
        smtp_server.quit()

    def _split_text_into_sentences(self, src_lang, tgt_lang, text):
        if text.strip() == '':
            return[text]

        config = json.load(open(self.config_file))

        if 'text_splitter' not in config:
            return [text]

        splitter_host = None
        splitter_path = None
        for langs in config['text_splitter'].keys():
            if langs == 'default':
                continue
            if src_lang in langs.split(','):
                splitter_host = config['text_splitter'][langs]['host']
                splitter_path = config['text_splitter'][langs]['path']
                break
        if splitter_host is None:
            if 'default' in config['text_splitter']:
                splitter_host = config['text_splitter']['default']['host']
                splitter_path = config['text_splitter']['default']['path']
            else:
                return [text]

        sentences = []
        params = urllib.parse.urlencode({'lang_source': src_lang.encode('utf-8'), 'lang_target': tgt_lang.encode('utf-8'), 'text': text.encode('utf-8')})
        headers = {'Content-type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}

        conn = http.client.HTTPConnection(splitter_host)
        try:
            conn.request('POST', splitter_path, params, headers)
            resp = conn.getresponse()
            data = resp.read()
            json_data = json.loads(data.decode('utf-8'))
            sentences = json_data['sentences']
        except Exception as ex:
            self.logger.exception(ex)
            self._send_mail(None, "Mail-Handler - ERROR", ex)
        finally:
            conn.close()
        return sentences

    def _translate_sentence(self, src_lang, tgt_lang, sentence):
        if sentence.strip() == '':
            return ''

        # self.logger.info("_translate_sentence src_lang={0} tgt_lang={1} sentence=@@@{2}@@@".format(src_lang, tgt_lang, sentence.decode('utf-8')))
        config = json.load(open(self.config_file))
        lang_pair = '{0}-{1}'.format(src_lang, tgt_lang)
        server_data = config['servers'][lang_pair]
        client = Client(server_data['host'], server_data['port'])
        resp = client.query(sentence,
                            nb_steps_ratio=2.5,
                            # beam_score_length_normalization='google',
                            # beam_score_length_normalization_strength=0.2,
                            # post_score_length_normalization='google',
                            # post_score_length_normalization_strength=0.2,
                            # beam_score_coverage_penalty='google',
                            # beam_score_coverage_penalty_strength=0.2,
                            post_score_coverage_penalty='google',
                            post_score_coverage_penalty_strength=0.01
                            )
        # self.logger.info("resp={0}".format(resp))
        json_resp = json.loads(resp)

        if 'out' not in json_resp:
            trans_err_msg = 'This sentence @@@{0}@@@ could not be translated from {1} to {2}.'.format(sentence, src_lang, tgt_lang)
            self.logger.error(trans_err_msg)
            self._send_mail(None, "Mail-Handler - ERROR", trans_err_msg)
            return ''

        translation = json_resp['out']

        if tgt_lang in ['ja', 'zh']:
            translation = translation.replace(' ', '')

        translation = translation.replace('&quot;', '"')
        translation = translation.replace('&apos;', "'")

        return translation

    def _parse_list_response(self, line):
        flags, delimiter, mailbox_name = list_resp_pattern.match(line.decode('utf-8')).groups()
        mailbox_name = mailbox_name.strip('"')
        return (flags, delimiter, mailbox_name)

    def _init_localization(self):
        self.logger.info("Initializating localization.")
        config = json.load(open(self.config_file))
        ui_locale = config.get('ui_locale', '')
        self.logger.info("Interface locale: {0}".format(ui_locale))
        locale.setlocale(locale.LC_ALL, ui_locale)
        filename = "res/messages_%s.mo" % locale.getlocale()[0][0:2]

        try:
            logging.debug("Opening message file %s for locale %s", filename, locale.getlocale()[0])
            trans = gettext.GNUTranslations(open(filename, "rb"))
        except IOError:
            logging.debug("Locale not found. Using default messages")
            trans = gettext.NullTranslations()

        trans.install()

    def _prepare_mailboxes(self):
        self.logger.info("Preparing mailboxes.")
        config = json.load(open(self.config_file))
        mail = None
        try:
            mail = imaplib.IMAP4_SSL(config['imap']['host'], config['imap']['port'])
            mail.login(config['imap']['user'], config['imap']['password'])

            typ, data = mail.list()
            processed_mailbox = config['imap']['processed_request_mailbox']
            ignored_mailbox = config['imap']['ignored_request_mailbox']
            is_processed_request_mailbox_found = False
            is_ignored_request_mailbox_found = False
            for line in data:
                flags, delimiter, mailbox_name = self._parse_list_response(line)
                if not is_processed_request_mailbox_found and mailbox_name == processed_mailbox:
                    is_processed_request_mailbox_found = True
                if not is_ignored_request_mailbox_found and mailbox_name == ignored_mailbox:
                    is_ignored_request_mailbox_found = True
            if not is_processed_request_mailbox_found:
                mail.create(processed_mailbox)
            if not is_ignored_request_mailbox_found:
                mail.create(ignored_mailbox)

        except Exception as ex:
            raise

        finally:
            if mail is not None:
                mail.logout()

    def _load_requests(self):
        config = json.load(open(self.config_file))
        requests_filename = "{0}_requests.json.gz".format(config['id'])
        if os.path.isfile(requests_filename):
            with gzip.open(requests_filename, 'r') as input_file:
                str_input = input_file.read().decode('utf-8')
                self._requests = json.loads(str_input)

    def _save_requests(self):
        config = json.load(open(self.config_file))
        requests_filename = "{0}_requests.json.gz".format(config['id'])
        if self._requests:
            with gzip.open(requests_filename, 'w') as output_file:
                output_file.write(json.dumps(self._requests, output_file, ensure_ascii=False).encode('utf-8'))
        else:
            if (os.path.isfile(requests_filename)):
                os.remove(requests_filename)

    def _enqueue_request(self, req):
        self._requests.append(req)
        self._save_requests()

    def _dequeue_request(self):
        self._requests.pop(0)
        self._save_requests()

    def _fetch_requests(self, mail):
        config = json.load(open(self.config_file))

        if self._first_time:
            self._load_requests()
            if len(self._requests) > 0:
                self.logger.info("Previous {0} pending requests found on disk.".format(len(self._requests)))
                return

        self._requests = []

        typ, data = mail.uid('SEARCH', None, 'ALL')
        mail_uids = data[0]
        uid_list = mail_uids.split()
        if (len(uid_list) > 0):
            first_email_uid = int(uid_list[0])
            last_email_uid = int(uid_list[-1])

            for mail_uid in range(first_email_uid, last_email_uid + 1):
                email_from = None
                try:
                    typ, data = mail.uid('FETCH', str(mail_uid), '(RFC822)')

                    for response_part in data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            email_date = msg['date']
                            email_from = msg['from']
                            email_subject = msg['subject']
                            email_body = get_decoded_email_body(response_part[1])

                            # Ignore warning messages from the MAILER-DAEMON as they are
                            # not translation requests.
                            if "daemon" in email_from.lower():
                                raise ValueError("MAILER-DAEMON messages should be ignored")

                            self.logger.info("subject_match={0}".format(email_subject))
                            subject_match = subject_pattern.match(email_subject)
                            if subject_match:
                                src_lang = subject_match.group(1).lower()
                                tgt_lang = subject_match.group(2).lower()
                                lang_pair = '{0}-{1}'.format(src_lang, tgt_lang)

                                if lang_pair not in config['servers']:
                                    allowed_lang_pairs = ", ".join(sorted(map((lambda x: x.replace('-', '_')), config['servers'])))
                                    raise Exception('Unsupported language pair: {0}. Allowed values are: {1}'.format(lang_pair, allowed_lang_pairs))

                                self.logger.info('Queuing request...')
                                self.logger.info("Uid: {0}".format(mail_uid))
                                self.logger.info("Date: {0}".format(email_date))
                                self.logger.info("From: {0}".format(email_from))
                                self.logger.info("Subject: {0}".format(email_subject))

                                reply = _("Translation request received.\n\n{0}Please wait...\n\n---------- Original text ----------\n\nLanguage pair: {1}-{2}\n\n{3}\n\n-- The content of this message has been generated by KNMT {4}. --")
                                queue_msg = ''
                                if (len(self._requests) > 0):
                                    queue_msg = _("Requests before you: {0}\n\n").format(len(self._requests))

                                subject = _('Translation request received: {0}').format(email_subject)
                                self._send_mail(email_from, subject, reply.format(queue_msg, src_lang, tgt_lang, email_body, knmt_version))

                                req = {'uid': mail_uid, 'date': email_date, 'ffrom': email_from, 'subject': email_subject, 'body': email_body.replace('\r', '')}
                                self._enqueue_request(req)

                            else:
                                raise Exception('Incorrect protocol or unknown language pair: {0}'.format(email_subject))
                except Exception as ex_msg:
                    self.logger.exception("Error: {0}\n\nMoving message to Ignored mailbox\n\n".format(ex_msg))
                    mail.uid('COPY', str(mail_uid), config['imap']['ignored_request_mailbox'])
                    mail.uid('STORE', str(mail_uid), '+FLAGS', '\\Deleted')
                    mail.expunge()

                    if "MAILER-DAEMON" not in str(ex_msg):
                        reply = _("Error: {0}\n\nMoving message to Ignored mailbox\n\n").format(ex_msg)
                        subject = _('Translation error: {0}').format(email_subject)
                        self._send_mail(email_from, subject, reply)

    def _process_mail(self):
        self._first_time = True
        while True:
            config = json.load(open(self.config_file))

            mail = None
            try:
                mail = imaplib.IMAP4_SSL(config['imap']['host'], config['imap']['port'])
                mail.login(config['imap']['user'], config['imap']['password'])
                mail.select(config['imap']['incoming_request_mailbox'])

                self._fetch_requests(mail)

                while self._requests:
                    req = self._requests[0]
                    try:
                        self.logger.info("Processing mail...")
                        self.logger.info("Uid: {0}".format(req['uid']))
                        self.logger.info("Date: {0}".format(req['date']))
                        self.logger.info("From: {0}".format(req['ffrom']))
                        self.logger.info("Subject: {0}".format(req['subject']))

                        # Ignore warning messages from the MAILER-DAEMON as they are
                        # not translation requests.
                        if "daemon" in req['ffrom'].lower():
                            raise ValueError("MAILER-DAEMON messages should be ignored")

                        subject_match = subject_pattern.match(req['subject'])
                        src_lang = subject_match.group(1).lower()
                        tgt_lang = subject_match.group(2).lower()
                        lang_pair = '{0}-{1}'.format(src_lang, tgt_lang)
                        self.logger.info('Text: {0}\n'.format(req['body']))

                        paragraphes = split_text_into_paragraphes(req['body'], src_lang, tgt_lang)

                        translation = ''
                        for paragraph in paragraphes:
                            sentences = self._split_text_into_sentences(src_lang, tgt_lang, paragraph)

                            if tgt_lang in ['ja', 'zh']:
                                translation += '\u3000'  # Ideographic space.

                            for sentence in sentences:
                                translated_sentence = self._translate_sentence(src_lang, tgt_lang, sentence)
                                translation += translated_sentence.rstrip()
                                if tgt_lang not in ['ja', 'zh']:
                                    translation += ' '
                            translation += "\n\n"

                        if 'post_processing_cmd' in config and lang_pair in config['post_processing_cmd']:
                            translation_file = tempfile.NamedTemporaryFile()
                            try:
                                translation_file.write(translation.encode('utf-8'))
                                translation_file.seek(0)
                                cmd = config['post_processing_cmd'][lang_pair].replace("$FILE", translation_file.name)
                                translation = subprocess.check_output(cmd, shell=True).decode('utf-8')
                            finally:
                                translation_file.close()

                        self.logger.info("Translation: {0}".format(translation))

                        reply = _("{0}\n\n---------- Original text ----------\n\n{1}\n\n-- The content of this message has been generated by KNMT {2}. --")

                        subject = _('Translation result: {0}').format(req['subject'])
                        self._send_mail(req['ffrom'], subject, reply.format(translation, req['body'], knmt_version))

                        self.logger.info("Moving message to Processed mailbox.\n\n")
                        mail.uid('COPY', str(req['uid']), config['imap']['processed_request_mailbox'])
                        mail.uid('STORE', str(req['uid']), '+FLAGS', '\\Deleted')
                        mail.expunge()

                    except Exception as ex_msg:
                        self.logger.exception('Error: {0}'.format(ex_msg))
                        self.logger.info("Moving message to Ignored mailbox.\n\n")
                        mail.uid('COPY', str(req['uid']), config['imap']['ignored_request_mailbox'])
                        mail.uid('STORE', str(req['uid']), '+FLAGS', '\\Deleted')
                        mail.expunge()

                        if "MAILER-DAEMON" not in str(ex_msg):
                            reply = "Error: {0}\n\nYour message has been discarded.\n\n-- The content of this message has been generated by KNMT {1}. --"
                            self._send_mail(req['ffrom'],
                                            'Translation result for {0}_{1} request'.format(src_lang, tgt_lang),
                                            reply.format(ex_msg, knmt_version))

                    self._dequeue_request()

            except gaierror:
                self.logger.error("Cannot reach IMAP server.  Will try again later.")

            except Exception as ex:
                self.logger.exception(ex)
                try:
                    self._send_mail(None, "Mail-Handler - ERROR", ex)
                except Exception as ex2:
                    self.logger.exception("Could not send mail notification because of: {0}".format(ex2))

            finally:
                if mail is not None:
                    try:
                        mail.close()
                        mail.logout()
                    except Exception as ex3:
                        self.logger.exception("Could not close connection.  Most likely that it was not properly opened in the first place so let's ignore this.")

            self._first_time = False
            time.sleep(int(config['next_mail_handling_delay']))

    def run(self):
        logging.config.fileConfig(self.log_config_file)
        self.logger = logging.getLogger('default')

        context = daemon.DaemonContext(working_directory='.')
        context.files_preserve = [self.logger.root.handlers[1].stream.fileno()]

        with context:
            try:
                start_msg = "Starting mail_handler daemon on {0}...".format(socket.gethostname())
                self.logger.info(start_msg)
                self._send_mail(None, "Mail-Handler - STARTED", start_msg)
                self._init_localization()
                self._prepare_mailboxes()
                self._process_mail()
            except Exception as ex:
                self.logger.exception(ex)
                self._send_mail(None, "Mail-Handler - ERROR", ex)
            finally:
                stop_msg = "Stopping mail_handler daemon on {0}.".format(socket.gethostname())
                self.logger.info(stop_msg)
                self._send_mail(None, "Mail-Handler - STOPPED", stop_msg)


def command_line(arguments=None):
    parser = argparse.ArgumentParser(description="Launch a translation mail handler daemon.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", nargs="?", help="Main configuration file.", default="conf/mail_handler.json")
    parser.add_argument("--log_config", nargs="?", help="Log configuration file.", default="conf/mail_handler_logging.conf")

    args = parser.parse_args(args=arguments)

    handler = MailHandler(args.config, args.log_config)
    handler.run()


if __name__ == "__main__":
    command_line()
