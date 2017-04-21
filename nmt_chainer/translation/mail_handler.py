#!/usr/bin/env python
"""mail_handler.py: Daemon that monitors an IMAP account for messages containing translation requests."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import argparse
from collections import namedtuple
import daemon
import email
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
import gettext
import httplib
import imaplib
import json
import locale
import logging
import logging.config
import re
import smtplib
import subprocess
import tempfile
import time
import urllib

from nmt_chainer._version import __version__ as knmt_version
from nmt_chainer.translation.client import Client

list_resp_pattern = re.compile('(.*?) "(.*)" (.*)')
subject_pattern = re.compile('([a-zA-Z][a-zA-Z])_([a-zA-Z][a-zA-Z])')

MailRequest = namedtuple('MailRequest', 'uid date ffrom subject body')


def send_mail(config_file, to, subject, text):
    config = json.load(open(config_file))
    resp_msg = MIMEMultipart()
    resp_msg['From'] = config['smtp']['from']
    resp_msg['To'] = to
    resp_msg['Subject'] = subject

    resp_msg.attach(MIMEText(text, 'plain', 'utf-8'))

    smtp_server = smtplib.SMTP(config['smtp']['host'], config['smtp']['port'])
    smtp_server.starttls()
    smtp_server.login(config['smtp']['user'], config['smtp']['password'])
    text = resp_msg.as_string().encode('ascii')
    smtp_server.sendmail(config['smtp']['from'], to, text)
    smtp_server.quit()


def split_text_into_paragraphes(text):
    paragraphes = text.split("\n")
    paragraphes = filter(lambda p: p != '', paragraphes)
    return paragraphes


def split_text_into_sentences(config_file, logger, src_lang, tgt_lang, text):
    config = json.load(open(config_file))
    sentences = []
    params = urllib.urlencode({'lang_source': src_lang, 'lang_target': tgt_lang, 'text': text})
    headers = {'Content-type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}
    conn = httplib.HTTPConnection("{0}".format(config['text_splitter']['host']))
    try:
        conn.request('POST', config['text_splitter']['path'], params, headers)
        resp = conn.getresponse()
        data = resp.read()
        json_data = json.loads(data)
        sentences = json_data['sentences']
    except Exception as ex:
        logger.error(ex)
    finally:
        conn.close()
    return sentences


def translate_sentence(config_file, src_lang, tgt_lang, sentence):
    config = json.load(open(config_file))
    lang_pair = '{0}-{1}'.format(src_lang, tgt_lang)
    server_data = config['servers'][lang_pair]
    client = Client(server_data['host'], server_data['port'])
    resp = client.query(sentence)
    json_resp = json.loads(resp)
    translation = json_resp['out'].encode('utf-8')

    if tgt_lang in ['ja', 'zh']:
        translation = translation.replace(' ', '')

    translation = translation.replace('&quot;', '"')
    translation = translation.replace('&apos;', "'")

    return translation


def parse_list_response(line):
    flags, delimiter, mailbox_name = list_resp_pattern.match(line).groups()
    mailbox_name = mailbox_name.strip('"')
    return (flags, delimiter, mailbox_name)


def init_localization(config_file, logger):
    logger.info("Initializating localization.")
    config = json.load(open(config_file))
    ui_locale = config.get('ui_locale', '')
    logger.info("Interface locale: {0}".format(ui_locale))
    locale.setlocale(locale.LC_ALL, ui_locale.encode('ascii'))
    filename = "res/messages_%s.mo" % locale.getlocale()[0][0:2]

    try:
        logging.debug("Opening message file %s for locale %s", filename, locale.getlocale()[0])
        trans = gettext.GNUTranslations(open(filename, "rb"))
    except IOError:
        logging.debug("Locale not found. Using default messages")
        trans = gettext.NullTranslations()

    trans.install()


def prepare_mailboxes(config_file, logger):
    logger.info("Preparing mailboxes.")
    config = json.load(open(config_file))
    mail = None
    try:
        mail = imaplib.IMAP4_SSL(config['imap']['host'], config['imap']['port'])
        mail.login(config['imap']['user'], config['imap']['password'])

        type, data = mail.list()
        processed_mailbox = config['imap']['processed_request_mailbox']
        ignored_mailbox = config['imap']['ignored_request_mailbox']
        is_processed_request_mailbox_found = False
        is_ignored_request_mailbox_found = False
        for line in data:
            flags, delimiter, mailbox_name = parse_list_response(line)
            if not is_processed_request_mailbox_found and mailbox_name == processed_mailbox:
                is_processed_request_mailbox_found = True
            if not is_ignored_request_mailbox_found and mailbox_name == ignored_mailbox:
                is_ignored_request_mailbox_found = True
        if not is_processed_request_mailbox_found:
            mail.create(processed_mailbox)
        if not is_ignored_request_mailbox_found:
            mail.create(ignored_mailbox)

    except Exception, ex:
        raise

    finally:
        if mail is not None:
            mail.logout()


# Taken from https://gist.githubusercontent.com/miohtama/5389146/raw/03362b9aa72f9d3a4e68fb99af40a0ef160e0ec5/gistfile1.py

def get_decoded_email_body(message_body):
    """ Decode email body.
    Detect character set if the header is not set.
    We try to get text/plain, but if there is not one then fallback to text/html.
    :param message_body: Raw 7-bit message body input e.g. from imaplib. Double encoded in quoted-printable and latin-1
    :return: Message body as unicode string
    """

    msg = email.message_from_string(message_body)

    text = ""
    if msg.is_multipart():
        html = None
        for part in msg.get_payload():

            print "%s, %s" % (part.get_content_type(), part.get_content_charset())

            if part.get_content_charset() is None:
                # We cannot know the character set, so return decoded "something"
                text = part.get_payload(decode=True)
                continue

            charset = part.get_content_charset()

            if part.get_content_type() == 'text/plain':
                text = unicode(part.get_payload(decode=True), str(charset), "ignore").encode('utf8', 'replace')

            if part.get_content_type() == 'text/html':
                html = unicode(part.get_payload(decode=True), str(charset), "ignore").encode('utf8', 'replace')

        if text is not None:
            return text.strip()
        else:
            return html.strip()
    else:
        text = unicode(msg.get_payload(decode=True), msg.get_content_charset(), 'ignore').encode('utf8', 'replace')
        return text.strip()


def fetch_requests(config_file, logger, mail):
    requests = []
    config = json.load(open(config_file))

    type, data = mail.uid('SEARCH', None, 'ALL')
    mail_uids = data[0]
    uid_list = mail_uids.split()
    if (len(uid_list) > 0):
        first_email_uid = int(uid_list[0])
        last_email_uid = int(uid_list[-1])

        for mail_uid in range(first_email_uid, last_email_uid + 1):
            email_from = None
            try:
                type, data = mail.uid('FETCH', mail_uid, '(RFC822)')

                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_string(response_part[1])
                        email_date = msg['date']
                        email_from = msg['from']
                        email_subject = msg['subject']
                        email_body = get_decoded_email_body(response_part[1])

                        subject_match = subject_pattern.match(email_subject)
                        if subject_match:
                            src_lang = subject_match.group(1).lower()
                            tgt_lang = subject_match.group(2).lower()
                            lang_pair = '{0}-{1}'.format(src_lang, tgt_lang)

                            if lang_pair not in config['servers']:
                                raise Exception('Unsupported language pair: {0}'.format(lang_pair))

                            logger.info('Queuing request...')
                            logger.info("Uid: {0}".format(mail_uid))
                            logger.info("Date: {0}".format(email_date))
                            logger.info("From: {0}".format(email_from))
                            logger.info("Subject: {0}".format(email_subject))

                            reply = _("Translation request received.\n\n{0}\n\nPlease wait...\n\n---------- Original text ----------\n\nLanguage pair: {1}-{2}\n\n{3}\n\n-- The content of this message has been generated by KNMT {4}. --")
                            if (len(requests) <= 1):
                                queue_msg = _("It will be processed shortly.")
                            else:
                                queue_msg = _("Requests before you: {0}").format(len(requests) - 1)

                            subject = _('Translation request received: {0}').format(email_subject)
                            send_mail(config_file, email_from, subject.decode('utf-8'),
                                      reply.format(queue_msg, src_lang, tgt_lang, email_body, knmt_version))

                            requests.append(MailRequest(mail_uid, email_date, email_from, email_subject, email_body.replace('\r', '')))
            except Exception, ex_msg:
                logger.info("Error: {0}\n\nMoving message to Ignored mailbox\n\n".format(ex_msg))
                mail.uid('COPY', mail_uid, config['imap']['ignored_request_mailbox'])
                mail.uid('STORE', mail_uid, '+FLAGS', '\\Deleted')
                mail.expunge()
                if email_from is not None:
                    reply = _("Error: {0}\n\nMoving message to Ignored mailbox\n\n").format(ex_msg)
                    subject = _('Translation error: {0}').format(email_subject)
                    send_mail(config_file, email_from, subject.decode('utf-8'), reply)

    return requests


def process_mail(config_file, logger):
    while True:
        config = json.load(open(config_file))

        mail = None
        try:
            mail = imaplib.IMAP4_SSL(config['imap']['host'], config['imap']['port'])
            mail.login(config['imap']['user'], config['imap']['password'])
            mail.select(config['imap']['incoming_request_mailbox'])

            requests = fetch_requests(config_file, logger, mail)

            for req in requests:
                try:
                    logger.info("Processing mail...")
                    logger.info("Uid: {0}".format(req.uid))
                    logger.info("Date: {0}".format(req.date))
                    logger.info("From: {0}".format(req.ffrom))
                    logger.info("Subject: {0}".format(req.subject))

                    subject_match = subject_pattern.match(req.subject)
                    src_lang = subject_match.group(1).lower()
                    tgt_lang = subject_match.group(2).lower()
                    lang_pair = '{0}-{1}'.format(src_lang, tgt_lang)
                    logger.info('Text: {0}\n'.format(req.body))

                    paragraphes = split_text_into_paragraphes(req.body)

                    translation = ''
                    for paragraph in paragraphes:

                        sentences = split_text_into_sentences(config_file, logger, src_lang, tgt_lang, paragraph)

                        if tgt_lang in ['ja', 'zh']:
                            translation += '\xe3\x80\x80'  # Ideographic space.

                        for sentence in sentences:
                            translated_sentence = translate_sentence(config_file, src_lang, tgt_lang, sentence.encode('utf-8'))
                            translation += translated_sentence.rstrip()
                        translation += "\n\n"

                    if 'post_processing_cmd' in config and lang_pair in config['post_processing_cmd']:
                        translation_file = tempfile.NamedTemporaryFile()
                        try:
                            translation_file.write(translation)
                            translation_file.seek(0)
                            cmd = config['post_processing_cmd'][lang_pair].replace("$FILE", translation_file.name)
                            translation = subprocess.check_output(cmd, shell=True)
                        finally:
                            translation_file.close()

                    logger.info("Translation: {0}".format(translation))

                    reply = _("{0}\n\n---------- Original text ----------\n\n{1}\n\n-- The content of this message has been generated by KNMT {2}. --")

                    subject = _('Translation result: {0}').format(req.subject)
                    send_mail(config_file, req.ffrom, subject.decode('utf-8'),
                              reply.format(translation, req.body, knmt_version))

                    logger.info("Moving message to Processed mailbox.\n\n")
                    mail.uid('COPY', req.uid, config['imap']['processed_request_mailbox'])
                    mail.uid('STORE', req.uid, '+FLAGS', '\\Deleted')
                    mail.expunge()

                except Exception, ex_msg:
                    logger.info('Error: {0}'.format(ex_msg))
                    logger.info("Moving message to Ignored mailbox.\n\n")
                    mail.uid('COPY', req.uid, config['imap']['ignored_request_mailbox'])
                    mail.uid('STORE', req.uid, '+FLAGS', '\\Deleted')
                    mail.expunge()

                    reply = "Error: {0}\n\nYour message has been discarded.\n\n-- The content of this message has been generated by KNMT {1}. --"
                    send_mail(config_file, email_from,
                              'Translation result for {0}_{1} request'.format(src_lang, tgt_lang),
                              reply.format(ex_msg, knmt_version))

        except Exception, ex:
            logger.error(ex)

        finally:
            if mail is not None:
                mail.close()
                mail.logout()

        time.sleep(int(config['next_mail_handling_delay']))


def run(args):
    logging.config.fileConfig(args.log_config)
    logger = logging.getLogger('default')

    config_file = args.config

    context = daemon.DaemonContext(working_directory='.')
    context.files_preserve = [logger.root.handlers[1].stream.fileno()]

    with context:
        try:
            logger.info("Starting mail_handler daemon...")
            init_localization(config_file, logger)
            prepare_mailboxes(config_file, logger)
            process_mail(config_file, logger)
        except Exception, ex:
            logger.error(ex)
        finally:
            logger.info("Stopping mail_handler daemon.")


def command_line(arguments=None):
    parser = argparse.ArgumentParser(description="Launch a translation mail handler daemon.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", nargs="?", help="Main configuration file.", default="conf/mail_handler.json")
    parser.add_argument("--log_config", nargs="?", help="Log configuration file.", default="conf/mail_handler_logging.conf")

    args = parser.parse_args(args=arguments)

    run(args)


if __name__ == "__main__":
    command_line()
