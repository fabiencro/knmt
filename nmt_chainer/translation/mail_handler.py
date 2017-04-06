#!/usr/bin/env python
"""mail_handler.py: Daemon that monitors an IMAP account for translation requests."""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "bergeron@pa.jst.jp"
__status__ = "Development"

import daemon
import email
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
import httplib
import imaplib
import json
import re
import smtplib
import time
import urllib

from nmt_chainer.translation.client import Client

CONFIG_FILE = 'conf/mail_handler.json'

list_resp_pattern = re.compile('(.*?) "(.*)" (.*)')


def send_mail(to, subject, text):
    config = json.load(open(CONFIG_FILE))
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


def split_text_into_sentences(src_lang, tgt_lang, text):
    config = json.load(open(CONFIG_FILE))
    sentences = None
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
        print "ex={0}".format(ex)
    finally:
        conn.close()
    return sentences


def translate_sentence(src_lang, tgt_lang, sentence):
    config = json.load(open(CONFIG_FILE))
    server_data = config['languages']['{0}-{1}'.format(src_lang, tgt_lang)]['server']
    client = Client(server_data['host'], server_data['port'])
    resp = client.query(sentence)
    json_resp = json.loads(resp)
    translation = json_resp['out'].encode('utf-8')
    if tgt_lang in ['ja', 'zh']:
        translation = translation.replace(' ', '')
    return translation


def parse_list_response(line):
    flags, delimiter, mailbox_name = list_resp_pattern.match(line).groups()
    mailbox_name = mailbox_name.strip('"')
    return (flags, delimiter, mailbox_name)


def prepare_mailboxes():
    config = json.load(open(CONFIG_FILE))
    mail = None
    try:
        mail = imaplib.IMAP4_SSL(config['imap']['host'])
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

    except Exception, e:
        print str(e)

    finally:
        if mail is not None:
            mail.logout()


def process_mail():
    while True:
        config = json.load(open(CONFIG_FILE))
        mail = None
        try:
            mail = imaplib.IMAP4_SSL(config['imap']['host'])
            mail.login(config['imap']['user'], config['imap']['password'])
            mail.select(config['imap']['incoming_request_mailbox'])

            type, data = mail.search(None, 'ALL')
            mail_ids = data[0]
            id_list = mail_ids.split()
            first_email_id = int(id_list[0])
            latest_email_id = int(id_list[-1])
            for i in range(first_email_id, latest_email_id + 1):
                type, data = mail.fetch(i, '(RFC822)')

                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_string(response_part[1])
                        email_subject = msg['subject']
                        email_from = msg['from']
                        if msg.is_multipart():
                            for child_msg in msg.get_payload():
                                if child_msg.get_content_type() == 'text/plain':
                                    email_body = child_msg.get_payload(decode=True)
                        else:
                            email_body = msg.get_payload(decode=True)
                        subject_pattern = re.compile('([a-z][a-z])_([a-z][a-z])')
                        subject_match = subject_pattern.match(email_subject)
                        if subject_match:
                            src_lang = subject_match.group(1)
                            tgt_lang = subject_match.group(2)
                            print "Request {0}_{1} from {2}:".format(src_lang, tgt_lang, email_from)
                            print 'Message: {0}\n'.format(email_body)

                            sentences = split_text_into_sentences(src_lang, tgt_lang, email_body)

                            translation = ''
                            for sentence in sentences:
                                translated_sentence = translate_sentence(src_lang, tgt_lang, sentence.encode('utf-8'))
                                translation += translated_sentence

                            print "translation={0}".format(translation)
                            send_mail(email_from, 'Translation', translation)

                            mail.copy(i, config['imap']['processed_request_mailbox'])
                            mail.store(i, '+FLAGS', '\\Deleted')
                            mail.expunge()
                        else:
                            print "Put this message in the ignored mailbox"
                            print 'From : ' + email_from + '\n'
                            print 'Subject : ' + email_subject + '\n'

                            mail.copy(i, config['imap']['ignored_request_mailbox'])
                            mail.store(i, '+FLAGS', '\\Deleted')
                            mail.expunge()

        except Exception, e:
            print str(e)

        finally:
            if mail is not None:
                mail.close()
                mail.logout()

        time.sleep(int(config['next_mail_handling_delay']))
        break


def run():
    with daemon.DaemonContext(working_directory='.'):
        prepare_mailboxes()
        process_mail()

if __name__ == "__main__":
    run()
