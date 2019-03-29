To generate the binary string file:

msgfmt -o res/messages_ja.mo res/messages_ja.po


If new strings are added to the source code, update the .po file and merge it
with the existing file:

xgettext nmt_chainer/translation/mail_handler.py
msgmerge -o res/messages_ja.po res/messages_ja.po messages.po

Edit the new string translations and regenerate the binary string file (see above).

Pay attention to strings that have the comments fuzzy.  I think it must be removed for
the translation string to be used properly.
