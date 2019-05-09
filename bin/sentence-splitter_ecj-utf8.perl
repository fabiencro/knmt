#!/usr/bin/env perl

# Split japanese, english, chinese text to sentences
#
# input format:
# ===========
# # <comment 1>
# zh: <chinese text 1>
# ja: <japanese text 1>
# en: <english text 1>
# # <comment 2>
# ja: <japanese text 2>
# ...
# ===========
# comment lines are ignored, outputted as is in the result file

# output format:
# ===========
# # <comment 1>
# zh: <chinese sentence 1-1>
# zh: <chinese sentence 1-2>
# zh: <chinese sentence 1-3>
# ja: <japanese sentence 1-1>
# ja: <japanese sentence 1-2>
# en: <english sentence 1-1>
# en: <english sentence 1-2>
# en: <english text 1-3>
# # <comment 2>
# ja: <japanese sentence 1-1>
# ...
# ===========


use SentenceExtractor;
use utf8;
use strict;
use FindBin qw/$Bin/;
use lib "$Bin/../perllib/";

while (<STDIN>) {

    # Since we use ";" as sentence boundaries, replace "&amp;" "&gt;" etc. before splitting 
    s/\&amp;/\&/g;
    s/\&gt;/>/g;
    s/\&lt;/</g;
    s/＆ｇｔ；/＞/g;
    s/＆ｌｔ；/＜/g;

    if (/^zh:/) {
        s/^zh: //;
        
        for my $sentence (SentenceExtractor->new($_, 'chinese')->GetSentences()) {
            if (length($sentence) > 0) {
                print "zh: $sentence\n";
            }
        }
    } elsif (/^ja:/) {
        s/^ja: //;
        
        for my $sentence (SentenceExtractor->new($_, 'japanese')->GetSentences()) {
            if (length($sentence) > 0) {
                print "ja: $sentence\n";
            }
        }
    } elsif (/^en:/) {
        s/^en: //;        
        # replace zenkaku spaces (many in science opinion data)
        s/　/ /g; 
        s/”/"/g;
        s/“/"/g;
        s/（/(/g;
        s/）/)/g;
        # preprocess text to replace "." with ". " and ";" with "; "
        
        # "." following lower case letter or ")", and followed by a capital letter or number (and further followed by anything but a "." to avoid splitting on Ph.D.)
        s/(?<=[a-z)])\.(?=[A-Z0-9][^.])/. /g; 
        
        # "." following a non isolated letter and followed by a capital letter (and further followed by anything but a "." to avoid splitting on Ph.D.)
        s/(?<=[^ .][A-Z])\.(?=[A-Z0-9][^.])/. /g;  
        
        # ";" not followed by space -> add space
        s/;(?=[^ ])/; /g; 
        
        for my $sentence (SentenceExtractor->new($_, 'english')->GetSentences()) {
            if (length($sentence) > 0) {
                print "en: $sentence\n";
            }
        }
    } else {
        print $_;
    }
}
