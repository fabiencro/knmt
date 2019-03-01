#!/bin/sh
#
# Usage: sentence_split.sh < input > output
# 
# 入出力形式は sentence-splitter_ecj-utf8.perl 内のコメント参照

# Test where we are before launching the executable.
# The current working directory differs in function of 
# the HTTP server.  
# For Apache: cwd = web_app/cgi-bin
# For Python's CGIHTTPServer: cwd = web_app
if [ -f sentence-splitter_ecj-utf8.perl ]; then
    SCRIPT_PATH=.
else
    SCRIPT_PATH=..
fi

perl -I $SCRIPT_PATH/perllib $SCRIPT_PATH/sentence-splitter_ecj-utf8.perl

