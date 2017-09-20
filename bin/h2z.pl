#!/usr/bin/env perl

use strict;
use utf8;

binmode STDIN, ":encoding(utf8)";
binmode STDOUT, ":encoding(utf8)";

autoflush STDOUT 1;

while (<STDIN>) {
    chomp;
    if ($_ !~ /^\# S-ID/) {
        s/ /　/g;
        tr/a-zA-Z0-9/ａ-ｚＡ-Ｚ０-９/;
    	tr/\!\"\#\$\%\&\'\(\)\=\~\|\`\{\+\*\}\<\>\?\_\-\^\\\@\[\;\:\]\,\.\/\\/！”＃＄％＆’（）＝〜｜‘｛＋＊｝＜＞？＿‐＾￥＠［；：］，．／￥/;
        s/\&amp;/＆/g;
        s/\&lt;/＜/g;
        s/\&gt;/＞/g;
        s/\&apos;/’/g;
        s/\&quot;/”/g;
    }
    print "$_\n";
}
