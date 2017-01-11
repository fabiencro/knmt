#!/usr/bin/env perl

use strict;
use utf8;

binmode STDIN, ":encoding(utf8)";
binmode STDOUT, ":encoding(utf8)";

while (<STDIN>) {
    chomp;
    if ($_ !~ /^\# S-ID/) {
        s/　/ /g;
        tr/[ａ-ｚＡ-Ｚ０-９]/[a-zA-Z0-9]/;
        tr/[！”＃＄％＆’（）＝〜｜‘｛＋＊｝＜＞？＿‐＾￥＠「［；：］」，．／￥]/[\!\"\#\$\%\&\'\(\)\=\~\|\`\{\+\*\}\<\>\?\_\-\^\\\@\[\[\;\:\]\]\,\.\/\\]/;
        s/＆/\&/g;
        s/＜/\</g;
        s/＞/\>/g;
    }
    print "$_\n";
}
