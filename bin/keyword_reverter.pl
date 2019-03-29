#!/usr/bin/env perl

use strict;
use utf8;
binmode STDIN, ':encoding(utf-8)';
binmode STDOUT, ':encoding(utf-8)';
binmode STDERR, ':encoding(utf-8)';

my %dic;
open(DIC, "<:encoding(utf8)", $ARGV[0]) || die;
while (<DIC>) {
    chomp;
    my ($source, $target) = split(/\t/);
    $dic{$source} = $target;
}
close DIC;

my %kw;
open(KW, "<:encoding(utf8)", $ARGV[1].".kw") || die;
while (<KW>) {
    chomp;
    if (/^(K\-\d): (.+)$/) {
        $kw{$1} = $2;
    }
}
close KW;

while (<STDIN>) {
    while (/<(K\-\d)>/) {
        my $kw_index = $1;
        if (defined $kw{$kw_index}) {
            if (defined $dic{$kw{$kw_index}}) {
                s/<K\-\d>/<$kw{$kw_index} = $dic{$kw{$kw_index}}>/;
            } else {
                s/<K\-\d>/<$kw{$kw_index} = ?>/;
            }
        } else {
            s/<K\-\d>//;
        }
    }
    print;
}
