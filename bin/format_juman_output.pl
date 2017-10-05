#!/usr/bin/env perl

use strict;
use utf8;

binmode STDIN, ":encoding(utf8)";
binmode STDOUT, ":encoding(utf8)";
autoflush STDOUT 1;

while (<STDIN>) {
    chomp;
    my @result;
    foreach my $word (split(/ /)) {
        my ($w, $pos, $cat) = split(/\_/, $word);
        push(@result, $w);
    }
    for (my $i = 0; $i < @result-4; $i++) {
        if ($result[$i] eq "＜" &&
            $result[$i+1] =~ /^[Ｋk]$/ &&
            $result[$i+2] eq "‐" &&
            $result[$i+3] =~ /^([０-９])$/ &&
            $result[$i+4] eq "＞") {
            my $num = $1;
            $num =~ tr/[０-９]/[0-9]/;
            splice(@result, $i, 5, "<K-$num>");
        }
    }
    printf "%s\n", join(" ", @result);
}
