#!/usr/bin/env perl

# 長い文を分割する
# 分割条件: 文が$LENGTH_THRESHOLDより長い
#           文の30%-70%の位置(つまり中央)に、強い区切り(<ID:〜が>など)がある
# 分割した場合に、1文目は接続表現より後の付属語を削除し、「。」を付加する

# Usage: perl -I ../perl sentence-splitter.perl < text.txt | perl long-sentence-splitter.perl > text-splitted.txt

use KNP;
use strict;
use utf8;
binmode STDIN, ':utf8';
binmode STDOUT, ':utf8';

our $LENGTH_THRESHOLD = 70;
our $BEGINNING_CHECKED_DELIMITER = 0.3;
our $ENDING_CHECKED_DELIMITER = 0.7;

my $knp = new KNP(-Option => '-tab -dpnd');
while (<STDIN>) {
    chomp;
    my $len = length($_);
    my $beginning_checked_pos = $len * $BEGINNING_CHECKED_DELIMITER;
    my $ending_checked_pos = $len * $ENDING_CHECKED_DELIMITER;
    my $split_flag = 0;
    my (@pre_splitted, @post_splitted);
    if ($len >= $LENGTH_THRESHOLD) {
	# KNP
	my $result = $knp->parse($_);
	my $pos = 0;
	my @mrphs;
	for my $tag ($result->tag) {
	    if ($pos > $ending_checked_pos && $split_flag == 0) {
		last;
	    }
	    # 強い区切りかどうかをチェック
	    my $boundary_type = &check_strong_boundary($tag);
	    if ($pos > $beginning_checked_pos && $boundary_type > 0) {
		$split_flag = $boundary_type;
		# 「が」より後を捨てる
		for my $mrph ($tag->mrph) {
		    if (&check_boundary_mrph($mrph, $boundary_type)) {
			last;
		    }
		    push(@mrphs, $mrph);
		}
		@pre_splitted = @mrphs;
		@mrphs = ();
		next;
	    }
	    for my $mrph ($tag->mrph) {
		$pos += length($mrph->midasi);
		push(@mrphs, $mrph);
	    }
	}
	if ($split_flag) {
	    push(@post_splitted, @mrphs);
	}
    }

    if ($split_flag) {
	&print_splitted_sentence(\@pre_splitted, \@post_splitted, $split_flag);
    }
    else {
	# そのままプリント
	print $_, "\n";
    }
}

sub print_splitted_sentence {
    my ($pre_splitted_ar, $post_splitted_ar, $boundary_type) = @_;

    for my $mrph (@{$pre_splitted_ar}) {
	print $mrph->midasi;
    }
    # 「であり」->「だ」
    if ($boundary_type == 2) {
	print 'だ';
    }
    # 「おり」->「いる」
    elsif ($boundary_type == 3) {
	print 'いる';
    }
    # 句点を挿入
    print "。\n";
    for my $mrph (@{$post_splitted_ar}) {
	print $mrph->midasi;
    }
    print "\n";
}

sub check_strong_boundary {
    my ($tag) = @_;

    if ($tag->fstring =~ /<ID:〜が>/) {
	return 1;
    }
    elsif ($tag->fstring =~ /<ID:〜であり>/) {
	return 2;
    }
    elsif ($tag->fstring =~ /<ID:〜ており>/) {
	return 3;
    }
    else {
	return 0;
    }
}

sub check_boundary_mrph {
    my ($mrph, $boundary_type) = @_;

    # 接続助詞「が」を捨てる
    if ($boundary_type == 1 && $mrph->genkei eq 'が' && $mrph->hinsi eq '助詞') {
	return 1;
    }
    # 「であり」を捨てて、後で「だ」を足す
    elsif ($boundary_type == 2 && $mrph->genkei eq 'だ' && $mrph->hinsi eq '判定詞') {
	return 1;
    }
    # 「おり」を捨てて、後で「いる」を足す
    elsif ($boundary_type == 3 && $mrph->genkei eq 'おる' && $mrph->hinsi eq '接尾辞') {
	return 1;
    }
    else {
	return 0;
    }
}
