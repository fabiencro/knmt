#!/usr/bin/env perl

use strict;
use utf8;
binmode STDIN, ':encoding(utf-8)';
binmode STDOUT, ':encoding(utf-8)';
binmode STDERR, ':encoding(utf-8)';
use KNP;
use CompoundNounExtractor;
use Juman::Grammar qw/ $FORM $TYPE $HINSI/;
use Encode;

autoflush STDIN 1;
autoflush STDOUT 1;
autoflush STDERR 1;

autoflush OUT 1;

# -e 用の置換マップ
my %FILTER = (
 "あ" => "え", "い" => "え", "う" => "え", "え" => "え", "お" => "え",
 "ぁ" => "ぇ", "ぃ" => "ぇ", "ぅ" => "ぇ", "ぇ" => "ぇ", "ぉ" => "ぇ",
 "か" => "け", "き" => "け", "く" => "け", "け" => "け", "こ" => "け",
 "が" => "げ", "ぎ" => "げ", "ぐ" => "げ", "げ" => "げ", "ご" => "げ",
 "さ" => "せ", "し" => "せ", "す" => "せ", "せ" => "せ", "そ" => "せ",
 "た" => "て", "ち" => "て", "つ" => "て", "っ" => "て", "て" => "て", "と" => "て",
 "だ" => "で", "ぢ" => "で", "づ" => "で", "で" => "で", "ど" => "で",
 "な" => "ね", "に" => "ね", "ぬ" => "ね", "ね" => "ね", "の" => "ね",
 "は" => "へ", "ひ" => "へ", "ふ" => "へ", "へ" => "へ", "ほ" => "へ",
 "ば" => "べ", "び" => "べ", "ぶ" => "べ", "べ" => "べ", "ぼ" => "べ",
 "ぱ" => "ぺ", "ぴ" => "ぺ", "ぷ" => "ぺ", "ぺ" => "ぺ", "ぽ" => "ぺ",
 "ま" => "め", "み" => "め", "む" => "め", "め" => "め", "も" => "め",
 "や" => "え", "ゆ" => "え", "よ" => "え", "ゃ" => "ぇ", "ゅ" => "ぇ", "ょ" => "ぇ",
 "ら" => "れ", "り" => "れ", "る" => "れ", "れ" => "れ", "ろ" => "れ",
 "わ" => "え", "を" => "え", "ん" => "え"
);

# utils for Katuyou
sub _zerop {
    ( $_[0] =~ /\D/ )? $_[0] eq '*' : $_[0] == 0;
}

sub _indexp {
    ( $_[0] !~ /\D/ and $_[0] >= 1 );
}

# 活用形のIDを取得
sub get_form_id {
    my( $type, $x ) = @_;

    $type = Encode::encode('utf-8',$type);
    $x = Encode::encode('utf-8',$x);
    
    if( $type eq '*' ){
        if( &_zerop($x) ){
            return 0;
        }
    } elsif( exists $FORM->{$type} ){
        if( exists $FORM->{$type}->[0]->{$x} ){
            return $FORM->{$type}->[0]->{$x};
        } elsif( &_indexp($x) and defined $FORM->{$type}->[$x] ){
            return $x;
        }
    }
    undef;
}

sub get_type_id {
    my( $x ) = @_;

    if (utf8::is_utf8($x)) { # encode if the input has utf8_flag
        $x = Encode::encode('utf-8', $x);
    }

    if( &_zerop($x) ){
        0;
    } elsif( exists $TYPE->[0]->{$x} ){
        $TYPE->[0]->{$x};
    } elsif( &_indexp($x) and defined $TYPE->[$x] ){
        $x;
    } else {
        undef;
    }
}

# 語尾を変化させる内部関数
sub _change_gobi {
    my( $str, $cut, $add ) = @_;

    unless( $cut eq '*' ){
        # エ基本形からほかの活用形へは変更できない．
        if($cut =~ /^-e/){
            return $str;
        }
        $str =~ s/$cut\Z//;
    }

    unless( $add eq '*' ){
        # -e の処理
        if( $add =~ /^-e(.*)$/ ){
            my $add_tail = $1;
            if( $str =~ /^(.*)(.)$/ ){

                my $head = $1;
                my $tail = $2;
                if( exists( $FILTER{$tail} )){
                    $str = $head.$FILTER{$tail};
                }
            }
            $str .= $add_tail;
        }else{
            $str .= $add;
        }
    }
    $str;
}

sub change_katuyou {
    my( $midasi, $form, $from_form, $type ) = @_;
    
    my $from_form_id = &get_form_id( $type, $from_form );
    my $id = &get_form_id( $type, $form );

    my $encoded_type = Encode::encode('utf-8',$type);
    if( defined $id and $id > 0 and defined $from_form_id and $from_form_id > 0){
        # 変更先活用形が存在する場合
        my @oldgobi = @{ $FORM->{$encoded_type}->[$from_form_id] }; 
        my @newgobi = @{ $FORM->{$encoded_type}->[$id] };

        # カ変動詞来の場合の処理
        if( $type eq 'カ変動詞来'){
            if( $midasi eq Encode::decode('utf-8',$oldgobi[1])){
                return &_change_gobi($midasi, Encode::decode('utf-8',$oldgobi[1]), Encode::decode('utf-8',$newgobi[1]) );
            }else{
                return &_change_gobi($midasi, Encode::decode('utf-8',$oldgobi[2]), Encode::decode('utf-8',$newgobi[2]) );
            }
        }else{
            return &_change_gobi( $midasi, Encode::decode('utf-8',$oldgobi[1]), Encode::decode('utf-8',$newgobi[1]) );
        }
    } else {
        # 変更先活用形が存在しない場合
        undef;
    }
}

# 語幹と活用語尾に分割し，配列を返す
sub split_gobi {
    my ( $midasi, $from_form, $type) = @_;
    my @out = ();

    # 語幹の取り出し
    my $gokan = &change_katuyou($midasi, "語幹", $from_form, $type);
    if($gokan ne ""){push(@out,$gokan)}

    # 語尾の取り出し
    $midasi =~ /^$gokan(.*)$/;
    my $gobi = $1;
    if($gobi ne ""){push(@out,$gobi)}

    return @out;
}

my %words;
open(IN, "<:encoding(utf8)", $ARGV[0]) || die;
while (<IN>) {
    chomp;
    my ($word, $count) = split(/\t/);
    $words{$word} = $count if ($count >= 20);
}
close IN;

open(OUT, ">:encoding(utf8)", $ARGV[1].".kw") || die;

my $cne = new CompoundNounExtractor({'get_verbose' => 1, 'get_start_end_mrphnum' => 1});
my $buf;
while (<STDIN>) {
    $buf .= $_;
    if (/^EOS$/) {
        my @mrph_list = ();
        my $result = new KNP::Result($buf);
        my $kw_index = 0;
        foreach my $bnst ($result->bnst) {
            my %kw_index_list;
            my @compound_nouns = $cne->ExtractCompoundNounfromBnst($bnst, { });
            foreach my $word (sort {$a->{start_mrphnum} <=> $b->{start_mrphnum}} @compound_nouns) {
                # print STDERR "midasi:$word->{midasi} repname:$word->{repname} start:$word->{start_mrphnum} end: $word->{end_mrphnum}\n";
                if (defined $kw_index_list{$word->{start_mrphnum}}) {
                    $kw_index_list{$word->{start_mrphnum}} = $word->{end_mrphnum} if ($kw_index_list{$word->{start_mrphnum}} < $word->{end_mrphnum});
                    next;
                }
                my @midasi_list = split(/\+/, $word->{midasi});
                my @bunrui_list = map((split(/\|/, $_))[2], split(/\+/, $word->{verbose}));
                my $flag = 0;
                for (my $i = 0; $i < @midasi_list; $i++) {
                    if ($midasi_list[$i] !~ /^[Ａ-Ｚａ-ｚ]+$/ && $bunrui_list[$i] ne '数詞' && !defined $words{$midasi_list[$i]}) {
                        $flag = 1;
                        last;
                    }
                }
                if ($flag && length($word->{midasi}) > 1) {
                    $kw_index_list{$word->{start_mrphnum}} = $word->{end_mrphnum};
                }
            }
            my @mrph_list_tmp = $bnst->mrph_list;
            for (my $mrph_index = 0; $mrph_index < @mrph_list_tmp; $mrph_index++) {
                if (defined $kw_index_list{$mrph_index}) {
                    push(@mrph_list, "<K-$kw_index>");
                    printf OUT "K-%d: %s\n", $kw_index, join("", map($_->midasi, @mrph_list_tmp[$mrph_index..$kw_index_list{$mrph_index}]));
                    $mrph_index += ($kw_index_list{$mrph_index} - $mrph_index);
                    $kw_index++;
                } else {
                    my @sp = &split_gobi($mrph_list_tmp[$mrph_index]->midasi, $mrph_list_tmp[$mrph_index]->katuyou2, $mrph_list_tmp[$mrph_index]->katuyou1);
                    push(@mrph_list, @sp);
                }
            }
        }
        print "@mrph_list"."\n";
        $buf = '';
    }
}

close OUT;
