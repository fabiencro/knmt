package CompoundNounExtractor;

# $Id$

# 複合名詞を抽出するPerlモジュール

# 使い方

# 最長のもののみ
# foreach my $bnst ($result->bnst) {
#   my $word = $cne->ExtractCompoundNounfromBnst($bnst, { longest => 1 });
#   print $word->{midasi}, "\n" if $word;
# }

# 複合名詞すべて
# foreach my $bnst ($result->bnst) {
#   my @words = $cne->ExtractCompoundNounfromBnst($bnst);

#   foreach my $tmp (@words) {
#     print $tmp->{midasi}, "\n";
#   }
# }

use strict;
use utf8;
use vars qw($NG_CHAR);

$NG_CHAR = '・|っ|ぁ|ぃ|ぅ|ぇ|ぉ|ゃ|ゅ|ょ|ー'; # 拗音、長音など

sub new {
    my ($this, $option) = @_;

    $this = {};

    $this->{option} = $option;

    $this->{MRPH_NUM_MAX} = 25; # 複合名詞中の形態素数の最大上限数
    if (defined $option->{MRPH_NUM_MAX}) {
        $this->{MRPH_NUM_MAX} = $option->{MRPH_NUM_MAX};
    }

    $this->{LENGTH_MAX} = 30;   # 複合名詞の文字数の最大上限数;
    if (defined $option->{LENGTH_MAX}) {
        $this->{LENGTH_MAX} = $option->{LENGTH_MAX};
    }

    $this->{LENGTH_MAX_ONE_WORD_EACH} = 10; # 複合名詞の文字数の最大上限数(すべての形態素が1文字の場合);
    if (defined $option->{LENGTH_MAX_ONE_WORD_EACH}) {
        $this->{LENGTH_MAX_ONE_WORD_EACH} = $option->{LENGTH_MAX_ONE_WORD_EACH};
    }

    $this->{CENTERED_DOT_NUM_MAX} = 3; # 中黒の最大上限数
    if (defined $option->{CENTERED_DOT_NUM_MAX}) {
        $this->{CENTERED_DOT_NUM_MAX} = $option->{CENTERED_DOT_NUM_MAX};
    }

    bless $this;

    return $this;
}

sub DESTROY {
    my ($this) = @_;

}

# 文節から複合名詞を抽出する
sub ExtractCompoundNounfromBnst {
    my ($this, $bnst, $option) = @_;

    my @ret_word_list;
    
    my $input_is_array_flag = 0;

    my @mrph_list;

    my %m2b;
    # bnstの配列が入力された場合
    if (ref($bnst) eq 'ARRAY') {
        for my $b ( @{$bnst} ) {
            my $bid = $b->id;
            for my $m ($b->mrph) {
                my $mid = $m->id;
                $m2b{$mid} = $bid;
            }
        }

        for my $b ( @{$bnst} ) {
            # 形態素列が入力された場合(クラスタリング用)
            if (ref($b) eq 'KNP::Morpheme') {
                push(@mrph_list, $b);
            } else {
                push(@mrph_list, $b->mrph);
            }
        }
        $input_is_array_flag = 1;
    }
    # bnstが一個入力された場合
    else {
        @mrph_list = $bnst->mrph;
    }

    # 複合名詞の先頭/真ん中/末尾に来ることができるかをチェック
    my (@is_ok_for_head, @is_ok_for_mid, @is_ok_for_tail); 
    for my $i (0..$#mrph_list) {
        my $mrph = $mrph_list[$i];
        my $midasi  = $mrph->midasi;
        my $fstring = $mrph->fstring;
        my $bunrui  = $mrph->bunrui;
        my $hinsi   = $mrph->hinsi;

        # 自分の前後の形態素の見出し(全角スペースの前後が同じ文字種かどうかチェックするため)
        my $midasi_pre = $i != 0 ? $mrph_list[$i - 1]->midasi : '';
        my $midasi_post = $i != $#mrph_list ? $mrph_list[$i + 1]->midasi : '';

        # 真ん中
        $is_ok_for_mid[$i] = $this->CheckConditionMid($midasi, $fstring, $bunrui, $hinsi, $midasi_pre, $midasi_post, $input_is_array_flag);

        # 先頭
        $is_ok_for_head[$i] = $is_ok_for_mid[$i] == 0 ? 0 : $this->CheckConditionHead($midasi, $fstring, $bunrui, $hinsi);

        # 末尾
        $is_ok_for_tail[$i] = $is_ok_for_mid[$i] == 0 ? 0 : $this->CheckConditionTail($midasi, $fstring, $bunrui, $hinsi, \@mrph_list, $i);
    }

    # debug print
    if ($this->{option}{debug}) {
        &print_conditions(\@mrph_list, \@is_ok_for_head, \@is_ok_for_mid, \@is_ok_for_tail);
    }


    # $option->{'subject'} が指定された場合は文中の主題をチェックする
    my @flag = ();
    if ($option->{'subject'}) {

        for (my $i = scalar @mrph_list - 1; $i >= 0; $i--) {

            #「は」の直前の形態素が複合名詞の末尾になり得れば，その形態素にフラグを立てる
            if ($mrph_list[$i]{'hinsi'} eq '助詞' && $mrph_list[$i]{'genkei'} eq 'は') {

                if ($i - 1 >= 0 && $is_ok_for_tail[$i-1]) {
                    $flag[$i-1] = 1;
                }
            }
        }
    }


    # ループを回して複合名詞を探す。
    #
    # ex) 自然 言語 処理 と は、
    #          $j  $i

    my $longest_tail_flag = 0;
    my $outputted_flag = 0;
    my $mrph_num_max_over_flag = 0; # array_input用
    my @mrph_used_num; # 各形態素について、複合名詞の要素となった回数を記録
    for my $i (reverse(0..$#mrph_list)) {

        my @word_list;

        my $mrphnum = 0;
        my $midasi = '';
        my $repname = '';
        my $verbose = '';
        my $jiritsu_mrph_num = 0; #接頭辞、接尾辞を除いた形態素数
        #★「自立」という表現はよろしくない気がする

        my $midasi_i =  $mrph_list[$i]->{midasi};
        my $fstring_i = $mrph_list[$i]->{fstring};

        # (array_input用)
        # 形態素数などのを上限超えている文節からは複合名詞を抽出しない
        # $is_ok_for_mid[$i]が0になったらそこから再開
        if ($mrph_num_max_over_flag) {
            if ($is_ok_for_mid[$i]) {
                next;
            } else {
                $mrph_num_max_over_flag = 0;
                next;
            }
        }

        if (!$is_ok_for_tail[$i]) {
            $longest_tail_flag = 0;
            $outputted_flag = 0;
            next;
        }

        $longest_tail_flag++;

        #　一番最後を固定してループ
        for my $j (reverse(0..$i)) {

            my $mrph_j = $mrph_list[$j];
            my $midasi_j = $mrph_j->midasi;
            my $hinsi_j = $mrph_j->hinsi;
            my $bunrui_j = $mrph_j->bunrui;
            my $fstring_j = $mrph_j->fstring;

            my $tmp = $mrph_j->repnames();

            # 読みを削除
            if ($this->{option}{no_yomi_in_repname}) {
                my @tmp;
                foreach my $mrph (split('\?', $tmp)) {
                    my $hyouki = (split('/', $mrph))[0];
                    push @tmp, $hyouki;
                }

                # 読みだけが異なるものは一つにするためにuniq
                # 例: 今日/きょう?今日/こんにち
                @tmp = delete_overlap(\@tmp);

                $tmp = join('?', @tmp);
            }


            my $repname_j = $tmp ? $tmp : $midasi_j;
            # 見出し|品詞|分類|fstring
            my $verbose_j = join('|', ($midasi_j, $hinsi_j, $bunrui_j, $fstring_j));

            if (!$is_ok_for_mid[$j]) {
                last;
            }

            $mrphnum++;
            $jiritsu_mrph_num++ if ($hinsi_j ne '接頭辞' && $hinsi_j ne '接尾辞');

            $midasi = $midasi_j . (($midasi) ?  '+' .  $midasi : '');
            $repname = $repname_j . (($repname) ?  '+' .  $repname : '');
            $verbose = $verbose_j . (($verbose) ?  '+' .  $verbose : '');

            # 形態素数の上限、文字数の上限を超えた場合
            # 一語ばかりからなる複合語は大抵ごみ (文字化けなど)
            if ($mrphnum >= $this->{MRPH_NUM_MAX} || length ($midasi) >= $this->{LENGTH_MAX} || ($mrphnum == $this->{LENGTH_MAX_ONE_WORD_EACH} && length ($midasi) == $mrphnum)) {
                if ($input_is_array_flag) {
                    $longest_tail_flag = 0;
                    $outputted_flag = 0;
                    @word_list = ();
                    $mrph_num_max_over_flag = 1;
                    last;
                } else {
                    return wantarray ? () : '';
                }
            }

            # 中黒の数の上限
            my $centered_dot_num = ($midasi =~ s/・/・/g);
            if ($this->{CENTERED_DOT_NUM_MAX} != -1 && $centered_dot_num >= $this->{CENTERED_DOT_NUM_MAX}) {
                if ($input_is_array_flag) {
                    $longest_tail_flag = 0;
                    $outputted_flag = 0;
                    @word_list = ();
                    $mrph_num_max_over_flag = 1;
                    last;
                } else {
                    return wantarray ? () : '';
                }
            }

            next if (!$is_ok_for_head[$j]);

            $mrph_used_num[$j]++;

            if (! $option->{'subject'}) {
                push @word_list, { midasi => $midasi, repname => $repname, mrphnum => $mrphnum , jiritsu_mrph_num => $jiritsu_mrph_num, mrphid => $i, bnstid => $m2b{$i}};
            }
            # $option->{'subject'} が指定された場合は文中の主題をチェックする
            else {
                if ($flag[$i]) {
                    push @word_list, { midasi => $midasi, repname => $repname, mrphnum => $mrphnum , jiritsu_mrph_num => $jiritsu_mrph_num, 'subject' => 1, mrphid => $i, bnstid => $m2b{$i}}; # subject = 1 の複合名詞は文中で主題として出現
                } else {
                    push @word_list, { midasi => $midasi, repname => $repname, mrphnum => $mrphnum , jiritsu_mrph_num => $jiritsu_mrph_num, 'subject' => 0, mrphid => $i, bnstid => $m2b{$i}};
                }
            }


            $word_list[-1]{verbose} = $verbose if $this->{option}{get_verbose};
            $word_list[-1]{start_mrphnum} = $j if $this->{option}{get_start_end_mrphnum};
            $word_list[-1]{end_mrphnum} = $i if $this->{option}{get_start_end_mrphnum};

            # 末尾の形態素が未定義語ならば、その複合名詞にundef_flagを追加
            # ただし、「品曖-その他」または「品曖-カタカナ」のみを未定義語とみなし、
            # 「品曖-アルファベット」は、みなさない
            if ($fstring_i =~ /品詞変更:[^\d]*\-15\-[1-2]\-/) {
                if ($word_list[-1]->{mrphnum} == 1) {
                    $word_list[-1]->{undef_flag} = 1;
                }
            }

            print "register $midasi\n" if ($this->{option}{debug});
            $outputted_flag = 1;
        }

        # 入力が複数文節の場合、最長のものにflagを立てる
        # 例：
        # 入力＝「世界各国の民主主義」ならば、「世界各国の民主主義」にflagを、
        # 入力＝「世界平和と戦争」ならば、「世界平和」と「戦争」にflagを立てる
        if ($input_is_array_flag) {
            if ($longest_tail_flag == 1 && $outputted_flag) {
                for my $i (0 .. $#mrph_used_num) {
                    if (defined $mrph_used_num[$i]) {
                        if ($mrph_used_num[$i] == 1) {
                            $word_list[-1]->{longest_flag} = 1;
                        } else {
                            last;
                        }
                    }
                }
            }
        } else {
            # 最長
            if ($option->{longest} && $outputted_flag) {
                return $word_list[-1];
            }
        }

        push @ret_word_list, @word_list if @word_list;
    }

    if ($input_is_array_flag && $option->{longest}) {
        # longest_flagのついているものだけを返す
        return grep {defined $_->{longest_flag}} @ret_word_list;
    } else {
        return wantarray ? @ret_word_list : $ret_word_list[-1];
    }
}

# 先頭に来れるかどうかをチェック
sub CheckConditionHead {
    my ($this, $midasi, $fstring, $bunrui, $hinsi) = @_;

    if (($fstring =~ /<(?:名詞相当語|漢字)>/ || $hinsi eq '接頭辞')
        && $hinsi ne '接尾辞'   # 「-性海棉状脳症」などを除く
        #		&& !$self->is_stopword ($mrph2, 'prefix')
        #		&& $mrph2->fstring !~ /末尾/ && # 人名末尾, 組織名末尾などで終るものを除く
        && $midasi !~ /^(?:$NG_CHAR)/) {


        if ($this->{option}{clustering}) {
            # 「本」研究、「当」病院、「同」病院などを除く
            # ★二つの条件を設けているのは過渡的（そのうち前の条件を削除）
            if (($fstring =~ /<独立タグ接頭辞>/ || ($fstring =~ /<内容語>/ && $bunrui eq '名詞接頭辞')) && $midasi =~ /^(?:本|同)$/) {
                return 0;
            } elsif ($hinsi eq '連体詞' && $midasi eq '当') {
                return 0;
            }
        }
        return 1;
    } else {
        return 0;
    }
}

# 真ん中に来れるかどうかをチェック
sub CheckConditionMid {
    my ($this, $midasi, $fstring, $bunrui, $hinsi, $midasi_pre, $midasi_post, $input_is_array_flag) = @_;

    # 入力が文節の配列の場合は、助詞の「の」でもOK
    if ($input_is_array_flag && $midasi eq 'の' && $hinsi eq '助詞') {
        return 1;
    } elsif (($fstring !~ /<(?:名詞相当語|漢字|複合←)>/ && $hinsi ne '接頭辞')
             || $fstring =~ /<記号>/
             || $midasi =~ /・・/
             || $bunrui =~ /(?:副詞的|形式)名詞/) {

        if ($midasi eq '・') {
            return 1;
        }
        if ($this->{option}{clustering}) {
            if ($fstring =~ /<記号>/ && $midasi eq '　' && 
                ($this->{option}{no_check_same_char_type} || (!$this->{option}{no_check_same_char_type} && &check_same_char_type($midasi_pre, $midasi_post)))) {
                return 1;
            }
        }
        if ($this->{option}{connect_hyphen}) {
            print "$midasi $fstring\n";
            if ($fstring =~ /<記号>/ && $midasi eq '−') {
                return 1;
            }
        }

        return 0;
    } else {
        return 1;
    }
}

# 最後に来れるかどうかをチェック
sub CheckConditionTail {
    my ($this, $midasi, $fstring, $bunrui, $hinsi, $mrph_list, $i) = @_;

    if (($midasi =~ /^\p{Hiragana}$/ && $hinsi !~ /^(?:接頭辞|接尾辞)$/) # ひらがな一文字(接尾辞と接頭辞を除く)
        # || $mrph->fstring =~ /<NE:[A-Z]*:(head|middle)>/ # 固有名詞の途中で終るものは登録しない
        || $fstring !~ /<(?:名詞相当語|かな漢字|カタカナ)>/ # 一番最後が名詞でない
        || $bunrui =~ /(?:副詞的|形式)名詞/ # 形式名詞（もの, こと..)/副詞的名詞(よう, とき..)
        || $fstring =~ /<記号>/ # 名詞相当語 かつ 記号 .. ●,《, ＠など
        # 名詞相当語 かつ 時間辞 .. 1960年代半ばの「代」など
        || $hinsi eq '接頭辞' # 「イースター島再来訪」から「イースター島再」を排除
        || $midasi =~ /・/
        # ★以下の二行を設けてるのは過渡的（そのうち上の行を削除し、下の行の<非独立タグ接尾辞>の条件を削除）
        || ($fstring =~ /<非独立タグ接尾辞>/ && $fstring !~ /<意味有>/) # 非独立接尾辞はNG ただし<意味有>がついている(個、つ、県、化、性など)ならばOK
        || ($hinsi eq '接尾辞' && $fstring !~ /<非独立タグ接尾辞>/ && $fstring !~ /<準?内容語>/) # 接尾辞はNG ただし<準?内容語>がついている(個、つ、県、化、性など)ならばOK
        || ($this->{option}{clustering} && $midasi eq '等')
        # 「飲み方」で「飲み」で終わるのを禁止
        || (defined $mrph_list->[$i + 1] && $hinsi eq '名詞' && $fstring =~ /<代表表記:.+?v>/ && $mrph_list->[$i + 1]->midasi eq '方')) {
        return 0;
    } else {
        return 1;
    }
}

# 文字種が同一かどうかをチェックする
sub check_same_char_type {
    my ($midasi_pre, $midasi_post) = @_;

    return 0 if !$midasi_pre || !$midasi_post;

    # 前後とも一文字の場合
    # 例) 「紹　介」
    return 0 if length($midasi_pre) == 1 && length($midasi_post) == 1;

    foreach my $type ('Hiragana', 'InKatakana', 'Han', 'InHalfwidthAndFullwidthForms') {
        if ($midasi_pre =~ /^\p{$type}+$/ && $midasi_post =~ /^\p{$type}+$/) {
            return 1;
            last;
        }
    }

    return 0;
}

sub print_conditions {
    my ($mrph_list, $is_ok_for_head, $is_ok_for_mid, $is_ok_for_tail) = @_;

    my @mrph_list = @{$mrph_list};

    for my $i (0..$#mrph_list) {
        my $mrph = $mrph_list[$i];
        print $mrph->midasi . "\n";
        print "  " . $mrph->hinsi . "\n";
        print "  " . $mrph->bunrui . "\n";
        print "  head $is_ok_for_head->[$i]  tail $is_ok_for_tail->[$i]  mid $is_ok_for_mid->[$i]\n";
        print "  " . $mrph->fstring . "\n";
        print "\n";
    }
}

# 配列の重複を削除
sub delete_overlap {
    my ($ar_list) = @_;

    my %seen = ();
    my @uniq = grep { ! $seen{$_}++ } @{$ar_list};

    return @uniq;
}

1;
