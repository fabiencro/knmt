package SentenceExtractor;

# 文章 -> 文 フィルタ
# from TextExtor.pm (Japanese, Chinese), sentence-boundary.pl (English)

# $Id$

use vars qw($comma $open_kakko $close_kakko $period $dot $alphabet_or_number $itemize_header @honorifics);
# use strict;
use utf8;
use Encode;
use Data::Dumper;
{
    package Data::Dumper;
    sub qquote { return shift; }
}
$Data::Dumper::Useperl = 1;

$comma = qr/，/;
$open_kakko  = qr/（|〔|［|｛|＜|≪|「|『|【|\(|\[|\{/;
$close_kakko = qr/）|〕|］|｝|＞|≫|」|』|】|\)|\]|\}/;

$period = qr/。|？|！|♪|…/;
$dot = qr/．/;
$alphabet_or_number = qr/(?:[Ａ-Ｚ]|[ａ-ｚ]|[０-９])/;
$itemize_header = qr/${alphabet_or_number}．/;

@honorifics = qw(Adj. Adm. Adv. Asst. Bart. Brig. Bros. Capt. Cmdr. Col. Comdr. Con. Cpl. Dr. Ens. Gen. Gov. Hon. Hosp. Insp. Lt. M. MM. Maj. Messrs. Mlle. Mme. Mr. Mrs. Ms. Msgr. Op. Ord. Pfc. Ph. Prof. Pvt. Rep. Reps. Res. Rev. Rt. Sen. Sens. Sfc. Sgt. Sr. St. Supt. Surg. vs. v.);

sub new
{
   my($this, $paragraph, $language) = @_;

   $this = {paragraph => $paragraph, 
	    sentences => []};

   bless($this);

   if ($language eq 'english') {
       @{$this->{sentences}} = &SplitEnglish($paragraph);
   }
   else { # for Japanese, Chinese
       @{$this->{sentences}} = &SplitJapanese($paragraph);
   }

   return $this;
}

sub GetSentences
{
    my ($this) = @_;

    return @{$this->{sentences}};
}


sub FixParenthesis {
    my ($slist) = @_;
    for my $i (0 .. scalar(@{$slist} - 1)) {
	# 1つ目の文以降で、閉じ括弧が文頭にある場合は、閉じ括弧をとって前の文にくっつける
        if ($i > 0 && $slist->[$i] =~ /^$close_kakko+/o) {
	    $slist->[$i - 1] .= $&;
	    $slist->[$i] = "$'";
	}

	# 1つ前の文と当該文に”が奇数個含まれている場合は、前の文に該当文をくっつける
	if ($i > 0) {
	    my $num_of_zenaku_quote_prev = scalar(split('”', $slist->[$i - 1], -1)) - 1;
	    my $num_of_zenaku_quote_curr = scalar(split('”', $slist->[$i], -1)) - 1;

	    if ($num_of_zenaku_quote_prev > 0 && $num_of_zenaku_quote_curr > 0) {
		if ($num_of_zenaku_quote_prev % 2 == 1 && $num_of_zenaku_quote_curr % 2 == 1) {
		    $slist->[$i - 1] .= $slist->[$i];
		    splice(@$slist, $i, 1);
		}
	    }
	}

	# 当該文が^$itemize_header$にマッチする場合、箇条書きと判断し、次の文とくっつける
	if (defined $slist->[$i + 1]) {
	    if ($slist->[$i] =~ /^$itemize_header$/o) { # added o by ynaga
		$slist->[$i] .= $slist->[$i + 1];
		splice(@$slist, $i + 1, 1);
	    }
 	}
    }
}

### テキストを句点で文単位に分割する
### カッコ内の句点では分割しない
sub SplitJapanese {
    my ($str) = @_;
    my (@chars, @buf, $ignore_level);

    my $level = 0;
    ## ynaga; 処理を改善したバージョン;
    ## 1) ・・・を区切り文字として扱う
    ## 2) 文頭を除いて区切り文字のみの文を禁止
    ## 3) 区切り文字を含まない場合にも文頭（）の処理を行う
    ## 注1) 区切り文字を含む顔文字がある場合、その処理を先にする方が良さそう
    ## 注2) ・・・を区切り文字として扱った関係でその直後の顔文字が次文に回される
    @buf = ();
    my @tmp = ();
    my $sent = '';
    my $cdot = '・';
    # while ($str =~ /(?:(?:$period)|((?:$cdot){3,})|(?<!$alphabet_or_number)(?:$dot)(?!$alphabet_or_number|$comma))(?:$dot|$period)*/o) { # check delimiter & split
    while ($str =~ /(?:(?:$period)|((?:$cdot){3,})|(?<!$alphabet_or_number)(?:$dot)|(?:$dot)(?!$alphabet_or_number|$comma))(?:$dot|$period)*/o) { # check delimiter & split
	my $pre = $`;
	$str = $'; ## this line stops incorrect coloring '
	$sent .= $` . $&;

	if ($1 eq '' || $pre ne '') { # a sentence should include strings other than $cdot{3,}
	    unless ($ignore_level) {
		while ($pre =~ /($open_kakko)|(?:$close_kakko)/o) { # mimicking the original routine to count kakko level
		    $level += ($1 ne '') ? 1 : -1;
		    $level = 0 if ($level < 0);
		    $pre = $'; #'
		}
	    }
	    # FEATURE: process only those strings recognized by $dot/$period
	    if ($level == 0) {
		push (@tmp, $sent);
		$sent = '';
	    }
	}
    }
    push (@tmp, $sent . $str);
    foreach my $s (@tmp) {
	# 先頭がカッコで囲まれた文から始まっている場合
	if ($s =~ /^(（.+?）)(.+)$/o) {
	    # カッコで囲まれた文
	    my $s_enclosed_by_kakko = $1;

	    # カッコで囲まれた文以降
	    $s = $2;

	    my $s_tmp = '';
	    # カッコで囲まれた文がデリミタを含んでいるならば区切る
	    while ($s_enclosed_by_kakko =~ /(?:(?:$period)|(?:$cdot){3,}|(?<!$alphabet_or_number)(?:$dot)(?!$alphabet_or_number|$comma))(?:$dot|$period)*/o) {
		my $pre = $`;
		$s_enclosed_by_kakko = $'; # '

		$s_tmp .= $` . $&;
		if ($& !~ /^$cdot/ || $pre ne '（') {
		    push(@buf, $s_tmp);
		    $s_tmp = '';
		}
	    }
	    push(@buf, $s_enclosed_by_kakko) if ($s_enclosed_by_kakko ne '');
	}
	push(@buf, $s) if ($s ne '');
    }

    my @buf2 = ();
    foreach my $y (@buf) {
        $y =~ s/^(?:\s|　)+//o;
        $y = reverse ($y); $y =~ s/^(?:\s|　)+//o; $y = reverse ($y);
	push(@buf2, $y);
    }

    if ($this->{opt}{debug}) {
	print Dumper(\@buf2) . "\n";
	print "-----\n";
    }

    &FixParenthesis(\@buf2);
    if ($this->{opt}{debug}) {
	print Dumper(\@buf2) . "\n";
	print "-----\n";
    }

    @buf = &concatSentences(\@buf2);
    if ($this->{opt}{debug}) {
	print Dumper(\@buf2) . "\n";
	print "-----\n";
    }

    pop(@buf) unless $buf[-1];
    return @buf;
}

sub concatSentences {
    my ($sents) = @_;
    my @buff = ();
    my $tail = scalar(@{$sents}) - 1;
    while ($tail > 0) {
        if ($sents->[$tail] =~ /^(?:と|っ|です)/o && $sents->[$tail - 1] =~ /(?:！|？|$close_kakko)$/o) {
	    $sents->[$tail - 1] .= $sents->[$tail];
	}
        elsif ($sents->[$tail] =~ /^(?:と|や|の)($itemize_header)?/o && $sents->[$tail - 1] =~ /$itemize_header$/o) {
	    $sents->[$tail - 1] .= $sents->[$tail];
	}
	else {
	    unshift(@buff, $sents->[$tail]);
	}
	$tail--;
    }
    unshift(@buff, $sents->[0]);
    return @buff;
}

sub SplitEnglish
{
   my ($paragraph) = @_;
   my (@sentences, @words, $sentence);

   # Split the paragraph into words
   @words = split(" ", $paragraph);

   $sentence = "";

   for $i (0..$#words)
   {
      $newword = $words[$i];

      # Print the words
      #print "word is: ($newword)\n";

      # Check the existence of a candidate
      $period_pos = rindex($newword, ".");
      $question_pos = rindex($newword, "?");
      $exclam_pos = rindex($newword, "!");

      # Determine the position of the rightmost candidate in the word
      $pos = $period_pos;
      $candidate = ".";
      if ($question_pos > $period_pos)
      {
         $pos = $question_pos;
         $candidate = "?";
      }
      if ($exclam_pos > $pos)
      {
         $pos = $exclam_pos;
         $candidate = "!";
      }

      # Do the following only if the word has a candidate
      if ($pos != -1)
      {
         # Check the previous word
         if (!defined($words[$i - 1]))
         {
            $wm1 = "NP";
            $wm1C = "NP";
            $wm2 = "NP";
            $wm2C = "NP";
         }
         else
         {
            $wm1 = $words[$i - 1];
            $wm1C = Capital($wm1);

            # Check the word before the previous one 
            if (!defined($words[$i - 2]))
            {
               $wm2 = "NP";
               $wm2C = "NP";
            }
            else
            {
               $wm2 = $words[$i - 2];
               $wm2C = Capital($wm2);
            }
         }
         # Check the next word
         if (!defined($words[$i + 1]))
         {
            $wp1 = "NP";
            $wp1C = "NP";
            $wp2 = "NP";
            $wp2C = "NP";
         }
         else
         {
            $wp1 = $words[$i + 1];
            $wp1C = Capital($wp1);

            # Check the word after the next one 
            if (!defined($words[$i + 2]))
            {
               $wp2 = "NP";
               $wp2C = "NP";
            }
            else
            {
               $wp2 = $words[$i + 2];
               $wp2C = Capital($wp2);
            }
         }

         # Define the prefix
         if ($pos == 0)
         {
            $prefix = "sp";
         }
         else
         {
            $prefix = substr($newword, 0, $pos);
         }
         $prC = Capital($prefix);

         # Define the suffix
         if ($pos == length($newword) - 1)
         {
            $suffix = "sp";
         }
         else
         {
            $suffix = substr($newword, $pos + 1, length($newword) - $pos);
         }
         $suC = Capital($suffix);
 
         # Call the Sentence Boundary subroutine
         $prediction = Boundary($candidate, $wm2, $wm1, $prefix, $suffix, $wp1, 
            $wp2, $wm2C, $wm1C, $prC, $suC, $wp1C, $wp2C);

         # Append the word to the sentence
         $sentence = join ' ', $sentence, $words[$i];
         if ($prediction eq "Y")
         {
            # Eliminate any leading whitespace
            $sentence = substr($sentence, 1);

	    push(@sentences, $sentence);
            $sentence = "";
         }
      }
      else
      { 
         # If the word doesn't have a candidate, then append the word to the sentence
         $sentence = join ' ', $sentence, $words[$i];
      }
   }
   if ($sentence ne "")
   {
      # Eliminate any leading whitespace
      $sentence = substr($sentence, 1);

      push(@sentences, $sentence);
      $sentence = "";
   }
   return @sentences;
}

sub get_sentences
{
    my ($this) = @_;

    return @{$this->{sentences}};
}

# This subroutine returns "Y" if the argument starts with a capital letter.
sub Capital
{
   my ($substring);

   $substring = substr($_[0], 0, 1);
   if ($substring =~ /[A-Z]/)
   {
      return "Y";
   }
   else
   {
      return "N";
   }
}

# This subroutine does all the boundary determination stuff
# It returns "Y" if it determines the candidate to be a sentence boundary,
# "N" otherwise
sub Boundary
{
   # Declare local variables
   my($candidate, $wm2, $wm1, $prefix, $suffix, $wp1, $wp2, $wm2C, $wm1C, 
         $prC, $suC, $wp1C, $wp2C) = @_;

   # Check if the candidate was a question mark or an exclamation mark
   if ($candidate eq "?" || $candidate eq "!")
   {
      # Check for the end of the file
      if ($wp1 eq "NP" && $wp2 eq "NP")
      {
         return "Y";
      }
      # Check for the case of a question mark followed by a capitalized word
      if ($suffix eq "sp" && $wp1C eq "Y")               
      {
         return "Y";
      }
      if ($suffix eq "sp" && StartsWithQuote($wp1))
      {
         return "Y";
      }
      if ($suffix eq "sp" && $wp1 eq "--" && $wp2C eq "Y") 
      {
         return "Y";
      }
      if ($suffix eq "sp" && $wp1 eq "-RBR-" && $wp2C eq "Y")
      {
         return "Y";
      }
      # This rule takes into account vertical ellipses, as shown in the
      # training corpus. We are assuming that horizontal ellipses are
      # represented by a continuous series of periods. If this is not a
      # vertical ellipsis, then it's a mistake in how the sentences were
      # separated.
      if ($suffix eq "sp" && $wp1 eq ".")
      {
         return "Y";
      }
      if (IsRightEnd($suffix) && IsLeftStart($wp1))
      {
         return "Y";
      }
      else 
      {
         return "N";
      }
   }
   else
   {
      # Check for the end of the file
      if ($wp1 eq "NP" && $wp2 eq "NP")
      {
         return "Y";
      }
      if ($suffix eq "sp" && StartsWithQuote($wp1))
      {
         return "Y";
      }
      if ($suffix eq "sp" && StartsWithLeftParen($wp1))
      {
         return "Y";
      }
      if ($suffix eq "sp" && $wp1 eq "-RBR-" && $wp2 eq "--")
      {
         return "N";
      }
      if ($suffix eq "sp" && IsRightParen($wp1))
      {
         return "Y";
      }
      # This rule takes into account vertical ellipses, as shown in the
      # training corpus. We are assuming that horizontal ellipses are
      # represented by a continuous series of periods. If this is not a
      # vertical ellipsis, then it's a mistake in how the sentences were
      # separated.
      if ($prefix eq "sp" && $suffix eq "sp" && $wp1 eq ".")
      {
         return "N";
      }
      if ($suffix eq "sp" && $wp1 eq ".")
      {
         return "Y";
      }
      if ($suffix eq "sp" && $wp1 eq "--" && $wp2C eq "Y" 
            && EndsInQuote($prefix))
      {
         return "N";
      }
      if ($suffix eq "sp" && $wp1 eq "--" && ($wp2C eq "Y" || 
               StartsWithQuote($wp2)))
      {
         return "Y";
      }
      if ($suffix eq "sp" && $wp1C eq "Y" && 
           ($prefix eq "p.m" || $prefix eq "a.m") && IsTimeZone($wp1))
      {
         return "N";
      }
      # Check for the case when a capitalized word follows a period,
      # and the prefix is a honorific
      if ($suffix eq "sp" && $wp1C eq "Y" && IsHonorific($prefix."."))
      {
         return "N";
      }
      # Check for the case when a capitalized word follows a period,
      # and the prefix is a honorific
      if ($suffix eq "sp" && $wp1C eq "Y" && StartsWithQuote($prefix))
      {
         return "N";
      }
      # This rule checks for prefixes that are terminal abbreviations
      if ($suffix eq "sp" && $wp1C eq "Y" && IsTerminal($prefix))
      {
         return "Y";
      }
      # Check for the case when a capitalized word follows a period and the
      # prefix is a single capital letter
      if ($suffix eq "sp" && $wp1C eq "Y" && $prefix =~ /^([A-Z]\.)*[A-Z]$/)
      {
         return "N";
      }
      # Check for the case when a capitalized word follows a period
      if ($suffix eq "sp" && $wp1C eq "Y")               
      {
         return "Y";
      }
      if (IsRightEnd($suffix) && IsLeftStart($wp1))
      {
         return "Y";
      }
   }
   return "N";
}


# This subroutine checks to see if the input string is equal to an element
# of the @honorifics array.
sub IsHonorific
{
   my($word) = @_;
   my($newword);

   foreach $newword (@honorifics)
   {
      if ($newword eq $word)
      {
         return 1;      # 1 means true
      }
   }
   return 0;            # 0 means false
}

# This subroutine checks to see if the string is a terminal abbreviation.
sub IsTerminal
{
   my($word) = @_;
   my($newword);
   my(@terminals) = ("Esq", "Jr", "Sr", "M.D");

   foreach $newword (@terminals)
   {
      if ($newword eq $word)
      {
         return 1;      # 1 means true
      }
   }
   return 0;            # 0 means false
}

# This subroutine checks if the string is a standard representation of a U.S.
# timezone
sub IsTimeZone
{
   my($word) = @_;
   
   $word = substr($word,0,3);
   if ($word eq "EDT" || $word eq "CST" || $word eq "EST")
   {
      return 1;
   }
   else
   {
      return 0;
   }
}

# This subroutine checks to see if the input word ends in a closing double
# quote.
sub EndsInQuote 
{
   my($word) = @_;

   if (substr($word,-2,2) eq "''" || substr($word,-1,1) eq "'" || 
         substr($word, -3, 3) eq "'''" || substr($word,-1,1) eq "\""
         || substr($word, -2,2) eq "'\"")
   {
      return 1;         # 1 means true
   }
   else
   {
      return 0;         # 0 means false
   }
}

# This subroutine checks to see if a given word starts with one or more quotes
sub StartsWithQuote 
{
   my($word) = @_;

   if (substr($word,0,1) eq "'" ||  substr($word,0,1) eq "\"" || 
         substr($word, 0, 1) eq "`")
   {
      return 1;         # 1 means true
   }
   else
   {
      return 0;         # 0 means false
   }
}

# This subroutine checks to see if a word starts with a left parenthesis, be it
# {, ( or <
sub StartsWithLeftParen 
{
   my($word) = @_;

   if (substr($word,0,1) eq "{" || substr($word,0,1) eq "(" 
         || substr($word,0,5) eq "-LBR-")
   {
      return 1;         # 1 means true
   }
   else
   {
      return 0;         # 0 means false
   }
}

# This subroutine checks to see if a word starts with a left quote, be it
# `, ", "`, `` or ```
sub StartsWithLeftQuote 
{
   my($word) = @_;

   if (substr($word,0,1) eq "`" || substr($word,0,1) eq "\"" 
         || substr($word,0,2) eq "\"`")
   {
      return 1;         # 1 means true
   }
   else
   {
      return 0;         # 0 means false
   }
}


sub IsRightEnd
{
   my($word) = @_;
   
   if (IsRightParen($word) || IsRightQuote($word))
   {
      return 1;
   }
   else
   {
      return 0;
   }
}

# This subroutine detects if a word starts with a start mark.
sub IsLeftStart
{
   my($word) = @_;

   if(StartsWithLeftQuote($word) || StartsWithLeftParen($word) 
         || Capital($word) eq "Y")
   {
      return 1;
   }
   else
   {
      return 0;
   }
}

# This subroutine checks to see if a word is a right parenthesis, be it ), }
# or >
sub IsRightParen 
{
   my($word) = @_;

   if ($word eq "}" ||  $word eq ")" || $word eq "-RBR-")
   {
      return 1;         # 1 means true
   }
   else
   {
      return 0;         # 0 means false
   }
}

sub IsRightQuote
{
   my($word) = @_;

   if ($word eq "'" ||  $word eq "''" || $word eq "'''" || $word eq "\"" 
         || $word eq "'\"")
   {
      return 1;         # 1 means true
   }
   else
   {
      return 0;         # 0 means false
   }
}

# This subroutine prints out the elements of an array.
sub PrintArray
{
   my($word);

   foreach $word (@_)
   {
      print "Array Element = ($word)\n";
   }
}

1;
