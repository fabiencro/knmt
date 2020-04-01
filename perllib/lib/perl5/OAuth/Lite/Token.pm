package OAuth::Lite::Token;

use strict;
use warnings;

use base 'Class::Accessor::Fast';

use OAuth::Lite::Util qw(
    encode_param
    decode_param
    gen_random_key
);

__PACKAGE__->mk_accessors(qw/token secret callback_confirmed/);

=head1 NAME

OAuth::Lite::Token - token class

=head1 SYNOPSIS

    my $token = OAuth::Lite::Token->new(
        token  => 'foo',
        secret => 'bar',
    );

    # or
    my $token = OAuth::Lite::Token->new;
    $tokne->token('foo');
    $secret->secret('bar');

    # and you also can make token which two params are filled in with random values.
    my $token = OAuth::Lite::Token->new_random;
    say $token->token;
    say $token->secret;

    my $encoded = $token->as_encoded;
    say $encoded;

    my $new_token = OAuth::Lite::Token->from_encoded($encoded);
    say $new_token->token;
    say $new_token->secret;

=head1 DESCRIPTION

Token class.

=head1 METHODS

=head2 new

=head3 parameters

=over 4

=item token

=item secret

=back

=cut

sub new {
    my ($class, %args) = @_;
    bless {
        token              => undef,
        secret             => undef,
        callback_confirmed => 0,
        %args
    }, $class;
}

=head2 new_random

Generate new object. and automatically filled token and secret value with random key.

    my $t = OAuth::Lite::Token->new_random;
    say $t->token;
    say $t->secret;

=cut

sub new_random {
    my $class = shift;
    my $token = $class->new;;
    $token->token(gen_random_key());
    $token->secret(gen_random_key());
    $token;
}

=head2 token($token_value)

Getter/Setter for token value.

    $token->token('foo');
    say $token->token;

=head2 secret($token_secret)

Getter/Setter for secret value.

    $token->secret('bar');
    say $token->secret;

=head2 from_encoded($encoded)

Generate token from encoded line (that service provider provides as response of request token.).

    my $line = "oauth_token=foo&oauth_token_secret=bar";
    my $token = OAuth::Lite::Token->from_encoded($encoded);
    say $token->token;
    say $token->secret;

=cut

sub from_encoded {
    my ($class, $encoded) = @_;

    $encoded =~ s/\r\n$//;
    $encoded =~ s/\n$//;

    my $token = $class->new;
    for my $pair (split /&/, $encoded) {
        my ($key, $val) = split /=/, $pair;
        if ($key eq 'oauth_token') {
            $token->token(decode_param($val));
        } elsif ($key eq 'oauth_token_secret') {
            $token->secret(decode_param($val));
        } elsif ($key eq 'oauth_callback_confirmed') {
            my $p = decode_param($val);
            if ($p && $p eq 'true') {
                $token->callback_confirmed(1);
            }
        }
    }
    return $token;
}

=head2 as_encoded

Returns encoded line from token object.

    my $token = OAuth::Lite::Token->new(
        token  => 'foo',
        secret => 'bar',
    );
    say $token->as_encoded; #oauth_token=foo&oauth_token_secret=bar

=cut

sub as_encoded {
    my $self = shift;
    my $token   = $self->token  || '';
    my $secret  = $self->secret || '';
    my $encoded = sprintf(q{oauth_token=%s&oauth_token_secret=%s},
      encode_param($token),
      encode_param($secret));

    $encoded .= q{&oauth_callback_confirmed=true} if $self->callback_confirmed;
    return $encoded;
}

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;
