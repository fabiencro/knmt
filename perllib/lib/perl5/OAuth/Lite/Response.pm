package OAuth::Lite::Response;

use strict;
use warnings;

use base 'Class::Accessor::Fast';

use OAuth::Lite::Util qw(decode_param);

__PACKAGE__->mk_accessors(qw/token/);

use OAuth::Lite::Token;

=head1 NAME

OAuth::Lite::Response - response class

=head1 SYNOPSIS

    my $res = $consumer->obtain_access_token(
        ...
    );

    my $token = $res->token;
    say $token->token;
    say $token->secret;

    my $other_param = $res->param('xauth_expires');

=head1 DESCRIPTION

Response class

=head1 METHODS

=head2 new

=cut

sub new {
    my $class = shift;
    bless {
        _params => {},
        token   => undef,
    }, $class;
}

=head2 from_encoded

Generate response from encoded line (that service provider provides as response of request token.).

    my $line = "oauth_token=foo&oauth_token_secret=bar&xauth_expires=0";
    my $res = OAuth::Lite::Response->from_encoded($encoded);

    my $token = $res->token;
    say $token->token;
    say $token->secret;

    say $res->param('xauth_expires');

=cut

sub from_encoded {
    my ($class, $encoded) = @_;
    $encoded =~ s/\r\n$//;
    $encoded =~ s/\n$//;
    my $res = $class->new;
    my $token = OAuth::Lite::Token->new;
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
        } else {
            $res->param($key, decode_param($val));
        }
    }
    $res->token($token);
    $res;
}

=head2 param

Get parameter.

    say $res->param('xauth_expires');

=cut

sub param {
    my ($self, $key, $value) = @_;
    if (defined $value) {
        $self->{_params}{$key} = $value;
    }
    $self->{_params}{$key};
}

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut


1;

