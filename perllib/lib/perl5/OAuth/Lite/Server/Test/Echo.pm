package OAuth::Lite::Server::Test::Echo;

use strict;
use warnings;

use base 'OAuth::Lite::Server::mod_perl2';

use bytes ();
use OAuth::Lite::Token;

my $CONSUMER_KEY         = 'key';
my $CONSUMER_SECRET      = 'secret';
my $REQUEST_TOKEN        = 'requesttoken';
my $REQUEST_TOKEN_SECRET = 'requesttokensecret';
my $ACCESS_TOKEN         = 'accesstoken';
my $ACCESS_TOKEN_SECRET  = 'accesstoken_secret';
my $CONTENT              = 'foobarbuz';

=head1 NAME

OAuth::Lite::Server::Test::Echo - simple echo server example

=head1 SYNOPSIS

    PerlModule OAuth::Lite::Server::Test::Echo
    PerlSetVar Realm "http://localhost/"

    <Location /request_token>
        SetHandler perl-script
        PerlSetVar Mode REQUEST_TOKEN
        PerlResponseHandler OAuth::Lite::Server::Test::Echo
    </Location>

    <Location /access_token>
        SetHandler perl-script
        PerlSetVar Mode ACCESS_TOKEN
        PerlResponseHandler OAuth::Lite::Server::Test::Echo
    </Location>

    <Location /resource>
        SetHandler perl-script
        PerlSetVar Mode PROTECTED_RESOURCE
        PerlResponseHandler OAuth::Lite::Server::Test::Echo
    </Location>

=head1 DESCRIPTION

This is very simple example for L<OAuth::Lite::Server::mod_perl2>

=head1 METHODS

=head2 init

=head2 get_request_token_secret

=head2 get_access_token_secret

=head2 get_consumer_secret

=head2 publish_request_token

=head2 publish_access_token

=head2 check_nonce_and_timestamp

=head2 service

=cut

sub init {
    my $self = shift;
    $self->oauth->allow_extra_params(qw/file size/);
    $self->oauth->support_signature_methods(qw/HMAC-SHA1 PLAINTEXT/);
}

sub get_request_token_secret {
    my ($self, $token) = @_;
    unless ($token eq $REQUEST_TOKEN) {
        return $self->error(q{Invalid token});
    }
    $REQUEST_TOKEN_SECRET;
}

sub get_access_token_secret {
    my ($self, $token) = @_;
    unless ($token eq $ACCESS_TOKEN) {
        return $self->error(q{Invalid token});
    }
    $ACCESS_TOKEN_SECRET;
}

sub get_consumer_secret {
    my ($self, $consumer_key) = @_;
    unless ($consumer_key eq $CONSUMER_KEY) {
        return $self->error(q{Invalid consumer key});
    }
    $CONSUMER_SECRET;
}

sub publish_request_token {
    my ($self, $consumer_key) = @_;
    unless ($consumer_key eq $CONSUMER_KEY) {
        return $self->error(q{Invalid consumer key});
    }
    my $token = OAuth::Lite::Token->new(
        token  => $REQUEST_TOKEN,
        secret => $REQUEST_TOKEN_SECRET,
    );
    $token;
}

sub publish_access_token {
    my ($self, $consumer_key, $request_token) = @_;
    unless ($consumer_key eq $CONSUMER_KEY) {
        return $self->error(q{Invalid consumer key});
    }
    unless ($request_token eq $REQUEST_TOKEN) {
        return $self->error(q{Invalid token});
    }
    my $token = OAuth::Lite::Token->new(
        token  => $ACCESS_TOKEN,
        secret => $ACCESS_TOKEN_SECRET,
    );
    $token;
}

sub check_nonce_and_timestamp {
    my ($self, $consumer_key, $nonce, $timestamp) = @_;
    1;
}

sub service {
    my ($self, $params) = @_;
    $self->set_authenticate_header();
    $self->request->status(200);
    $self->request->content_type(q{text/plain; charset=utf-8});
    $self->request->set_content_length(bytes::length($CONTENT));
    $self->request->print($CONTENT);
    return Apache2::Const::OK;
}

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;

