package OAuth::Lite::SignatureMethod::HMAC_SHA1;

use strict;
use warnings;

use base 'OAuth::Lite::SignatureMethod';

__PACKAGE__->method_name('HMAC-SHA1');

use Digest::SHA;
use MIME::Base64;

=head1 NAME

OAuth::Lite::SignatureMethod::HMAC_SHA1 - HMAC_SHA1 signature method class;

=head1 SYNOPSIS

    # Consumer side
    my $method = OAuth::Lite::SignatureMethod::HMAC_SHA1->new(
        consumer_secret => 'foo',
        token_secret    => 'bar',
    );

    my $signature = $method->sign($base_string);

    # Service Provider side
    my $method = OAuth::Lite::SignatureMethod::HMAC_SHA1->new(
        consumer_secret => 'foo',
        token_secret    => 'bar',
    );
    unless ($method->verify($base_string, $signature)) {
        say "Signature is invalid!";
    }

=head1 DESCRIPTION

HMAC_SHA1 signature method class.

=head1 METHODS

=head2 method_name

Class method. Returns this method's name.

    say OAuth::Lite::SignatureMethod::HMAC_SHA1->method_name;
    # HMAC_SHA1

=head2 build_body_hash

    say OAuth::Lite::SignatureMethod::HMAC_SHA1->build_body_hash($content);

=cut

sub build_body_hash {
    my ( $class, $content ) = @_;
    my $hash = MIME::Base64::encode_base64(Digest::SHA::sha1($content));
    $hash =~ s/\n//g;
    return $hash;
}

=head2 new(%params)

=head3 parameters

=over 4

=item consumer_secret

=item token_secret

=back

    my $method = OAuth::Lite::SignatureMethod::HMAC_SHA1->new(
        consumer_secret => $consumer_secret, 
        token_secret    => $bar,
    );

=head2 sign($base_string)

Generate signature from base string.

    my $signature = $method->sign($base_string);

=cut

sub sign {
    my ($self, $base_string) = @_;
    my $key = $self->secrets_as_key();
    my $sign = MIME::Base64::encode_base64(Digest::SHA::hmac_sha1($base_string, $key));
    chomp $sign;
    $sign;
}

=head2 verify($base_string, $signature)

Verify signature with base string.

    my $signature_is_valid = $method->verify($base_string, $signature);
    unless ($signature_is_valid) {
        say "Signature is invalid!";
    }

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;
