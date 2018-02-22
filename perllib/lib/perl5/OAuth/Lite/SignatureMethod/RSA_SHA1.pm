package OAuth::Lite::SignatureMethod::RSA_SHA1;

use strict;
use warnings;

use base 'OAuth::Lite::SignatureMethod';

use Crypt::OpenSSL::RSA;
use Digest::SHA ();
use MIME::Base64 ();

__PACKAGE__->method_name('RSA-SHA1');

=head1 NAME

OAuth::Lite::SignatureMethod::RSA_SHA1 - RSA_SHA1 signature method class;

=head1 SYNOPSIS

    # Consumer side
    my $signer = OAuth::Lite::SignatureMethod::RSA_SHA1->new(
        consumer_secret => $rsa_private_key,
    );

    my $signature = $signer->sign($base_string);

    # Service Provider side
    my $verifier = OAuth::Lite::SignatureMethod::RSA_SHA1->new(
        consumer_secret => $rsa_public_key,
    );
    unless ($verifier->verify($base_string, $signature)) {
        say "Signature is invalid!";
    }

=head1 DESCRIPTION

RSA_SHA1 signature method class.

=head1 PRIVATE KEY AND PUBLIC KEY

RSA needs two keys that called public key and private key.
If you runs OAuth consumer application and want to use this RSA_SHA1 method
for signature on OAuth protocol, you have to prepare these keys.

To generate them in Perl, here is an example.

    my $rsa = Crypt::OpenSSL::RSA->generate_key(1024);
    my $public_key  = $rsa->get_public_key_string();
    my $private_key = $rsa->get_private_key_string();

And prior to use OAuth protocol with a service provider,
you have to register public key onto the service provider.

=head1 METHODS

=head2 method_name

Class method. Returns this method's name.

    say OAuth::Lite::SignatureMethod::RSA_SHA1->method_name;
    # RSA_SHA1

=head2 build_body_hash

    say OAuth::Lite::SignatureMethod::RSA_SHA1->build_body_hash($content);

=cut

sub build_body_hash {
    my ( $class, $content ) = @_;
    my $hash = MIME::Base64::encode_base64(Digest::SHA::sha1($content));
    $hash =~ s/\n//g;
    return $hash;
}

=head2 new(%params)

On the consumer side, you should pass RSA private key for consumer_secret,
But for service provider to verify signature, pass RSA public key that
consumer register on service provider beforehand.

=head3 parameters

=over 4

=item consumer_secret

=back

    my $signer = OAuth::Lite::SignatureMethod::RSA_SHA1->new(
        consumer_secret => $rsa_private_key, 
    );

    my $verifier = OAuth::Lite::SignatureMethod::RSA_SHA1->new(
        consumer_secret => $rsa_public_key, 
    );

=head2 sign($base_string)

Generate signature from base string.

    my $signature = $method->sign($base_string);

=cut

sub sign {
    my ($self, $base_string) = @_;
    my $private_key_pem = $self->{consumer_secret};
    my $private_key = Crypt::OpenSSL::RSA->new_private_key($private_key_pem);
    my $signature = MIME::Base64::encode_base64($private_key->sign($base_string));
    chomp $signature;
    $signature;
}

=head2 verify($base_string, $signature)

Verify signature with base string.

    my $signature_is_valid = $method->verify($base_string, $signature);
    unless ($signature_is_valid) {
        say "Signature is invalid!";
    }

=cut

sub verify {
    my ($self, $base_string, $signature) = @_;
    my $public_key_pem = $self->{consumer_secret};
    my $public_key = Crypt::OpenSSL::RSA->new_public_key($public_key_pem);
    return $public_key->verify($base_string, MIME::Base64::decode_base64($signature));
}

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;
