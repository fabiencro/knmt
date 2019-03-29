package OAuth::Lite::SignatureMethod::PLAINTEXT;

use strict;
use warnings;

use base 'OAuth::Lite::SignatureMethod';

__PACKAGE__->method_name('PLAINTEXT');

use OAuth::Lite::Util qw(encode_param);

=head1 NAME

OAuth::Lite::SignatureMethod::PLAINTEXT - PLAINTEXT signature method class;

=head1 SYNOPSIS

    # Consumer side
    my $method = OAuth::Lite::SignatureMethod::PLAINTEXT->new(
        consumer_secret => 'foo',
        token_secret    => 'bar',
    );

    my $signature = $method->sign($base_string);

    # Service Provider side
    my $method = OAuth::Lite::SignatureMethod::PLAINTEXT->new(
        consumer_secret => 'foo',
        token_secret    => 'bar',
    );
    unless ($method->verify($base_string, $signature)) {
        say "Signature is invalid!";
    }

=head1 DESCRIPTION

PLAINTEXT signature method class.

=head1 METHODS

=head2 method_name

Class method. Returns this method's name.

    say OAuth::Lite::SignatureMethod::PLAINTEXT->method_name;
    # PLAINTEXT

=head2 new(%params)

=head3 parameters

=over 4

=item consumer_secret

=item token_secret

=back

    my $method = OAuth::Lite::SignatureMethod::PLAINTEXT->new(
        consumer_secret => $consumer_secret, 
        token_secret    => $bar,
    );

=head2 sign($base_string)

Generate signature from base string.

    my $signature = $method->sign($base_string);

=head2 verify($base_string, $signature)

Verify signature with base string.

    my $signature_is_valid = $method->verify($base_string, $signature);
    unless ($signature_is_valid) {
        say "Signature is invalid!";
    }

=cut

sub sign {
    my ($self, $base_string) = @_;
    my $key = $self->secrets_as_key();
    $key;
}

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;
