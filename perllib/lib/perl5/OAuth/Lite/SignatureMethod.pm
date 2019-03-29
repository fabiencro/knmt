package OAuth::Lite::SignatureMethod;

use strict;
use warnings;

use base 'Class::Data::Accessor';

__PACKAGE__->mk_classaccessor('method_name');

use OAuth::Lite::Util qw(encode_param);

=head1 NAME

OAuth::Lite::SignatureMethod - signature method base class

=head1 SYNOPSIS

    say $signature_method_class->method_name;

    my $method = $signature_method_class->new(
        consumer_secret => $consumer_secret,
        token_secret    => $token_secret,
    );

    my $signature = $method->sign($text);

    if ($method->verify($text, $signature)) {
        say "valid signature";
    }


=head1 DESCRIPTION

SignatureMethod base class.
Create subclasses for arbitrary signature method inheriting this class.

=head1 METHODS

=head2 build_body_hash($content)

Build body hash according to the spec
http://oauth.googlecode.com/svn/spec/ext/body_hash/1.0/drafts/4/spec.html

    my $hash = $method_class->build_body_hash($content);
    say $hash;

=cut

sub build_body_hash {
    my ( $class, $content ) = @_;
    return;
}

=head2 method_name($method_name)

Set signature method name.
Use this in subclass.

    $method_class->method_name('HMAC-SHA1');
    say $method_class->method_name;

=head2 new(%params)

    my $method = $signature_method_class->new(
        consumer_secret => $consumer_secret,
        token_secret    => $token_secret,
    );

=cut

sub new {
    my ($class, %args) = @_;
    my $self = bless {
        consumer_secret => $args{consumer_secret} || '',
        token_secret    => $args{token_secret}    || '',
    }, $class;
    $self;
}

=head2 secrets_as_key

Returns consumer_secret and token_secret as encoded key format.

    my $key = $method->secrets_as_key;

=cut

sub secrets_as_key {
    my $self = shift;
    join '&', (map encode_param($self->{$_}),
        qw/consumer_secret token_secret/);
}

=head2 sign($base_text)

Create signature from passed base text

This is an abstract method.
Define this in subclass.

    my $signature = $method->sign($base_text);

=cut

sub sign {
    my ($self, $base_string) = @_;
}

=head2 verify($base_text, $signature)

Check if signature is valid with base text

    my $signature_is_valid = $method->verify($base_text, $signature);
    if ($signature_is_valid) {
        ...
    }

=cut

sub verify {
    my ($self, $base_string, $signature) = @_;
    return ($signature eq $self->sign($base_string)) ? 1 : 0;
}

=head1 SEE ALSO

L<OAuth::Lite::SignatureMethod::HMAC_SHA1>
L<OAuth::Lite::SignatureMethod::PLAINTEXT>
L<OAuth::Lite::SignatureMethod::RSA_SHA1>

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;
