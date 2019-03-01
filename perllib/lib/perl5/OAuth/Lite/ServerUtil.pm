package OAuth::Lite::ServerUtil;

use strict;
use warnings;

use base 'Class::ErrorHandler';

use OAuth::Lite::Util qw(
    decode_param
    create_signature_base_string
);
use OAuth::Lite::Problems qw(:all);
use List::MoreUtils qw(any none);
use UNIVERSAL::require;
use Carp ();

=head1 NAME

OAuth::Lite::ServerUtil - server side utility

=head1 SYNOPSIS

    my $util = OAuth::Lite::ServerUtil->new;
    $util->support_signature_method('HMAC-SHA1');
    $util->allow_extra_params(qw/file size/);

    unless ($util->validate_params($oauth_params)) {
        return $server->error(400, $util->errstr);
    }

    $util->verify_signature(
        method          => $r->method,
        params          => $oauth_params,
        url             => $request_uri,
        consumer_secret => $consumer->secret,
    ) or return $server->error(401, $util->errstr);

And see L<OAuth::Lite::Server::mod_perl2> source code.

=head1 DESCRIPTION

This module helps you to implement application that acts as OAuth Service Provider.

=head1 PAY ATTENTION

If you use OAuth 1.31 or older version, its has invalid way to normalize params.
(when there are two or more same key and they contain ASCII and non ASCII value)

But the many services have already supported deprecated version, 
and the correct way breaks backward compatibility.
So, from 1.32, supported both correct and deprecated method. 

use $OAuth::Lite::USE_DEPRECATED_NORMALIZER to switch behaviour.
Currently 1 is set by default to keep backward compatibility.

    use OAuth::Lite::ServerUtil;
    use OAuth::Lite;

    $OAuth::Lite::USE_DEPRECATED_NORMALIZER = 0;
    ...


=head1 METHODS

=head2 new

Constructor

    my $util = OAuth::Lite::ServerUtil->new;

Set strict true by default, and it judge unsupported param as invalid when validating params.
You can build ServerUtil as non-strict mode, then it accepts unsupported parameters.

    my $util = OAuth::Lite::ServerUtil->new( strict => 0 );

=cut

sub new {
    my $class = shift;
    my %args = @_;
    my $strict = exists $args{strict} ? $args{strict} : 1;
    my $self = bless {
        supported_signature_methods => {},
        allowed_extra_params        => [],
        strict                      => $strict,
    }, $class;
    $self;
}

=head2 allow_extra_param($param_name);

When you validate oauth parameters, if an extra parameter
is included, the validation will fail.

    my $params = {
        oauth_version => '1.0',
        ...and other oauth parameters,
    };
    $params->{file} = "foo.jpg";

    # fail!
    unless ($util->validate_params($params)) {
        $your_app->error( $util->errstr );
    }

So, if you want allow extra parameter, use this method.

    $util->allow_extra_param('file');

    my $params = {
        oauth_version => '1.0',
        ...and other oauth parameters,
    };
    $params->{file} = "foo.jpg";

    # Now this results successfully.
    unless ($util->validate_params($params)) {
        $your_app->error( $util->errstr );
    }

=cut

sub allow_extra_param {
    my ($self, $param) = @_;
    push @{ $self->{allowed_extra_params} }, $param;
}

=head2 allow_extra_params($param1, $param2, ...)

You can allow multiple extra parameters at once.

    $util->allow_extra_params(qw/file size/);

=cut

sub allow_extra_params {
    my $self = shift;
    $self->allow_extra_param($_) for @_;
}

=head2 support_signature_method($method_class_name);

Set the signature method class's name that your server can supports.

    $util->support_signature_method('HMAC_SHA1');

This method requires indicated signature method class inside.
So, you should install OAuth::Lite::SignatureMethod::$method_class_name beforehand.
For example, when your choise is HMAC_SHA1, you need to have
OAuth::Lite::SignatureMethod::HMAC_SHA1 installed in your server.

=cut

sub support_signature_method {
    my ($self, $method_class) = @_;
    $method_class =~ s/-/_/g;
    my $class = join('::', 'OAuth::Lite::SignatureMethod', $method_class);
    $class->require or Carp::croak sprintf(q{Couldn't require class, %s}, $class);
    $self->{supported_signature_methods}{$class->method_name} = $class;
}

=head2 support_signature_methods($method1, $method2, ...);

You can set multiple signature method class at once.

    $util->support_signature_methods(qw/HMAC_SHA1 RSA_SHA1/);

=cut

sub support_signature_methods {
    my $self = shift;
    $self->support_signature_method($_) for @_;
}

=head2 validate_params($params, [$check_token]);

Check if the request includes all required params
and doesn't include unsupported params.
It doesn't check unsupported params when working on strict mode.

    unless ($util->validate_params($params)) {
        $your_app->error( $util->errstr );
    }

When the request is to exchange tokens or to access to protected resources,
pass 1 for second argument. This flag indicates that oauth_token param is needed.

    unless ($util->validate_params($params, 1)) {
        $your_app->error( $util->errstr );
    }

=cut

sub validate_params {
    my ($self, $origin_params, $check_token) = @_;
    my $params = {%$origin_params}; #copy
    delete $params->{oauth_consumer_key} or return $self->error(PARAMETER_ABSENT);
    delete $params->{oauth_nonce} or return $self->error(PARAMETER_ABSENT);
    delete $params->{oauth_timestamp} or return $self->error(PARAMETER_ABSENT);
    delete $params->{oauth_signature_method} or return $self->error(PARAMETER_ABSENT);
    delete $params->{oauth_signature} or return $self->error(PARAMETER_ABSENT);
    delete $params->{oauth_version};
    if ($check_token) {
        delete $params->{oauth_token} or return $self->error(PARAMETER_ABSENT);
    }
    if ( $self->{strict} ) {
        my @extra_params = keys %$params;
        my @allowed = @{ $self->{allowed_extra_params} };
        for my $extra ( @extra_params ) {
            if (none { $extra eq $_ } @allowed) {
                return $self->error(PARAMETER_REJECTED);
            }
        }
    }
    1;
}

=head2 validate_signature_method($method_name)

    unless ($util->validate_signature_method('HMAC-SHA1')) {
        
        $your_app->error(qq/Unsupported signature method/);
        ...
    }

=cut

sub validate_signature_method {
    my ($self, $method) = @_;
    return unless $method;
    any { $_ eq $method } keys %{$self->{supported_signature_methods}};
}

=head2 verify_signature(%args)

=over 4

=item method - HTTP request method

=item params - parameters hash reference

=item url - requested uri

=item consumer_secret - consumer secret value(optional)

=item token_secret - token secret value(optional)

=back

    # you can omit consumer_secret and token_secret if you don't need them.
    $util->verify_signature(
        method          => $r->method, 
        params          => $params,
        url             => $requested_uri,
        consumer_secret => $consumer_secret,
        token_secret    => $token_secret,
    ) or die $utl->errstr;

=cut

sub verify_signature {
    my ($self, %args) = @_;

    my $http_method = $args{method} or Carp::croak(qq/method not found/);
    my $url         = $args{url}    or Carp::croak(qq/url not found/);
    my $params      = $args{params} or Carp::croak(qq/params not found/);

    my $consumer_secret  = $args{consumer_secret} || '';
    my $token_secret     = $args{token_secret} || '';
    my $signature_method = $params->{oauth_signature_method};
    my $signature        = $params->{oauth_signature};

    my $base_string = create_signature_base_string($http_method, $url, $params);
    unless ($self->validate_signature_method($signature_method)) {
        return $self->error(SIGNATURE_METHOD_REJECTED);
    }
    my $method_class = $self->{supported_signature_methods}{$signature_method};
    my $method = $method_class->new(
        consumer_secret => $consumer_secret,
        token_secret    => $token_secret,
    );
    unless ($method->verify($base_string, $signature)) {
        return $self->error(SIGNATURE_INVALID);
    }
    1;
}

=head1 SEE ALSO

L<OAuth::Lite::Server::mod_perl2>

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;
