package OAuth::Lite::Util;

use strict;
use warnings;

use OAuth::Lite;
use URI;
use URI::Escape;
use Crypt::OpenSSL::Random;
use Carp ();

use base 'Exporter';

our %EXPORT_TAGS = ( all => [qw/
    gen_random_key
    encode_param
    decode_param
    create_signature_base_string
    parse_auth_header
    build_auth_header
    normalize_params
/]);

our @EXPORT_OK = map { @$_ } values %EXPORT_TAGS;

=head1 NAME

OAuth::Lite::Util - utility for OAuth

=head1 SYNPSIS

    use OAuth::Lite::Util qw(
        gen_random_key
        encode_param
        decode_param
        create_signature_base_string
        parse_auth_header
    );

    my $random = gen_random_key(8);
    my $enocded = encode_param($param);
    my $deocded = decode_param($encoded);

    my $base_string = create_signature_base_string('GET',
        'http://example.com/path?query', $params);

    my $header = q{OAuth realm="http://example.com/api/resource", oauth_consumer_key="hogehoge", ... };
    my ($realm, $oauth_params) = parse_auth_header($header);
    say $realm;
    say $oauth_params->{oauth_consumer_key};
    say $oauth_params->{oauth_version};
    ...

=head1 DESCRIPTION

Utilty functions for OAuth are implemented here.

=head1 PAY ATTENTION

If you use OAuth 1.31 or older version, its has invalid way to normalize params.
(when there are two or more same key and they contain ASCII and non ASCII value)

But the many services have already supported deprecated version, 
and the correct way breaks backward compatibility.
So, from 1.32, supported both correct and deprecated method. 

use $OAuth::Lite::USE_DEPRECATED_NORMALIZER to switch behaviour.
Currently 1 is set by default to keep backward compatibility.

    use OAuth::Lite::Util;
    use OAuth::Lite;
    $OAuth::Lite::USE_DEPRECATED_NORMALIZER = 0;
    ...



=head1 METHODS

=head2 gen_random_key($length)

Generate random octet string.
You can indicate the byte-length of generated string. (10 is set by default)
If 10 is passed, returns 20-length octet string.

    use OAuth::Lite::Util qw(gen_random_key);
    my $key1 = gen_random_key();
    my $key2 = gen_random_key();

=cut

sub gen_random_key {
    my $length = shift || 10;
    return unpack("H*", Crypt::OpenSSL::Random::random_bytes($length));
}

=head2 encode_param($param)

Encode parameter according to the way defined in OAuth Core spec.

=cut

sub encode_param {
    my $param = shift;
    URI::Escape::uri_escape($param, '^\w.~-');
}

=head2 decode_param($encoded_param)

Decode the encoded parameter.

=cut

sub decode_param {
    my $param = shift;
    URI::Escape::uri_unescape($param);
}

=head2 create_signature_base_string($http_method, $request_uri, $params);

    my $method = "GET";
    my $uri = "http://example.com/api/for/some-resource";
    my $parmas = {
        oauth_consumer_key     => 'foo-bar',
        oauth_signature_method => 'HMAC-SHA1',
        oauth_version => '1.0',
        ...
    };
    my $base_string = create_signature_base_string($method, $uri, $params);

=cut

sub create_signature_base_string {
    my ($method, $url, $params) = @_;
    $method = uc $method;
    $params = {%$params};
    delete $params->{oauth_signature};
    delete $params->{realm};
    my $normalized_request_url = normalize_request_url($url);
    my $normalized_params = normalize_params($params);
    my $signature_base_string = join('&', map(encode_param($_),
        $method, $normalized_request_url, $normalized_params));
    $signature_base_string;
}

=head2 normalize_request_url($url);

Normalize url according to the way the OAuth Core spec defines.

    my $string = normalize_request_url('http://Example.com:80/path?query');
    # http://example.com/path
    my $string = normalize_request_url('https://Example.com:443/path?query');
    # https://example.com/path
    my $string = normalize_request_url('http://Example.com:8080/path?query');
    # http://example.com:8080/path

=cut

sub normalize_request_url {
    my $uri = shift;
    $uri = URI->new($uri) unless (ref $uri && ref $uri eq 'URI');
    unless (lc $uri->scheme eq 'http' || lc $uri->scheme eq 'https') {
        Carp::croak qq/Invalid request url, "$uri"/;
    }
    my $port = $uri->port;
    my $request_url = ($port && ($port == 80 || $port == 443))
        ? sprintf(q{%s://%s%s}, lc($uri->scheme), lc($uri->host), $uri->path)
        : sprintf(q{%s://%s:%d%s}, lc($uri->scheme), lc($uri->host), $port, $uri->path);
    $request_url;
}

=head2 normalize_params($params);

Sort and encode params and concatenates them
according to the way OAuth Core spec defines.

    my $string = normalize_params({
        a => 1, c => 'hi%20there', f => [25, 50, 'a'], z => [ 'p', 't' ]
    });

=cut

sub normalize_params {
    $OAuth::Lite::USE_DEPRECATED_NORMALIZER 
        ? _normalize_params_deprecated(@_)
        : _normalize_params(@_);
}

sub _normalize_params {
    my $params = shift;
    my %encoded_params = ();
    for my $k (keys %$params) {
        if (!ref $params->{$k}) {
            $encoded_params{encode_param($k)} = encode_param($params->{$k});
        } elsif (ref $params->{$k} eq 'ARRAY') {
            $encoded_params{encode_param($k)} = [ map { encode_param($_) } @{$params->{$k}} ];
        }
    }
    my @pairs = ();
    for my $k (sort keys %encoded_params) {
        if (!ref $encoded_params{$k}) {
            push @pairs, sprintf(q{%s=%s}, $k, $encoded_params{$k});
        }
        elsif (ref $encoded_params{$k} eq 'ARRAY') {
            for my $v (sort @{ $encoded_params{$k} }) {
                push @pairs, sprintf(q{%s=%s}, $k, $v);
            }
        }
    }
    return join('&', @pairs);
}

sub _normalize_params_deprecated {
    my $params = shift;
    my @pairs = ();
    for my $k (sort keys %$params) {
        if (!ref $params->{$k}) {
            push @pairs, 
                sprintf(q{%s=%s}, encode_param($k), encode_param($params->{$k}));
        }
        elsif (ref $params->{$k} eq 'ARRAY') {
            for my $v (sort @{ $params->{$k} }) {
                push @pairs, 
                    sprintf(q{%s=%s}, encode_param($k), encode_param($v));
            }
        }
    }
    return join('&', @pairs);
}

=head2 parse_auth_header($header)

Parse authorization/www-authentication header for OAuth.
And return the realm and other params.

    # service provider side
    my $header = $r->headers_in->{Authorization};
    my ($realm, $params) = parse_auth_header($header);
    say $params->{oauth_token};
    say $params->{oauth_consumer_key};
    say $params->{oauth_signature_method};
    ...

    # consumer side
    my $header = $res->header('WWW-Authenticate');
    my ($realm) = parse_auth_header($header);

=cut

sub parse_auth_header {
    my $header = shift;
    $header =~ s/^\s*OAuth\s*//;
    my $params = {};
    for my $attr (split /,\s*/, $header) {
        my ($key, $val) = split /=/, $attr, 2;
        $val =~ s/^"//;
        $val =~ s/"$//;
        $params->{$key} = decode_param($val);
    }
    my $realm = delete $params->{realm};
    return wantarray ? ($realm, $params) : $realm;
}

=head2 build_auth_header(%params)

    my $header = build_auth_header($realm, {
        oauth_consumer_key     => '...', 
        oauth_signature_method => '...',
        ... and other oauth params
    });

=cut

sub build_auth_header {
    my ($realm, $params) = @_;
    my $head = sprintf q{OAuth realm="%s"}, $realm || '';
    my $authorization_header = join(', ', $head,
        sort { $a cmp $b } map(sprintf(q{%s="%s"}, encode_param($_), encode_param($params->{$_})),
            grep { /^x?oauth_/ } keys %$params));
    $authorization_header;
}

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;
