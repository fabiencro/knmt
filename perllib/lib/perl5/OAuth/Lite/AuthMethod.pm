package OAuth::Lite::AuthMethod;

use strict;
use warnings;

use base 'Exporter';

use List::MoreUtils qw(any);

our %EXPORT_TAGS = ( all => [qw/AUTH_HEADER POST_BODY URL_QUERY/] );
our @EXPORT_OK = map { @$_ } values %EXPORT_TAGS;

use constant AUTH_HEADER => 'auth_header';
use constant POST_BODY   => 'post_body';
use constant URL_QUERY   => 'url_query';

sub validate_method {
    my ($class, $method) = @_;
    my @methods = (AUTH_HEADER, POST_BODY, URL_QUERY);
    any { $method eq $_ } @methods;
}

1;

=head1 NAME

OAuth::Lite::AuthMethod - auth method constants.

=head1 SYNOPSIS

    use OAuth::Lite::AuthMethod qw(
        AUTH_HEADER
        POST_BODY
        URL_QUERY
    );

    $consumer = OAuth::Lite::Consumer->new(
        ...
        auth_method => URL_QUERY,
    );

=head1 DESCRIPTION

This mere holds constants for auth method on requesting with OAuth.

=head1 CONSTANTS

=head2 AUTH_HEADER

Using Authorization header for OAuth-authentication.

    Authorization: OAuth realm="http://example.org", oauth_consuemr_key="key", ...

=head2 POST_BODY

Embed OAuth authentication data into request body.

=head2 URL_QUERY

Append OAuth authentication data as query-string.

=head1 METHODS

=head2 validate_method($method)

    unless ( OAuth::Lite::AuthMethod->validate_method($method) ) {
        say 'Invalid auth method.';
    }

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

