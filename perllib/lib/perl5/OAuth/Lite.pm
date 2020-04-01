package OAuth::Lite;

use strict;
use warnings;

our $VERSION = "1.34";
our $OAUTH_DEFAULT_VERSION = "1.0";
our $USE_DEPRECATED_NORMALIZER = 1;

1;

__END__

=head1 NAME

OAuth::Lite - OAuth framework

=head1 SYNOPSIS

=head2 CONSUMER SIDE

See L<OAuth::Lite::Consumer>

=head2 SERVICE PROVIDER SIDE

See L<OAuth::Lite::ServerUtil>.

or if you want to build server on mod_perl2,
See L<OAuth::Lite::Server::mod_perl2>.

=head1 DESCRIPTION

This framework allows you to make "OAuth Consumer Application" / "OAuth Service Provider" easily.

=head1 SEE ALSO

http://oauth.net/

L<OAuth::Lite::Consumer>
L<OAuth::Lite::ServerUtil>
L<OAuth::Lite::Server::mod_perl2>

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut
