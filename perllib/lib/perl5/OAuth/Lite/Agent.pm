package OAuth::Lite::Agent;

use strict;
use warnings;

use LWP::UserAgent;
use List::MoreUtils;

sub new {
    my ($class, $ua) = @_;
    my $self = bless {
        ua => $ua || LWP::UserAgent->new,
    }, $class;
    return $self;
}

sub agent {
    my ($self, $agent) = @_;
    $self->{ua}->agent($agent);
}

sub request {
    my ($self, $req) = @_;
    $req = $self->filter_request($req);
    my $res = $self->{ua}->request($req);
    return $self->filter_response($res);
}

sub filter_request {
    my ($self, $req) = @_;
    my $accept_encoding = $req->header('Accept-Encoding') || '';
    my @encodings = split(/\s*,\s*/, $accept_encoding);
    if (@encodings == 0 || List::MoreUtils::none { $_ eq 'gzip' } @encodings) {
        push(@encodings, 'gzip');
    }
    $req->header('Accept-Encoding' => join(', ', @encodings));
    return $req;
}


sub filter_response {
    my ($self, $res) = @_;
    return $res;
}

1;

=head1 NAME

OAuth::Lite::Agent - default agent class

=head1 SYNOPSIS

    $agent = OAuth::Lite::Agent->new;
    my $res = $agent->request($req);

=head1 DESCRIPTION

Default user agent for OAuth::Lite::Consuemr.

=head1 METHODS

=head2 new

constructor.

    OAuth::Lite::Agent->new();

You can pass custom agent, that is required to implement
'request' and 'agent' methods corresponding to the same named methods of LWP::UserAgnet.

    my $custom_agent = LWP::UserAgent->new(timeout => 10);
    OAuth::Lite::Agent->new($custom_agent);

=head2 agent

agent setter

    $agent->agent("MyAgent/1.0.0");

=head2 request

As same as LWP::UserAgent, pass HTTP::Request object
and returns HTTP::Response object.

    $res = $agent->request($req);

=head2 filter_request

filter the HTTP::Request object before request.

=head2 filter_response

filter the HTTP::Response object after request.

=head1 SEE ALSO

http://oauth.pbwiki.com/ProblemReporting

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

