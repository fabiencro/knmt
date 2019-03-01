package OAuth::Lite::Server::mod_perl2;

use strict;
use warnings;

use Apache2::Connection  ();
use Apache2::RequestIO   ();
use Apache2::RequestRec  ();
use Apache2::RequestUtil ();
use Apache2::Response    ();
use Apache2::URI         ();
use Apache2::ServerRec   ();
use URI::Escape          ();
use Apache2::Const -compile => qw(OK);

use List::MoreUtils qw(none any);
use bytes ();

use OAuth::Lite::Util qw(:all);
use OAuth::Lite::ServerUtil;
use OAuth::Lite::AuthMethod qw(:all);
use OAuth::Lite::Problems qw(:all);

use base qw(
    Class::Accessor::Fast
    Class::ErrorHandler
);

use constant PROTECTED_RESOURCE => 'PROTECTED_RESOURCE';
use constant REQUEST_TOKEN      => 'REQUEST_TOKEN';
use constant ACCESS_TOKEN       => 'ACCESS_TOKEN';

__PACKAGE__->mk_accessors(qw/request realm oauth xrds_location/);

=head1 NAME

OAuth::Lite::Server::mod_perl2 - mod_perl2 OAuth server

=head1 SYNOPSIS

Inherit this class, build your service with mod_perl2.
For example, write MyServiceWithOAuth.pm
And the source-code of L<OAuth::Lite::Server::Test::Echo> is nice example.
See it.

    package MyServiceWithOAuth;
    use base 'OAuth::Lite::Server::mod_perl2';

    sub init {
        my $self = shift;
        $self->oauth->allow_extra_params(qw/file size/);
        $self->oauth->support_signature_methods(qw/HMAC-SHA1 PLAINTEXT/);
    }

    sub get_request_token_secret {
        my ($self, $token_string) = @_;
        my $token = MyDB::Scheme->resultset('RequestToken')->find($token_string);
        unless ($token
            &&  $token->is_authorized_by_user
            &&  !$token->is_exchanged_to_access_token
            &&  !$token->is_expired) {
            return $self->error(q{Invalid token});
        }
        return $token->secret;
    }

    sub get_access_token_secret {
        my ($self, $token_string, $consumer_key) = @_;
        my $token = MyDB::Scheme->resultset('AccessToken')->find($token_string);
        unless ($token
            && !$token->is_expired) {
            return $self->error(q{Invalid token});
        }
        return $token->secret;
    }

    sub get_consumer_secret {
        my ($self, $consumer_key) = @_;
        my $consumer = MyDB::Shceme->resultset('Consumer')->find($consumer_key);
        unless ($consumer
             && $consumer->is_valid) {
            return $self->error(q{Inalid consumer_key});
        }
        return $consumer->secret;
    }

    sub publish_request_token {
        my ($self, $consumer_key, $callback_url) = @_;
        my $token = OAuth::Lite::Token->new_random;
        MyDB::Scheme->resultset('RequestToken')->create({
            token        => $token->token,
            secret       => $token->secret,
            realm        => $self->realm,
            consumer_key => $consumer_key,
            expired_on   => '',
            callback     => $callback_url,
        });
        return $token;
    }

    sub publish_access_token {
        my ($self, $consumer_key, $request_token_string, $verifier) = @_;
        my $request_token = MyDB::Scheme->resultset('RequestToken')->find($request_token_string);
        unless ($request_token
            &&  $request_token->is_authorized_by_user
            && !$request_token->is_exchanged_to_access_token
            && !$request_token->is_expired
            &&  $request_token->has_verifier
            &&  $request_token->verifier eq $verifier) {
            return $self->error(q{Invalid token});
        }
        my $access_token = OAuth::Lite::Token->new_random;
        MyDB::Scheme->resultset('AccessToken')->create({
            token        => $request_token->token,
            realm        => $self->realm,
            secret       => $request_token->secret,
            consumer_key => $consumer_key,
            author       => $request_token->author,
            expired_on   => '',
        });

        $request_token->is_exchanged_to_access_token(1);
        $request_token->update();

        return $access_token;
    }

    sub check_nonce_and_timestamp {
        my ($self, $consumer_key, $nonce, $timestamp) = @_;
        my $request_log = MyDB::Scheme->resultset('RequestLog');
        # check against replay-attack
        my $count = $request_log->count({
            consumer_key => $consumer_key,
            -nest => [
                nonce     => $nonce,
                timestamp => { '>' => $timestamp }, 
            ], 
        });
        if ($count > 0) {
            return $self->error(q{Invalid timestamp or consumer});
        }
        # save new request log.
        $request_log->create({
            consumer_key => $consumer_key,
            nonce        => $nonce,
            timestamp    => $timestamp,
        });
        return 1;
    }

    sub service {
        my $self = shift;
    }

in httpd.conf

    PerlSwitches -I/var/www/MyApp/lib
    PerlModule MyServiceWithOAuth

    <VirtualHost *>

        ServerName api.example.com
        DocumentRoot /var/www/MyApp/root

        PerlSetVar Realm "http://api.example.com/picture"

        <Location /picture/request_token>
            SetHandler perl-script
            PerlSetVar Mode REQUEST_TOKEN
            PerlResponseHandler MyServiceWithOAuth
        </Location>

        <Location /picture/access_token>
            SetHandler perl-script
            PerlSetVar Mode ACCESS_TOKEN
            PerlResponseHandler MyServiceWithOAuth
        </Location>

        <Location /picture/resource>
            SetHandler perl-script
            PerlSetVar Mode PROTECTED_RESOURCE
            PerlResponseHandler MyServiceWithOAuth
        </Location>

    </VirtualHost>

=head1 DESCRIPTION

This module is for mod_perl2 PerlResponseHandler, and allows you to
build services with OAuth easily.

=head1 TUTORIAL

All you have to do is to make a package inheritting this module,
and override some methods, and in httpd.conf file, write
three configuration, each configuration needs to be set Mode value.
The each value must be REQUEST_TOKEN, ACCESS_TOKEN, or PROTECTED_RESOURCE.
And the Realm value is needed for each resource.

The methods you have to override is bellow.

=head1 METHODS YOU HAVE TO OVERRIDE

=head2 init

In this method, you can do some initialization.
For example, set what signature method your service supports,
and what extra-param is allowed.

    sub init {
        my $self = shift;
        $self->oauth->support_signature_method(qw/HMAC-SHA1 PLAINTEXT/);
        $self->oauth->allow_extra_params(qw/file size/);
    }

=head2 get_request_token_secret($token_string)

In this method, you should check if the request-token-string is
valid, and returns token-secret value corresponds to the
token value passed as argument.
If the token is invalid, you should call 'error' method.

=head2 get_access_token_secret($token_string, $consumer_key)

In this method, you should check if the access-token-string is
valid, and returns token-secret value corresponds to the
token value passed as argument.
If the token is invalid, you should call 'error' method.

=head2 get_consumer_secret($consumer_key)

In this method, you should check if the consumer_key is valid,
and returns consumer_secret value corresponds to the consumer_key
passed as argument.
If the consumer is invalid, you should call 'error' method.

=head2 check_nonce_and_timestamp($consumer_key, $nonce, $timestamp)

Check passed nonce and timestamp.
Among requests the consumer send service-provider, there shouldn't be
same nonce, and new timestamp should be greater than old ones.
If they are valid, returns 1, or returns 0.

=head2 publish_request_token($consumer_key, $callback_url)

Create new request-token, and save it,
and returns it as L<OAuth::Lite::Token> object.

=head2 publish_access_token($consumer_key, $request_token_string, $verifier)

If the passed request-token is valid,
create new access-token, and save it,
and returns it as L<OAuth::Lite::Token> object.
And disables the exchanged request-token.

=head2 verify_requestor_approval($consumer_key, $requestor_id)

When the request is for OpenSocial Reverse Phone Home,
Check if the requestor has already given approval to consumer
to access the requestor's data.

=head2 service

Handle protected resource.
This method should returns Apache2::Const::OK.

    sub service {
        my $self = shift;
        my $params = $self->{params};
        my $token_string = $params->{oauth_token};
        my $access_token = MyDB::Scheme->resultset('RequestToken')->find($token_string);
        my $user = $access_token->author;

        my $resource = $user->get_my_some_resource();

        $self->request->status(200);
        $self->request->content_type(q{text/html; charset=utf-8});
        $self->print($resource);
        return Apache2::Const::OK;
    }

=head1 API

=head2 handler

Trigger method as response handler.

=head2 new

Constructor

=head2 request

Returns Apache request object.
See L<Apache2::RequestRec>, L<Apache2::RequestIO>, and etc...

    $self->request;

=head2 realm

The realm value you set in httpd.conf by PerlSetVar.

=head2 oauth

Returns l<OAuth::Lite::ServerUtil> object.

=head2 allow_extra_param

=head2 allow_extra_params

=head2 support_signature_method

=head2 support_signature_methods

These methods are just only delegate methods.
For example, 

    $self->allow_extra_param('foo');

is same as

    $self->oauth->allow_extra_param('foo');

=head2 request_method

Request method (Upper Case).
When the raw request method is POST and X-HTTP-Method-Override is define in header,
return the value of X-HTTP-Method-Override.

=head2 request_uri

Returns request uri

=head2 request_body

Requets body data when the request's http-method is POST or PUT

=head2 set_authenticate_header

Set proper 'WWW-Authentication' response header

=head2 is_required_request_token

Check if current request requires request-token.

=head2 is_required_access_token

Check if current request requires access-token.

=head2 is_required_protected_resource

Check if current request requires protected-resource.

=head2 is_consumer_request

Chekc if the server accepts consumer-request and
the request is for protected resource without token.

=head2 is_reverse_phone_home

Check if the server accepts open-social reverse-phone-home
and the requests is for protected resource without token.

=head2 xrds_location

If you want to support OAuth Discovery, you need to
prepare XRDS document, and set the location as XRDSLocation value.
See below.

  <Location /resource>
  PerlSetVar Mode PROTECTED_RESOURCE
  PerlSetVar XRDSLocation "http://myservice/discovery/xrdsdocument"
  PerlResponseHandler MyServiceWithOAuth
  </Location>


Then you can get this url in your script.

  sub service {
    my $self = shift;
    my $xrds_location = $self->xrds_location;
  }

But normalry all you have to do is write location on httpd.conf.
And "errout" method automatically put it into response header properly.

=head2 build_xrds

In case client send request which includes application/xrds+xml in Accept header,
if The server is set XRDSLocation as above, return resuponse with it in header.
But you can also directly return XRDS-Document.

Override build_xrds document.

  sub build_xrds {
    my $self = shift;
    my $xrds = q{
      <?xml version="1.0" encoding="UTF-8"?>
      <XRDS xmlns="xri://$xrds">
      ...
      </XRDS>
    };
    return $xrds;
  }

If the server doesn't support both XRDSLocation and build_xrds overriding,
The server doesn't support OAuth Discovery.

=head2 accepts_consumer_request

You can adopt OAuth Consumer Request 1.0.

See http://oauth.googlecode.com/svn/spec/ext/consumer_request/1.0/drafts/1/spec.html

To adopt this spec, you have to set var 'AcceptConsumerRequest' on httpd.conf

	<Location /resource>
	PerlSetVar Mode PROTECTED_RESOURCE
	PerlSetVar AcceptConsumerRequest 1
	PerlResponseHandler MyServiceWithOAuth
	</Location>

Then override service method for protected resource.

	sub service {
		my $self = shift;
        my $params = $self->{params};

		my $resource_owner_id;

		if (exists $params->{oauth_token}) {

			my $access_token_value = $params->{oauth_token};
			$resource_owner_id = $self->get_user_id_of_access_token($access_token_value);

		} else {

			my $consumer_key = $params->{oauth_consumer_key};
			$resource_owner_id = $self->get_user_id_of_consumer_developer($consumer_key);

		}

		my @resources = MyDB::Scheme->resultset('SomeResource')->search({
				user_id => $resource_owner_id,	
		});

		# output resource data in the manner your api defines.
		...

		return Apache2::Const::OK;

	}

=head2 accepts_reverse_phone_home

You can adopt OpenSocial Reverse Phone Home.

	<Location /resource>
	PerlSetVar Mode PROTECTED_RESOURCE
	PerlSetVar AcceptReversePhoneHome 1
	PerlResponseHandler MyServiceWithOAuth
	</Location>

=head2 error

L<Class::ErrorHandler> method.
In some check-method, when you find invalid request value,
call this method with error message and return it.

    sub check_nonce_and_timestamp {
        my ($self, $consumer_key, $nonce, $timestamp) = @_;
        if ($timestamp ...) {
            return $self->error(q{Invalid timestamp});
        }
        return 1;
    }

=head2 errstr

L<Class::ErrorHandler> method.
You can get error message that you set with error method.

    my $valid = $self->check_nonce_and_timestamp($consumer_key, $nonce, $timestamp);
    if (!$valid) {
        return $self->errout(401, $self->errstr);
    }

=head2 output(%params)

Simply output response.
You can set 3 params, code, type and content.

    return $self->output(
        code    => 200,
        type    => 'text/plain; charset=utf-8'
        content => 'success',
    );


=head2 errout($code, $message)

Output error message. This returns Apache2::Const::OK,
so, don't forget 'return';

    return $self->errout(400, q{Bad request});

And you can override this and put some function into this process.
For example, logging.

    sub errout {
        my ($self, $code, $message) = @_;
        $self->my_log_process($code, $message);
        return $self->SUPER::errout($code, $message);
    }

    sub my_log_process {
        my ($self, $code, $message) = @_;
        warn ...
    }

=cut

sub handler : method {
    my $class = shift;
    my $server = $class->new(@_);
    return $server->__service();
}

sub new {
    my $class = shift;
    my $r = shift;
    my $self = bless {
        request => $r,
        oauth => OAuth::Lite::ServerUtil->new( strict => 0 ),
        realm => undef,
        secure => 0,
        mode => PROTECTED_RESOURCE,
		accepts_consumer_request => 0,
		accepts_reverse_phone_home => 0,
        params => {},
        completed_validation => 0,
    }, $class;
    my $realm = $self->request->dir_config('Realm');
    $self->{realm} = $realm if $realm;
    my $accept_cr = $self->request->dir_config('AcceptConsumerRequest');
    $self->{accepts_consumer_request} = 1 if $accept_cr;
    my $accept_rp = $self->request->dir_config('AcceptReversePhoneHome');
    if ($accept_rp) {
        $self->{accepts_reverse_phone_home} = 1;
        $self->allow_extra_param('xoauth_requestor_id');
    }
    my $mode = $self->request->dir_config('Mode');
    my @valid_modes = (PROTECTED_RESOURCE, REQUEST_TOKEN, ACCESS_TOKEN);
    if ($mode) {
        if (none { $mode eq $_ } @valid_modes) {
            die "Invalid mode."; 
        } else {
            $self->{mode} = $mode;
        }
    }
    my $xrds_location = $self->request->dir_config('XRDSLocation');
    $self->{xrds_location} = $xrds_location if $xrds_location;
    if ( ($INC{'Apache2/ModSSL.pm'} && $r->connection->is_https)
      || ($r->subprocess_env('HTTPS') && $r->subprocess_env('HTTPS') eq 'ON') ) {
        $self->{secure} = 1;
    }
    $self->init(@_);
    $self;
}

sub init {
    my ($self, %args) = @_;
}

sub request_body {
    my $self = shift;
    unless (defined $self->{_request_body}) {
        my $length = $self->request->headers_in->{'Content-Length'} || 0;
        my $body = "";
        while ($length) {
            $self->request->read( my $buffer, ($length < 8192) ? $length : 8192 );
            $length -= bytes::length($buffer);
            $body .= $buffer;
        }
        $self->{_request_body} = $body;
    }
    $self->{_request_body};
}

sub request_uri {
    my $self = shift;
    return $self->request->uri;
}

sub request_method {
    my $self = shift;
    unless (defined $self->{_request_method}) {
        my $method = uc($self->request->method);
        my $x_method = uc($self->request->headers_in->{'X-HTTP-Method-Override'} || '');
        if ($method eq 'POST' && ($x_method eq 'PUT' || $x_method eq 'DELETE')) {
            $self->{_request_method} = $x_method;
        } else {
            $self->{_request_method} = $method;
        }
    }
    return $self->{_request_method};
}

sub __service {

    my $self = shift;

    my $realm;
    my $params = {};
    my $is_authorized = 0;

    my $needs_to_check_token = ( $self->is_required_request_token
        || (   $self->is_required_protected_resource
            && ($self->accepts_consumer_request || $self->accepts_reverse_phone_home) ) )
        ? 0
        : 1;

    my $authorization = $self->request->headers_in->{Authorization};
    if ($authorization && $authorization =~ /^\s*OAuth/) {
        ($realm, $params) = parse_auth_header($authorization);
    }

    if ( $self->request_method() eq 'POST'
          &&  $self->request->headers_in->{'Content-Type'} =~ m!application/x-www-form-urlencoded!) {
        for my $pair (split /&/, $self->request_body) {
            my ($key, $value) = split /=/, $pair;
            $params->{$key} = decode_param($value);
        }
    }

    for my $pair (split /&/, $self->request->args) {
        my ($key, $value) = split /=/, $pair;
        $params->{$key} = decode_param($value);
    }

    unless ($self->oauth->validate_params($params, $needs_to_check_token)) {
        return $self->errout(400, $self->oauth->errstr);
    }

    $self->{params} = $params;

    my $consumer_key = $params->{oauth_consumer_key};
    my $timestamp    = $params->{oauth_timestamp};
    my $nonce        = $params->{oauth_nonce};

    my $consumer_secret = $self->get_consumer_secret($consumer_key);
    unless (defined $consumer_secret) {
        return $self->errout(401, $self->errstr||CONSUMER_KEY_UNKNOWN);
    }

    $self->check_nonce_and_timestamp($consumer_key, $nonce, $timestamp)
        or return $self->errout(400, $self->errstr||TIMESTAMP_REFUSED);

    my $request_uri = $self->__build_request_uri();

    if ($self->is_required_request_token) {

        $self->oauth->verify_signature(
            method          => $self->request_method(),
            params          => $params,
            url             => $request_uri,
            consumer_secret => $consumer_secret,
        ) or return $self->errout(401, $self->oauth->errstr||SIGNATURE_INVALID);

        my $callback_url = $params->{oauth_callback};
        my $request_token = $self->publish_request_token($consumer_key, $callback_url)
            or return $self->errout(401, $self->errstr);
        return $self->__output_token($request_token);

    } elsif ($self->is_required_access_token) {

        my $token_value = $params->{oauth_token};
        my $token_secret = $self->get_request_token_secret($token_value);
        unless (defined $token_secret) {
            return $self->errout(401, $self->errstr||TOKEN_REJECTED);
        }
        $self->oauth->verify_signature(
            method          => $self->request_method(),
            params          => $params,
            url             => $request_uri,
            consumer_secret => $consumer_secret || '',
            token_secret    => $token_secret || '',
        ) or return $self->errout(401, $self->oauth->errstr||SIGNATURE_INVALID);
        my $verifier = $params->{oauth_verifier} || '';
        my $access_token = $self->publish_access_token($consumer_key, $token_value, $verifier)
            or return $self->errout(401, $self->errstr);
        return $self->__output_token($access_token);

    } else {

        my $token_secret = '';
        if (exists $params->{oauth_token}) {
            my $token_value = $params->{oauth_token};
            $token_secret = $self->get_access_token_secret($token_value, $consumer_key);
            unless (defined $token_secret) {
                return $self->errout(401, $self->errstr||TOKEN_REJECTED);
            }
        }

        $self->oauth->verify_signature(
            method          => $self->request_method(),
            params          => $params,
            url             => $request_uri,
            consumer_secret => $consumer_secret || '',
            token_secret    => $token_secret,
        ) or return $self->errout(401, $self->oauth->errstr||SIGNATURE_INVALID);

        if ($self->is_reverse_phone_home) {
            $self->verify_requestor_approval($consumer_key, $params->{xoauth_requestor_id})
                or return $self->errout(401, q{No approval});
        }

        $self->{completed_validation} = 1;

        return $self->service($params);
    }
}

sub __build_request_uri {

    # borrowed from Catalyst::Engine::Apache
    my $self = shift;

    my $scheme = $self->{secure} ? 'https' : 'http';
    my $host   = $self->request->hostname || 'localhost';
    my $port   = $self->request->get_server_port;

    PROXY_CHECK: {
        last PROXY_CHECK unless $self->request->headers_in->{'X-Forwarded-Host'};
        
        $host = $self->request->headers_in->{'X-Forwarded-Host'};

        if ( $host =~ /^(.+):(\d+)$/ ) {
            $host = $1;
            $port = $2;
        } else {
            $port = $self->{secure} ? 443 : 80;
        }
    }

    my $base_path = '';

    my $location = $self->request->location;
    if ( $location && $location ne '/' ) {
        $base_path = $location;
    }
   
    if ($host !~ /:/ ) {
        $host .= ":$port";
    }

    my ($path) = split /\?/, $self->request->unparsed_uri;

    if ( $base_path =~ m/[$URI::uric]/o ) {
        my $match = qr/($base_path)/;
        my ($base_match) = $path =~ $match;
        $base_path = $base_match;
    }

    $path =~ s{^/+}{};
    $base_path = '/' unless length $base_path;
    if ( $base_path ne '/' && $base_path ne $path && $base_path =~ m{/$path} ) {
        $path = $base_path;
        $path =~ s{^/+}{};
    }
    my $request_uri  = $scheme . '://' . $host . '/' . $path;
    return $request_uri;

}

sub __output_token {
    my ($self, $token) = @_;
    my $token_string = $token->as_encoded;
    return $self->output(
        code    => 200,
        type    => q{text/plain; charset=utf-8},
        content => $token_string,
    );
}

sub is_consumer_request {
    my $self = shift;
    return ($self->is_required_protected_resource
        && $self->accepts_consumer_request
        && !exists $self->{params}{oauth_token}) ? 1 : 0;
}

sub is_required_request_token {
    my $self = shift;
    return ($self->{mode} eq REQUEST_TOKEN) ? 1 : 0;
}

sub is_required_access_token {
    my $self = shift;
    return ($self->{mode} eq ACCESS_TOKEN) ? 1 : 0;
}

sub is_required_protected_resource {
    my $self = shift;
    return ($self->{mode} eq PROTECTED_RESOURCE) ? 1 : 0;
}

sub accepts_consumer_request {
    my $self = shift;
    return $self->{accepts_consumer_request};
}

sub accepts_reverse_phone_home {
    my $self = shift;
    return $self->{accepts_reverse_phone_home};
}

sub service {
    my ($self, $params) = @_;
}

sub get_request_token_secret {
    my ($self, $token) = @_;
    my $secret;
    return $secret;
}

sub get_access_token_secret {
    my ($self, $token, $consumer_key) = @_;
    my $secret;
    return $secret;
}

sub get_consumer_secret {
    my ($self, $consumer_key);
    my $consumer_secret;
    return $consumer_secret;
}

sub publish_request_token {
    my ($self, $consumer_key, $callback_url) = @_;
    my $token = OAuth::Lite::Token->new;
    return $token;
}

sub publish_access_token {
    my ($self, $request_token_string, $verifier) = @_;
    # validate request token
    # and publish access token
    # return $token;
    my $token = OAuth::Lite::Token->new;
    return $token;
}

sub check_nonce_and_timestamp {
    my ($self, $consumer_key, $timestamp, $nonce) = @_;
    return $self->error(q{Invalid Consumer});
    return $self->error(q{Invalid Timestamp});
    return $self->error(q{Invalid Nonce});
    return 1;
}

sub set_authenticate_header {
    my $self = shift;
    my $problem = shift;
    my %params = ();
    $params{realm} = $self->realm if $self->realm;
    $params{oauth_problem} = $problem if $problem;
    my $header = "OAuth " . join(", ", map sprintf(q{%s="%s"}, $_, $params{$_}), keys %params);
    $self->request->err_headers_out->add('WWW-Authenticate', $header);
}

sub _check_if_request_accepts_xrds {
    my $self = shift;
    unless (defined $self->{__request_accepts_xrds}) {
        my $accept = $self->request->headers_in->{Accept} || '';
        my @types = map { (split ";", $_)[0] } split /\*s,\*s/, $accept;
        if (any { $_ eq q{application/xrds+xml} } @types) {
            $self->{__request_accepts_xrds} = 1;
        } else {
            $self->{__request_accepts_xrds} = 0;
        }
    }
    return $self->{__request_accepts_xrds};
}

sub build_xrds {
  my $self = shift;
  return;
}

sub errout {
    my ($self, $code, $message, $description) = @_;

    if ( ( $self->request_method() eq 'GET'
        || $self->request_method() eq 'HEAD') && 
        $self->is_required_protected_resource &&
        $self->_check_if_request_accepts_xrds ) {
        if ($self->xrds_locaton) {
            $self->request->status(200);
            $self->request->err_headers_out->add('X-XRDS-Location' => $self->xrds_location);
            return Apache2::Const::OK;
        } elsif ($self->request_method() eq 'GET' && 
            (my $xrds = $self->build_xrds())) {
            return $self->output(
                code    => 200,
                type    => q{application/xrds+xml},
                content => $xrds,
            );
        }
    }
    my $problem;
    if (OAuth::Lite::Problems->match($message)) {
       $problem = $message; 
       $message = sprintf(q{oauth_problem=%s}, $message);
    }

    $self->set_authenticate_header($problem);
    return $self->output(
        code    => $code, 
        type    => q{text/plain; charset=utf-8},
        content => $message,
    );
}

sub output {
    my $self = shift;
    my %args = @_;
    my $code = $args{code} || 200;
    my $type = $args{type} || q{text/plain; charset=utf-8};
    my $content = $args{content} || '';
    $self->request->status($code);
    if ($content) {
        $self->request->content_type($type);
        $self->request->set_content_length(bytes::length($content));
        $self->request->print($content);
    }
    return Apache2::Const::OK;
}

sub allow_extra_param {
    my $self = shift;
    $self->oauth->allow_extra_param(@_);
}

sub allow_extra_params {
    my $self = shift;
    $self->oauth->allow_extra_params(@_);
}

sub support_signature_method {
    my $self = shift;
    $self->oauth->support_signature_method(@_);
}

sub support_signature_methods {
    my $self = shift;
    $self->oauth->support_signature_methods(@_);
}

sub is_reverse_phone_home {
    my $self = shift;
    return ( $self->is_required_protected_resource
        && $self->accepts_reverse_phone_home
        && !exists $self->{params}{oauth_token}
        && exists $self->{params}{xoauth_requestor_id}) ? 1 : 0
}

sub verify_requestor_approval {
    my $self = shift;
    my ($consumer_key, $user_id) = @_;
    return 1;
}

=head1 SEE ALSO

L<OAuth::Lite::ServerUtil>
L<OAuth::Lite::Server::Test::Echo>

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;

