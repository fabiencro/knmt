package OAuth::Lite::Consumer;

use strict;
use warnings;

use base qw(
    Class::ErrorHandler
    Class::Accessor::Fast
);

__PACKAGE__->mk_accessors(qw(
    consumer_key
    consumer_secret
    oauth_request
    oauth_response
    request_token
    access_token
));

*oauth_req = \&oauth_request;
*oauth_res = \&oauth_response;

use Carp ();
use bytes ();
use URI;
use HTTP::Request;
use HTTP::Headers;
use UNIVERSAL::require;
use List::MoreUtils qw(any);

use OAuth::Lite;
use OAuth::Lite::Agent;
use OAuth::Lite::Token;
use OAuth::Lite::Response;
use OAuth::Lite::Util qw(:all);
use OAuth::Lite::AuthMethod qw(:all);

=head1 NAME

OAuth::Lite::Consumer - consumer agent

=head1 SYNOPSIS

    my $consumer = OAuth::Lite::Consumer->new(
        consumer_key       => $consumer_key,
        consumer_secret    => $consumer_secret,
        site               => q{http://api.example.org},
        request_token_path => q{/request_token},
        access_token_path  => q{/access_token},
        authorize_path     => q{http://example.org/authorize},
    );

    # At first you have to publish request-token, and
    # with it, redirect end-user to authorization-url that Service Provider tell you beforehand.

    my $request_token = $consumer->get_request_token(
        callback_url => q{http://yourservice/callback},
    );

    $your_app->session->set( request_token => $request_token );

    $your_app->redirect( $consumer->url_to_authorize(
        token        => $request_token,
    ) );

    # After user authorize the request on a Service Provider side web application.

    my $verifier = $your_app->request->param('oauth_verifier');
    my $request_token = $your_app->session->get('request_token');

    my $access_token = $consumer->get_access_token(
        token    => $request_token,
        verifier => $verifier,
    );

    $your_app->session->set( access_token => $access_token );
    $your_app->session->remove('request_token');

    # After all, you can request protected-resources with access token

    my $access_token = $your_app->session->get('access_token');

    my $res = $consumer->request(
        method => 'GET',
        url    => q{http://api.example.org/picture},
        token  => $access_token,
        params => { file => 'mypic.jpg', size => 'small' },
    );

    unless ($res->is_success) {
        if ($res->status == 400 || $res->status == 401) {
            my $auth_header = $res->header('WWW-Authenticate');
            if ($auth_header && $auth_header =~ /^OAuth/) {
                # access token may be expired,
                # get request-token and authorize again
            } else {
                # another auth error.
            }
        }
        # another error.
    }

    # OAuth::Lite::Agent automatically adds Accept-Encoding gzip header to
    # request, so, when you use default agent, call decoded_content.
    my $resource = $res->decoded_content || $res->content;

    $your_app->handle_resource($resource);


=head1 DESCRIPTION

This module helps you to build OAuth Consumer application.

=head1 PAY ATTENTION

If you use OAuth 1.31 or older version, its has invalid way to normalize params.
(when there are two or more same key and they contain ASCII and non ASCII value)

But the many services have already supported deprecated version, 
and the correct way breaks backward compatibility.
So, from 1.32, supported both correct and deprecated method. 

use $OAuth::Lite::USE_DEPRECATED_NORMALIZER to switch behaviour.
Currently 1 is set by default to keep backward compatibility.

    use OAuth::Lite::Consumer;
    use OAuth::Lite;

    $OAuth::Lite::USE_DEPRECATED_NORMALIZER = 0;
    ...

=head1 METHODS

=head2 new(%args)

=head3 parameters

=over 4

=item consumer_key

consumer_key value

=item consumer_secret 

consumer_secret value

=item signature_method

Signature method you can choose from 'HMAC-SHA1', 'PLAINTEXT', and 'RSA-SHA1' (optional, 'HMAC-SHA1' is set by default)

=item http_method

HTTP method (GET or POST) when the request is for request token or access token. (optional, 'POST' is set by default)

=item auth_method

L<OAuth::Lite::AuthMethod>'s value you can choose from AUTH_HEADER, POST_BODY and URL_QUERY (optional, AUTH_HEADER is set by default)

=item realm

The OAuth realm value for a protected-resource you wanto to access to. (optional. empty-string is set by default)

=item use_request_body_hash

If you use Request Body Hash extension, set 1.
See Also L<http://oauth.googlecode.com/svn/spec/ext/body_hash/1.0/drafts/4/spec.html>

=item site 

The base site url of Service Provider

=item request_token_path

=item access_token_path

=item authorize_path

=item callback_url

=back

Site and other paths, simple usage.

    my $consumer = OAuth::Lite::Consumer->new(
        ...
        site => q{http://example.org},
        request_token_path => q{/request_token},
        access_token_path  => q{/access_token},
        authorize_path     => q{/authorize},
    );

    say $consumer->request_token_url; # http://example.org/request_token
    say $consumer->access_token_url;  # http://example.org/access_token
    say $consumer->authorization_url; # http://example.org/authorize

If the authorization_url is run under another domain, for example.

    my $consumer = OAuth::Lite::Consumer->new(
        ...
        site => q{http://api.example.org}, 
        request_token_path => q{/request_token},
        access_token_path  => q{/access_token},
        authorize_path     => q{http://www.example.org/authorize},
    );
    say $consumer->request_token_url; # http://api.example.org/request_token
    say $consumer->access_token_url;  # http://api.example.org/access_token
    say $consumer->authorization_url; # http://www.example.org/authorize

Like this, if you pass absolute url, consumer uses them as it is.

You can omit site param, if you pass all paths as absolute url.

    my $consumer = OAuth::Lite::Consumer->new(
        ...
        request_token_path => q{http://api.example.org/request_token},
        access_token_path  => q{http://api.example.org/access_token},
        authorize_path     => q{http://www.example.org/authorize},
    );


And there is a flexible way.

    # don't set each paths here.
    my $consumer = OAuth::Lite::Consumer->new(
        consumer_key    => $consumer_key,
        consumer_secret => $consumer_secret,
    );

    # set request token url here directly
    my $rtoken = $consumer->get_request_token(
        url          => q{http://api.example.org/request_token},
        callback_url => q{http://www.yourservice/callback},
    );

    # set authorize path here directly
    my $url = $consumer->url_to_authorize(
        token        => $rtoken,
        url          => q{http://www.example.org/authorize},
    );

    # set access token url here directly
    my $atoken = $consumer->get_access_token(
        url      => q{http://api.example.org/access_token},
        verifier => $verfication_code,
    );

So does callback_url. You can set it on consutructor or get_request_token method directly.

    my $consumer = OAuth::Lite::Consumer->new(
        ...
        callback_url => q{http://www.yourservice/callback},
    );
    ...
    my $url = $consumer->get_request_token();

Or

    my $consumer = OAuth::Lite::Consumer->new(
        ...
    );
    ...
    my $url = $consumer->get_request_token(
        callback_url => q{http://www.yourservice/callback},
    );

=cut

sub new {
    my ($class, %args) = @_;
    my $ua = delete $args{ua};
    unless ($ua) {
        $ua = OAuth::Lite::Agent->new;
        $ua->agent(join "/", __PACKAGE__, $OAuth::Lite::VERSION);
    }
    my $self = bless {
        ua => $ua,
    }, $class;
    $self->_init(%args);
    $self;
}

sub _init {
    my ($self, %args) = @_;

    my $signature_method_class = exists $args{signature_method}
        ? $args{signature_method}
        : 'HMAC_SHA1';
    $signature_method_class =~ s/-/_/g;
    $signature_method_class = join('::',
        'OAuth::Lite::SignatureMethod',
        $signature_method_class
    );
    $signature_method_class->require
        or Carp::croak(
            sprintf
                qq/Could't find signature method class, %s/,
                $signature_method_class
        );

    $self->{consumer_key} = $args{consumer_key} || '';
    $self->{consumer_secret} = $args{consumer_secret} || '';
    $self->{http_method} = $args{http_method} || 'POST';
    $self->{auth_method} = $args{auth_method} || AUTH_HEADER;
    unless ( OAuth::Lite::AuthMethod->validate_method( $self->{auth_method} ) ) {
        Carp::croak( sprintf
            qq/Invalid auth method "%s"./, $self->{auth_method} );
    }
    $self->{signature_method} = $signature_method_class;
    $self->{realm} = $args{realm};
    $self->{site} = $args{site};
    $self->{request_token_path} = $args{request_token_path};
    $self->{access_token_path} = $args{access_token_path};
    $self->{authorize_path} = $args{authorize_path};
    $self->{callback_url} = $args{callback_url};
    $self->{oauth_request} = undef;
    $self->{oauth_response} = undef;
    $self->{use_request_body_hash} = $args{use_request_body_hash} ? 1 : 0;
    $self->{_nonce} = $args{_nonce};
    $self->{_timestamp} = $args{_timestamp};
}

=head2 request_token_url

=cut

sub request_token_url {
    my $self = shift;
    $self->{request_token_path} =~ m!^http(?:s)?\://!
        ? $self->{request_token_path}
        : sprintf q{%s%s}, $self->{site}, $self->{request_token_path};
}


=head2 access_token_url

=cut

sub access_token_url {
    my $self = shift;
    $self->{access_token_path} =~ m!^http(?:s)?\://!
        ? $self->{access_token_path}
        : sprintf q{%s%s}, $self->{site}, $self->{access_token_path};
}

=head2 authorization_url

=cut

sub authorization_url {
    my $self = shift;
    $self->{authorize_path} =~ m!^http(?:s)?\://!
        ? $self->{authorize_path}
        : sprintf q{%s%s}, $self->{site}, $self->{authorize_path};
}


=head2 url_to_authorize(%params)

=head3 parameters

=over 4

=item url

authorization url, you can omit this if you set authorization_path on constructor.

=item token

request token value

=back

    my $url = $consumer->url_to_authorize(
        url          => q{http://example.org/authorize}, 
        token        => $request_token,
        callback_url => q{http://www.yousrservice/callback},
    );

=cut

sub url_to_authorize {
    my ($self, %args) = @_;
    $args{url} ||= $self->authorization_url;
    my $url = $args{url}
        or Carp::croak qq/url_to_authorize needs url./;
    my %params = ();
    if (defined $args{token}) {
        my $token = $args{token};
        $params{oauth_token} = ( eval { $token->isa('OAuth::Lite::Token') } )
            ? $token->token
            : $token;
    }
    $url = URI->new($url);
    $url->query_form(%params);
    $url->as_string;
}

=head2 obtain_request_token(%params)

Returns a request token as an L<OAuth::Lite::Response> object.
Except for that, this method behaves same as get_request_token.

=cut

sub obtain_request_token {
    my $self = shift;
    my $res = $self->_get_request_token(@_);
    unless ($res->is_success) {
        return $self->error($res->status_line);
    }
    my $resp = OAuth::Lite::Response->from_encoded($res->decoded_content||$res->content);
    return $self->error(qq/oauth_callback_confirmed is not true/)
        unless $resp && $resp->token && $resp->token->callback_confirmed;
    $self->request_token($resp->token);
    $resp;
}

=head2 get_request_token(%params)

Returns a request token as an L<OAuth::Lite::Token> object.

=head3 parameters

=over 4

=item url

Request token url. You can omit this if you set request_token_path on constructor

=item realm

Realm for the resource you want to access to.
You can omit this if you set realm on constructor.

=item callback_url

Url which service provider redirect end-user to after authorization.
You can omit this if you set callback_url on constructor.

=back

    my $token = $consumer->get_request_token(
        url   => q{http://api.example.org/request_token},
        realm => q{http://api.example.org/picture},
    ) or die $consumer->errstr;

    say $token->token;
    say $token->secret;

=cut

sub get_request_token {
    my $self = shift;
    my $res = $self->_get_request_token(@_);
    unless ($res->is_success) {
        return $self->error($res->status_line);
    }
    my $token = OAuth::Lite::Token->from_encoded($res->decoded_content||$res->content);
    return $self->error(qq/oauth_callback_confirmed is not true/)
        unless $token && $token->callback_confirmed;
    $self->request_token($token);
    $token;
}

sub _get_request_token {
    my ($self, %args) = @_;
    $args{url} ||= $self->request_token_url;
    my $request_token_url = delete $args{url}
        or Carp::croak qq/get_request_token needs url in hash params
            or set request_token_path on constructor./;
    my $realm = delete $args{realm} || $self->{realm} || '';
    my $callback = delete $args{callback_url} || $self->{callback_url} || 'oob';
    my $res = $self->__request(
        realm  => $realm,
        url    => $request_token_url,
        params => {%args, oauth_callback => $callback},
    );
    return $res;
}

=head2 obtain_access_token

    my $res = $consumer->obtain_access_token(
        url    => $access_token_url,
        params => {
            x_auth_username => "myname",
            x_auth_password => "mypass",
            x_auth_mode     => "client_auth",
        },
    );

    my $access_token = $res->token;
    say $acces_token->token;
    say $acces_token->secret;
    my $expires = $res->param('x_auth_expires');

What is the difference between obtain_access_token and get_access_token?
get_access_token requires token and verifier.
obtain_access_token doesn't. these parameters are optional.
You can pass extra parameters like above example.(see x_auth_XXX parameters)
And get_access_token returns OAuth::Lite::Token object directly,
obtain_access_token returns OAuth::Lite::Response object that includes
not only Token object but also other response parameters.
the extra parameters are used for some extensions.(Session extension, xAuth, etc.)

Of cource, if you don't need to handle these extensions,
You can continue to use get_access_token for backward compatibility.

    my $token = $consumer->get_access_token(
        url      => $access_token_url,
        token    => $request_token,
        verifier => $verification_code,
    );

    # above code's behavior is same as (but response objects are different)

    my $res = $consumer->obtain_access_token(
        url => $access_token_url,
        token => $request_token,
        params => {
            oauth_verifier => $verification_code, 
        },
    );

=cut

sub obtain_access_token {
    my ($self, %args) = @_;
    $args{url} ||= $self->access_token_url;
    my $access_token_url = $args{url}
        or Carp::croak qq/get_access_token needs access_token_url./;
    my $realm = $args{realm} || $self->{realm} || '';

    my $token = defined $args{token} ? $args{token} : undef;
    my $params = $args{params} || {};

    my $res = $self->__request(
        realm  => $realm,
        url    => $access_token_url,
        token  => $token,
        params => $params,
    );
    unless ($res->is_success) {
        return $self->error($res->status_line);
    }
    my $resp = OAuth::Lite::Response->from_encoded($res->decoded_content||$res->content);
    $self->access_token($resp->token);
    $resp;
}


=head2 get_access_token(%params)

Returns a access token as an L<OAuth::Lite::Token> object.

=head3 parameters

=over 4

=item url

Request token url. You can omit this if you set request_token_path on constructor

=item realm

Realm for the resource you want to access to.
You can omit this if you set realm on constructor.

=item token

Request token object.

=item verifier

Verfication code which provider returns.

=back

    my $token = $consumer->get_access_token(
        url      => q{http://api.example.org/request_token},
        realm    => q{http://api.example.org/picture},
        token    => $request_token,
        verifier => $verification_code,
    ) or die $consumer->errstr;

    say $token->token;
    say $token->secret;


=cut

sub get_access_token {
    my ($self, %args) = @_;
    $args{url} ||= $self->access_token_url;
    my $access_token_url = $args{url}
        or Carp::croak qq/get_access_token needs access_token_url./;
    my $token = defined $args{token} ? $args{token} : $self->request_token;
    Carp::croak qq/get_access_token needs token./ unless defined $token;
    my $realm = $args{realm} || $self->{realm} || '';
    my $verifier = $args{verifier}
        or Carp::croak qq/verfier not found./;
    my $res = $self->__request(
        realm => $realm,
        url   => $access_token_url,
        token => $token,
        params => { oauth_verifier => $verifier },
    );
    unless ($res->is_success) {
        return $self->error($res->status_line);
    }
    my $access_token = OAuth::Lite::Token->from_encoded($res->decoded_content||$res->content);
    $self->access_token($access_token);
    $access_token;
}

=head2 gen_oauth_request(%args)

Returns L<HTTP::Request> object.

    my $req = $consumer->gen_oauth_request(
        method  => 'GET', 
        url     => 'http://example.com/',
        headers => [ Accept => q{...}, 'Content-Type' => q{...}, ... ],
        content => $content,
        realm   => $realm,
        token   => $token,
        params  => { file => 'mypic.jpg', size => 'small' },
    );

=cut

sub gen_oauth_request {

    my ($self, %args) = @_;

    my $method  = $args{method} || $self->{http_method};
    my $url     = $args{url};
    my $content = $args{content};
    my $token   = $args{token};
    my $extra   = $args{params} || {};
    my $realm   = $args{realm}
                || $self->{realm}
                || $self->find_realm_from_last_response
                || '';

    if (ref $extra eq 'ARRAY') {
        my %hash;
        for (0...scalar(@$extra)/2-1) {
            my $key = $extra->[$_ * 2];
            my $value = $extra->[$_ * 2 + 1];
            $hash{$key} ||= [];
            push @{ $hash{$key} }, $value;
        }
        $extra = \%hash;
    }

    my $headers = $args{headers};
    if (defined $headers) {
        if (ref($headers) eq 'ARRAY') {
            $headers = HTTP::Headers->new(@$headers);
        } else {
            $headers = $headers->clone;
        }
    } else {
        $headers = HTTP::Headers->new;
    }

    my @send_data_methods = qw/POST PUT/;
    my @non_send_data_methods = qw/GET HEAD DELETE/;

    my $is_send_data_method = any { $method eq $_ } @send_data_methods;

    my $auth_method = $self->{auth_method};
    $auth_method = AUTH_HEADER
        if ( !$is_send_data_method && $auth_method eq POST_BODY );

    if ($auth_method eq URL_QUERY) {
        if ( $is_send_data_method && !$content ) {
            Carp::croak
                qq(You must set content-body in case you use combination of URL_QUERY and POST/PUT http method);
        } else {
            if ( $is_send_data_method ) {
                if ( my $hash = $self->build_body_hash($content) ) {
                    $extra->{oauth_body_hash} = $hash;
                }
            }
            my $query = $self->gen_auth_query($method, $url, $token, $extra);
            $url = sprintf q{%s?%s}, $url, $query;
        }
    } elsif ($auth_method eq POST_BODY) {
        my $query = $self->gen_auth_query($method, $url, $token, $extra);
        $content = $query;
        $headers->header('Content-Type', q{application/x-www-form-urlencoded});
    } else {
        my $origin_url = $url;
        my $copied_params = {};
        for my $param_key ( keys %$extra ) {
            next if $param_key =~ /^x?oauth_/;
            $copied_params->{$param_key} = $extra->{$param_key};
        }
        if ( keys %$copied_params > 0 ) {
            my $data = normalize_params($copied_params);
            if ( $is_send_data_method && !$content ) {
                $content = $data;
            } else {
                $url = sprintf q{%s?%s}, $url, $data;
            }
        }
        if ( $is_send_data_method ) {
            if ( my $hash = $self->build_body_hash($content) ) {
                $extra->{oauth_body_hash} = $hash;
            }
        }
        my $header = $self->gen_auth_header($method, $origin_url,
            { realm => $realm, token => $token, extra => $extra });
        $headers->header( Authorization => $header );
    }
    if ( $is_send_data_method ) {
        $headers->header('Content-Type', q{application/x-www-form-urlencoded})
            unless $headers->header('Content-Type');
        $headers->header('Content-Length', bytes::length($content) || 0 );
    }
    my $req = HTTP::Request->new( $method, $url, $headers, $content );
    $req;
}

=head2 request(%params)

Returns L<HTTP::Response> object.

=head3 parameters

=over 4

=item realm

Realm for a resource you want to access

=item token

Access token  L<OAuth::Lite::Token> object

=item method

HTTP method.

=item url

Request URL

=item parmas

Extra params.

=item content

body data sent when method is POST or PUT.

=back

    my $response = $consumer->request(
        method  => 'POST',
        url     => 'http://api.example.com/picture',
        headers => [ Accept => q{...}, 'Content-Type' => q{...}, ... ],
        content => $content,
        realm   => $realm,
        token   => $access_token,
        params  => { file => 'mypic.jpg', size => 'small' },
    );

    unless ($response->is_success) {
        ...
    }

=cut

sub request {
    my ($self, %args) = @_;
    $args{token} ||= $self->access_token;
    $self->__request(%args);
}

sub __request {
    my ($self, %args) = @_;
    my $req = $self->gen_oauth_request(%args);
    $self->oauth_clear();
    $self->oauth_request($req);
    my $res = $self->{ua}->request($req);
    $self->oauth_response($res);
    $res;
}

=head2 get

There are simple methods to request protected resources.
You need to obtain access token and set it to consumer beforehand.

    my $access_token = $consumer->get_access_token(
        token    => $request_token,
        verifier => $verifier,
    );
    # when successfully got an access-token,
    # it internally execute saving method like following line.
    # $consumer->access_token( $access_token )

or
    my $access_token = $your_app->pick_up_saved_access_token();
    $consumer->access_token($access_token);

Then you can access protected resource in a simple way.

    my $res = $consumer->get( 'http://api.example.com/pictures' );
    if ($res->is_success) {
        say $res->decoded_content||$res->content;
    }

This is same as

    my $res = $consumer->request(
        method => q{GET},
        url    => q{http://api.example.com/picture},
    );
    if ($res->is_success) {
        say $res->decoded_content||$res->content;
    }

=cut

sub get {
    my ( $self, $url, $args ) = @_;
    $args ||= {};
    $args->{method} = 'GET';
    $args->{url}    = $url;
    $self->request(%$args);
}

=head2 post

    $res = $consumer->post( 'http://api.example.com/pictures', $content );
    if ($res->is_success) {
        ...
    }

This is same as

    $res = $consumer->request(
        method  => q{POST},
        url     => q{http://api.example.com/picture},
        content => $content,
    );
    if ($res->is_success) {
        ...
    }


=cut

sub post {
    my ( $self, $url, $content, $args ) = @_;
    $args ||= {};
    $args->{method}  = 'POST';
    $args->{url}     = $url;
    $args->{content} = $content;
    $self->request(%$args);

}

=head2 put

    $res = $consumer->put( 'http://api.example.com/pictures', $content );
    if ($res->is_success) {
        ...
    }

This is same as

    my $res = $consumer->request(
        method  => q{PUT},
        url     => q{http://api.example.com/picture},
        content => $content,
    );
    if ($res->is_success) {
        ...
    }


=cut

sub put {
    my ( $self, $url, $content, $args ) = @_;
    $args ||= {};
    $args->{method}  = 'PUT';
    $args->{url}     = $url;
    $args->{content} = $content;
    $self->request(%$args);

}

=head2 delete

    my $res = $consumer->delete('http://api.example.com/delete');
    if ($res->is_success) {
        ...
    }

This is same as

    my $res = $consumer->request(
        method  => q{DELETE},
        url     => q{http://api.example.com/picture},
    );
    if ($res->is_success) {
        ...
    }

=cut

sub delete {
    my ( $self, $url, $args ) = @_;
    $args ||= {};
    $args->{method} = 'DELETE';
    $args->{url}    = $url;
    $self->request(%$args);
}

=head2 find_realm_from_last_response

=cut

sub find_realm_from_last_response {
    my $self = shift;
    return unless $self->oauth_response;
    my $authenticate = $self->oauth_response->header('WWW-Authenticate');
    return unless ($authenticate && $authenticate =~ /^\s*OAuth/);
    my $realm = parse_auth_header($authenticate);
    $realm;
}

=head2 gen_auth_header($http_method, $request_url, $params);

=head3 parameters

=over 4

=item realm

realm for a resource you want to access

=item token

OAuth::Lite::Token object(optional)

=back

    my $header = $consumer->gen_auth_header($method, $url, {
        realm => $realm,
        token => $token,
    });

=cut

sub gen_auth_header {
    my ($self, $method, $url, $args) = @_;
    my $extra = $args->{extra} || {};
    my $params = $self->gen_auth_params($method, $url, $args->{token}, $extra);
    my $realm = $args->{realm} || '';
    my $authorization_header = build_auth_header($realm, $params);
    $authorization_header;
}

=head2 gen_auth_query($http_method, $ruqest_url, $token, $extra)

=cut

sub gen_auth_query {
    my ($self, $method, $url, $token, $extra) = @_;
    $extra ||= {};
    my $params = $self->gen_auth_params($method, $url, $token, $extra);
    my %all = (%$extra, %$params);
    normalize_params({%all});
}

=head2 gen_auth_params($http_method, $request_url, [$token])

Generates and returns all oauth params.

    my $params = $consumer->gen_auth_params($http_method, $request_url);
    say $params->{oauth_consumer_key};
    say $params->{oauth_timestamp};
    say $params->{oauth_nonce};
    say $params->{oauth_signature_method};
    say $params->{oauth_signature};
    say $params->{oauth_version};

If you pass token as third argument, the result includes oauth_token value.

    my $params = $consumer->gen_auth_params($http_method, $request_url, $token);
    say $params->{oauth_consumer_key};
    say $params->{oauth_timestamp};
    say $params->{oauth_nonce};
    say $params->{oauth_signature_method};
    say $params->{oauth_signature};
    say $params->{oauth_token};
    say $params->{oauth_version};

=cut

sub gen_auth_params {
    my ($self, $method, $url, $token, $extra) = @_;
    my $params = {};
    $extra ||= {};
    $params->{oauth_consumer_key} = $self->consumer_key || '';
    $params->{oauth_timestamp} = $self->{_timestamp} || time();
    $params->{oauth_nonce} = $self->{_nonce} || gen_random_key();
    $params->{oauth_version} = $OAuth::Lite::OAUTH_DEFAULT_VERSION;
    my $token_secret = '';
    if (defined $token) {
        if (eval { $token->isa('OAuth::Lite::Token') }) {
            $params->{oauth_token} = $token->token;
            $token_secret = $token->secret;
        } else {
            $params->{oauth_token} = $token;
        }
    }
    my $consumer_secret = $self->consumer_secret || '';
    $params->{oauth_signature_method} = $self->{signature_method}->method_name;
    if ($params->{oauth_signature_method} eq 'PLAINTEXT' && lc($url) !~ /^https/) {
        warn qq(PLAINTEXT signature method should be used on SSL/TSL.);
    }
    $params = {%$params, %$extra};
    my $base = create_signature_base_string($method, $url, $params);
    $params->{oauth_signature} = $self->{signature_method}->new(
        consumer_secret => $consumer_secret,
        token_secret    => $token_secret,
    )->sign($base);
    $params;
}

=head2 oauth_request

=head2 oauth_req

Returns last oauth request.

    my $req_token = $consumer->get_request_token(...);
    say $consumer->oauth_request->uri;

    my $req_token = $consumer->get_access_token(...);
    say $consumer->oauth_request->uri;

=head2 oauth_response

=head2 oauth_res

Returns last oauth response.

    my $req_token = $consumer->get_request_token(...);
    say $consumer->oauth_response->status;

    my $req_token = $consumer->get_access_token(...);
    say $consumer->oauth_response->status;

=head2 oauth_clear

remove last oauth-request and oauth-response.

=cut

sub oauth_clear {
    my $self = shift;
    $self->{oauth_request}  = undef;
    $self->{oauth_response} = undef;
}

=head2 build_body_hash

Build body hash according to the spec for 'OAuth Request Body Hash extension'
http://oauth.googlecode.com/svn/spec/ext/body_hash/1.0/drafts/4/spec.html

    my $hash = $self->build_body_hash($content);

=cut

sub build_body_hash {
    my ( $self, $content ) = @_;
    if ( $self->{use_request_body_hash} ) {
        my $hash = $self->{signature_method}->build_body_hash($content);
        return $hash;
    }
    return;
}

=head1 AUTHOR

Lyo Kato, C<lyo.kato _at_ gmail.com>

=head1 COPYRIGHT AND LICENSE

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.6 or,
at your option, any later version of Perl 5 you may have available.

=cut

1;
