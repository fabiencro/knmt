package SockClient;

#
# use SockClient;
#
# my $rep = SockClient::reply($host, $port, $cmd);
#

use utf8;
use strict;
use Encode;
use FileHandle;
use Socket;

binmode STDIN, ":encoding(utf8)";
binmode STDOUT, ":encoding(utf8)";
binmode STDERR, ":encoding(utf8)";

sub reply
{
    my ($host, $port, $cmd) = @_;

    my $req_str;
    $req_str .= $cmd;
    $req_str .= "--END--";

    my $iaddr = inet_aton($host) or return ""; # die "$host: No host!";
    my $paddr = sockaddr_in($port, $iaddr);

    local *SOCK;
    my $proto = getprotobyname('tcp');
    if (socket(SOCK, PF_INET, SOCK_STREAM, $proto) && connect(SOCK, $paddr)){
        SOCK->autoflush(1);
        binmode SOCK, ':encoding(utf8)';
        printf(SOCK "%s", $req_str);

        my $buf;
        while(my $line = <SOCK>) {
            if($line =~ /--END--/) {
                last;
            }
            $buf .= $line;
        }
        close(SOCK);
        return $buf;
    }
    
    print STDERR "Cannot connect $host($port).";
    return "";
}

1;
