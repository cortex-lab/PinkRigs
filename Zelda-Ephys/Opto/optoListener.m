%% udp testing script

% warning('off', 'MATLAB:nargchk:deprecated')
% warning('off', 'MATLAB:Axes:UpVector');

%%

uOpto = udp('0.0.0.0', 1111, 'LocalPort', 1006);
set(uOpto, 'DatagramReceivedFcn', @optoUDPCallback);
% set(u, 'DatagramReceivedFcn', 'start(t);');
fopen(uOpto);
% echoudp('off');

% fclose(u);
% delete(u);

% fclose(u); fopen(u);