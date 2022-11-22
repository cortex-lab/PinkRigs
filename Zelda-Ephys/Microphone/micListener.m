%% udp testing script

% warning('off', 'MATLAB:nargchk:deprecated')
% warning('off', 'MATLAB:Axes:UpVector');

%%

uMic = udp('0.0.0.0', 1111, 'LocalPort', 1002);
set(uMic, 'DatagramReceivedFcn', @micUDPCallback);
% set(u, 'DatagramReceivedFcn', 'start(t);');
fopen(uMic);
% echoudp('off');

% fclose(u);
% delete(u);

% fclose(u); fopen(u);