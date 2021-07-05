function s = findService(id, varargin)
%SRV.FINDSERVICE Returns experiment service(s) with specified id(s)
%   This and EXP.BASICSERVICES has been replaced by SRV.LOADSERVICE. See
%   also SRV.SERVICE, SRV.LOADSERVICE, SRV.BASICSERVICES.
%sr
% Part of Rigbox

% 2013-06 CB created

% Checking which hosts to use, strings should be as they are 
% known to MC (defined in Rigging/config/remote.mat)
timelineHost = iff(any(strcmp(id, 'timeline')), {'zelda-time4'}, {''});
micHost = iff(any(strcmp(id, 'microphone')), {'zelda-ephys4'}, {''});
eyeCamHost = iff(any(strcmp(id, 'eyeCam')), {'zelda-time4'}, {''});
frontCamHost = iff(any(strcmp(id, 'frontCam')), {'zelda-time4'}, {''});
sideCamHost = iff(any(strcmp(id, 'sideCam')), {'zelda-time4'}, {''});
timelinePort = 1001;
micPort = 1002;
eyePort = 1003;
frontPort = 1004;
sidePort = 1005;

remoteHosts = [timelineHost, micHost, eyeCamHost, frontCamHost, sideCamHost];
remotePorts = {timelinePort, micPort, eyePort, frontPort, sidePort};

emp = cellfun(@isempty, remoteHosts);
% 
MpepHosts = io.MpepUDPDataHosts(remoteHosts(~emp));
MpepHosts.ResponseTimeout = 60;
MpepHosts.Id = 'MPEP-Hosts';
MpepHosts.Title = 'mPep Data Acquisition Hosts'; % name displayed on startup
MpepHosts.RemotePorts = remotePorts(~emp);
MpepHosts.open();
s = {MpepHosts};
end

