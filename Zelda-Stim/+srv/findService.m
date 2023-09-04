function s = findService(id, varargin)
%SRV.FINDSERVICE Returns experiment service(s) with specified id(s)
%   This and EXP.BASICSERVICES has been replaced by SRV.LOADSERVICE. See
%   also SRV.SERVICE, SRV.LOADSERVICE, SRV.BASICSERVICES.

% Part of Rigbox

% 2013-06 CB created

rig = hostname;
rig = lower(rig);

if strcmp(rig,'poppy-stim')
    timelineHost = iff(any(strcmp(id, 'timeline')), {'poppy-timeline'}, {''});
    % micHost = iff(any(strcmp(id, 'microphone')), {'poppy-ephys'}, {''});
    micHost = iff(any(strcmp(id, 'microphone')), {'128.40.198.112'}, {''});
    topCamHost = iff(any(strcmp(id, 'topCam')), {'poppy-timeline'}, {''});
    
    timelinePort = 1001;
    micPort = 1002;
    topCamPort = 1003;
    
    remoteHosts = [timelineHost micHost topCamHost];
    remotePorts = {timelinePort micPort topCamPort};
else
    rigTime = regexprep(rig,'stim','time');
    rigEphys = regexprep(rig,'stim','ephys');
    
    % Checking which hosts to use, strings should be as they are
    % known to MC (defined in Rigging/config/remote.mat)
    timelineHost = iff(any(strcmp(id, 'timeline')), {rigTime}, {''});
    micHost = iff(any(strcmp(id, 'microphone')), {rigEphys}, {''});
    eyeCamHost = iff(any(strcmp(id, 'eyeCam')), {rigTime}, {''});
    frontCamHost = iff(any(strcmp(id, 'frontCam')), {rigTime}, {''});
    sideCamHost = iff(any(strcmp(id, 'sideCam')), {rigTime}, {''});
    timelinePort = 1001;
    micPort = 1002;
    eyePort = 1003;
    frontPort = 1004;
    sidePort = 1005;
    
    remoteHosts = [timelineHost, micHost, eyeCamHost, frontCamHost, sideCamHost];
    remotePorts = {timelinePort, micPort, eyePort, frontPort, sidePort};
end

emp = cellfun(@isempty, remoteHosts);

MpepHosts = io.MpepUDPDataHosts(remoteHosts(~emp));
MpepHosts.ResponseTimeout = 60;
MpepHosts.Id = 'MPEP-Hosts';
MpepHosts.Title = 'mPep Data Acquisition Hosts'; % name displayed on startup
MpepHosts.RemotePorts = remotePorts(~emp);
MpepHosts.open();
s = {MpepHosts};
end

