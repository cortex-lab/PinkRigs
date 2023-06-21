function open_all_tl()
paths = dat.paths;
FileObj = java.io.File(paths.localRepository);
if FileObj.getUsableSpace<150e9
%     error(['There is not enough space. Please clear out: ' paths.localRepository])
else
%     continue;%fprintf('There is %d GB of space, continuing experiment...\n', round(FileObj.getUsableSpace/1e9));
end
%%

rig = hostname;
if contains(rig,'zelda')
    % open frontcam
    eval('!matlab -nodesktop -nosplash -r "my_frontcam" &')
    
    % open sidecam
    eval('!matlab -nodesktop -nosplash -r "my_sidecam" &')
    
    % open eyecam
    eval('!matlab -nodesktop -nosplash -r "my_eyecam" &')
elseif contains(rig,'poppy')
    % open topcam
    eval('!matlab -nodesktop -nosplash -r "my_topcam" &')
else
    warning('Not sure what the cameras are for this rig...')
    return
end

tl.mpepListenerWithWS
end 