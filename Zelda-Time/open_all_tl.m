function open_all_tl()
paths = dat.paths;
FileObj = java.io.File(paths.localRepository);
if FileObj.getUsableSpace<150e9
%     error(['There is not enough space. Please clear out: ' paths.localRepository])
else
%     continue;%fprintf('There is %d GB of space, continuing experiment...\n', round(FileObj.getUsableSpace/1e9));
end
%%
% open frontcam
% eval('!matlab -nodesktop -nosplash -r "my_frontcam" &')

% open sidecam
eval('!matlab -nodesktop -nosplash -r "my_sidecam" &')

% open eyecam 
eval('!matlab -nodesktop -nosplash -r "my_eyecam" &')

% microphone
% eval('!matlab -nodesktop -nosplash -r "micListener" &')
% timeline
tl.mpepListenerWithWS
end 