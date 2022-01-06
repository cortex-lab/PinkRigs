function []=initialise_kilosort()
%% deal with paths
addpath(genpath('C:\Users\Flora\Documents\GitHub\KiloSort2')) % path to kilosort folder
addpath('C:\Users\Flora\Documents\GitHub\npy-matlab') % for converting to Phy

csvRoot='\\zserver.cortexlab.net\Code\AVrig\'; % folder in which queue csv resides
pathToKSConfigFile = 'C:\Users\Flora\Documents\Github\AV_passive\preprocessing\configFiles_kilosort2'; 
kilosortworkfolder='C:\Users\Flora\Documents\KSworkfolder'; % local folder on ssd where I process the data for whitening (rootH) 
kilosortoutputfolder='C:\Users\Flora\Documents\kilosort'; % local temporal folder for output (rootZ)
defaultP3Bchanmap='C:\Users\Flora\Documents\Github\AV_passive\preprocessing\configFiles_kilosort2\neuropixPhase3B2_kilosortChanMap.mat';


%% the rest 
% check which days from the mice's folder contain ephys data
if ~exist(kilosortworkfolder, 'dir')
   mkdir(kilosortworkfolder)
end

if ~exist(kilosortoutputfolder, 'dir')
   mkdir(kilosortoutputfolder)
end
% check active mice
csvLocation = [csvRoot 'kilosort_queue.csv'];
csvData = readtable(csvLocation,'Delimiter',',');
activeSortQueue=csvData.ephysName(csvData.sortedTag==0);
idx=find(csvData.sortedTag==0);

% loop over recordings to be sorted 
for recidx=1:numel(activeSortQueue)
% identify parent folder where we will push the output
    myAPdata=[activeSortQueue{1}]; 
    [ephys_folder,b,c]=fileparts(myAPdata); myapbin=strcat(b,c);

    % indentify meta file and create channel map
    meta=ReadMeta_GLX(myAPdata,ephys_folder); 
    if contains(meta.imDatPrb_type,'0')
    % phase 3B probe -- just load the default kilosort map
    channelmapdir=defaultP3Bchanmap;

    elseif contains(meta.imDatPrb_type,'2')
        % create channelmap (good for all phase2, even single shank) or copy P3B map?    
        fprintf('creating custom channelmap...') 
        [~]=create_channelmapMultishank(myAPdata,apbinfiles(myprobe).folder,0);        
        channelmapfile=dir([apbinfiles(myprobe).folder '\\**\\*_channelmap.mat*']);
        channelmapdir=[channelmapfile(1).folder '\' channelmapfile(1).name]; % channelmap for the probe - should be in the same folder
    end
    %%
    if ~exist([kilosortoutputfolder myapbin])==1
        disp('copying data to local SSD');          
        copyfile(myAPdata,kilosortoutputfolder);
        disp('copied data') 
    else 
        disp('data already copied');
    end

    try
        Kilosort2Matlab(kilosortoutputfolder,kilosortworkfolder,channelmapdir,pathToKSConfigFile)
        delete([kilosortoutputfolder '\' myapbin]); % delete .bin file from KS output
        movefile(kilosortoutputfolder,ephys_folder) % copy KS output back to server


        % extract sync pulse 
        probesortedfolder=[ephys_folder '\\kilosort'];
        d=dir([probesortedfolder '\**\sync.mat']);
        if numel(d)<1            
            syncFT(myAPdata, 385, probesortedfolder);
        else 
            disp('sync extracted already.');
        end

        % overwrite the queue
        csvData.sortedTag(idx)=1; 
    catch 
        % sorting was not successful write a permanent tag indicating that
        csvData.sortedTag(idx)=-1;
        errorMsge=jsonencode(lastwarn);
        fid = fopen([ephys_folder '\KSerror.json'], 'w');
        fprintf(fid, '%s', errorMsge);
        fclose(fid);
    end
    % save the updated queue

    writetable(csvData,csvLocation,'Delimiter',',');
end
end
