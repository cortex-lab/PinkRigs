function []=run_kilosortFT(ephys_folder)
% this funtion takes a single ephys folder (most often mname/date/ephys)
% and sorts and CARs raw data within it
    kilosortworkfolder='C:\Users\Flora\Documents\KSworkfolder'; % local folder on ssd where I process the data ==rootH 
    % check which days from the mice's folder contain ephys data
    if ~exist(kilosortworkfolder, 'dir')
       mkdir(kilosortworkfolder)
    end
    
    kilosortoutputfolder='C:\Users\Flora\Documents\kilosort'; % rootZ
    if ~exist(kilosortoutputfolder, 'dir')
       mkdir(kilosortoutputfolder)
    end    
    
    apbinfiles=dir([ephys_folder '\**\*ap.bin*']); % get all the probe filenames   

    
    %% run kilosort
    % remove filenames that are in the kilosort folder 
    while contains(apbinfiles(end).folder,'kilosort')==1
        apbinfiles(end)=[];
    end

    %%
    proctag=0; 
    for myprobe=1:size(apbinfiles,1)    
        myapbin=apbinfiles(myprobe).name;
        probename=myapbin((length(myapbin)-11):(length(myapbin)-7)); % imec0/1
        probesortedfolder=[kilosortoutputfolder sprintf('\\%s',probename)]; 
        myAPdata=[apbinfiles(myprobe).folder '\' apbinfiles(myprobe).name]; 
        
        
        % read metadata to decide whether to create channelmap or supply
        % from KS 
        meta = kilo.ReadMeta_GLX(myAPdata,apbinfiles(myprobe).folder);
        if contains(meta.imDatPrb_type,'0')
            % phase 3B probe -- just load the default kilosort map
            channelmapdir='C:\Users\Flora\Documents\Github\AV_passive\preprocessing\configFiles_kilosort2\neuropixPhase3B2_kilosortChanMap.mat';
        elseif contains(meta.imDatPrb_type,'2')
            % create channelmap (good for all phase2, even single shank) or copy P3B map?
            [~]=create_channelmapMultishank(myAPdata,apbinfiles(myprobe).folder,0);
            channelmapfile=dir([apbinfiles(myprobe).folder '\\**\\*_channelmap.mat*']);
            channelmapdir=[channelmapfile(1).folder '\' channelmapfile(1).name]; % channelmap for the probe - should be in the same folder
        end
        if ~exist([ephys_folder '\\kilosort' sprintf('\\%s\\spike_templates.npy',probename)])==1
            proctag=1; %
            fprintf('need to process %s file\n',myapbin);
            
            if ~exist(probesortedfolder, 'dir')
               mkdir(probesortedfolder)
            end 
            
            if ~exist([probesortedfolder myapbin])==1
                disp('copying data to local SSD');          
                copyfile(myAPdata,probesortedfolder);
                disp('copied data') 
            else 
                disp('data already copied');
            end
            
            
            kilo.Kilosort2Matlab(probesortedfolder,kilosortworkfolder,channelmapdir)
            delete([probesortedfolder '\' myapbin]); % delete if you also
            %CAR the data
            % copy all other output to znas  
            
            % Get quality metrics
            kilo.getQualityMetrics(kilosortworkfolder, apbinfiles(myprobe).folder)
        else 
            fprintf('already processed %s file\n',myapbin);
        end
    end

    % copy the output back to znas  if there was some processing 
    if proctag==1 
        movefile(kilosortoutputfolder,ephys_folder)
    end 
    %% extract sync pulse  
    for myprobe=1:size(apbinfiles,1)    
        myapbin=apbinfiles(myprobe).name;
        myAPdata=[apbinfiles(myprobe).folder '\' apbinfiles(myprobe).name]; 
        probename=myapbin((length(myapbin)-11):(length(myapbin)-7)); 
        probesortedfolder=[ephys_folder '\\kilosort' sprintf('\\%s',probename)];
        d=dir([probesortedfolder '\**\sync.mat']);
        if numel(d)<1            
            kilo.syncFT(myAPdata, 385, probesortedfolder);
        else 
            disp('sync extracted already.');
        end 
    end 
    %% CAR the data
%     for myprobe=1:size(apbinfiles,1)    
%         myapbin=apbinfiles(myprobe).name;
%         myAPdata=[apbinfiles(myprobe).folder '\' apbinfiles(myprobe).name]; 
%         probename=myapbin((length(myapbin)-11):(length(myapbin)-7)); 
%         probesortedfolder=[ephys_folder '\\kilosort' sprintf('\\%s',probename)];
%         d=dir([probesortedfolder '\**\*CAR.bin*']);
%         if numel(d)<1            
%             %disp('calculating CARed data...');
%             %applyCARtoDat(myAPdata, 385, probesortedfolder);% if the data is spikeGLX the channel no is 385 if openephys: 384
%             %syncFT(myAPdata, 385, probesortedfolder);
% %             [~,sync] = applyCARandSync(myAPdata, 385, probesortedfolder);
% %             disp('saving sync...')
% %             save(sprintf('%s\\sync.mat', probesortedfolder),'sync');
% %             disp('done.');
%         else 
%             disp('CAR done already.');
%         end 
%     end

   
    
end


% alignments --
