    function [ichanidx]=create_channelmapMultishank(binname,path,savemap)

    % binname leads to ap bin file, path is the relevant folder 
    % redundant, but this is how Bill Karsh has wrote the readMeta
    % savemap: 1/0 dependent on whether you want to save the channelmap to
    % path

    meta = readMetaData_spikeGLX(binname,path); 
    

    % phase 2 probe 
    expression = '((\d+):(\d+):(\d+):(\d+))';%
    shankdat=regexp(meta.snsShankMap,expression,'tokens');
    %%
    shankID=zeros(size(shankdat,2),1); % determines which shank we are recording from 
    sideID=zeros(size(shankdat,2),1); % 0 - left side, 1 - right side 
    rowID=zeros(size(shankdat,2),1); % determines which row from the bottom at a given shank
    connectedID=zeros(size(shankdat,2),1); 

    for i=1:size(shankdat,2)
        currentband=shankdat{1,i}{1}; 
        currentband_dat=regexp(currentband,':','split');
        shankID(i)=str2double(currentband_dat{1});
        sideID(i)=str2double(currentband_dat{2});
        rowID(i)=str2double(currentband_dat{3}); 
        connectedID(i)=str2double(currentband_dat{4}); 
    end 
    %%
    xcoords=sideID.*32+shankID.*200;
    ycoords=rowID.*15; 
    chanMap=int32(1:384);
    if savemap==1
        fileName=regexp(meta.fileName,'/','split'); 
        apbinname=regexp(fileName{end},'imec[\d*]','match');
        channelmapname=[path '\' apbinname{1} '_channelmap.mat'];
        save(channelmapname,'xcoords', 'ycoords', 'chanMap');
    end

    [~,ichanidx]=sort(ycoords); % this sorts the channels of acquisiton order according to depth -- have to index into raw with this. 

    end 


