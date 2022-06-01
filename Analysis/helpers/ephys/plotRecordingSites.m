function plotRecordingSites(recFolderList,savePlt,recompute)
    %%% This function will read the recordings' metadata and plot the 
    %%% recording sites (and save the plot if savePlt).
    %%% recFolderList contains the list of recordings (path up to the  
    %%% folder containing both probes)
    
    if ~exist('recompute','var')
        recompute = 0;
    end
    
    probeColor = [0.4 0.6 0.2; ...
        0.9 0.3 0.0];
    
    for rr = 1:numel(recFolderList)
        recFullPath = recFolderList{rr};
        [~,recName] = fileparts(recFullPath);
        
        probeFolders = dir(fullfile(recFullPath,'*imec*'));
        
        if isempty(probeFolders)
            sprintf('No recording here: %s. Skipping.\n', recFullPath)
            
        else
            pltName = fullfile(probeFolders(1).folder,sprintf('RecordingSites_%s.png',recName));
            
            if exist(pltName,'file') && ~recompute
                % sprintf('Already saved for %s. Skipping.\n', recFullPath)
            else
                if savePlt
                    f = figure('Position', [992   566   248   412],'visible','off'); hold all
                else
                    f = figure('Position', [992   566   248   412]); hold all
                end
                for probeNum = 1:numel(probeFolders)
                    binFile = dir(fullfile(probeFolders(probeNum).folder,probeFolders(probeNum).name,'*ap.*bin'));
                    metaData = readMetaData_spikeGLX(binFile.name,binFile.folder);
                    
                    %% Extract info from metadata
                    %%% Same as in plotIMROProtocol
                    out = regexp(metaData.imroTbl,'\(|\)(|\)','split');
                    out(1:2) = []; % empty + extra channel or something?
                    out(end) = []; % empty
                    
                    chans = nan(1,numel(out));
                    shank = nan(1,numel(out));
                    bank = nan(1,numel(out));
                    elecInd = nan(1,numel(out));
                    for c = 1:numel(out)
                        chanProp = regexp(out{c},' ','split');
                        chans(c) = str2double(chanProp{1});
                        shank(c) = str2double(chanProp{2});
                        bank(c) = str2double(chanProp{3});
                        % 4 is refElec
                        elecInd(c) = str2double(chanProp{5});
                    end
                    
                    %% plot data -- taken from IMRO generation script
                    % NP 2.0 MS (4 shank), probe type 24 electrode positions
                    nElec = 1280;   %per shank; pattern repeats for the four shanks
                    vSep = 15;      % in um
                    hSep = 32;
                    
                    elecPos = zeros(nElec, 2);
                    
                    elecPos(1:2:end,1) = 0;                %sites 0,2,4...
                    elecPos(2:2:end,1) =  hSep;            %sites 1,3,5...
                    
                    % fill in y values
                    viHalf = (0:(nElec/2-1))';                %row numbers
                    elecPos(1:2:end,2) = viHalf * vSep;       %sites 0,2,4...
                    elecPos(2:2:end,2) = elecPos(1:2:end,2);  %sites 1,3,5...
                    
                    chanPos = elecPos(elecInd+1,:);
                    
                    % make a plot of all the electrode positions
                    shankSep = 250;
                    for sI = 0:3
                        cc = find(shank == sI);
                        scatter((probeNum-1)*8*shankSep + shankSep*sI + elecPos(:,1), elecPos(:,2), 10, probeColor(probeNum,:), 'square' );
                        scatter((probeNum-1)*8*shankSep + shankSep*sI + chanPos(cc,1), chanPos(cc,2), 20, [.4 .2 .6], 'square', 'filled' );
                    end
                    xlim([-16,(numel(probeFolders)-1)*8*shankSep + numel(probeFolders)*3*shankSep+64]);
                    ylim([-10,10000]);
                    text((probeNum-1)*8*shankSep + 0.5*shankSep+64,10000,sprintf('Probe %d', probeNum))
                end
                title(regexprep(recName,'_',' '))
                if savePlt
                    saveas(f,fullfile(probeFolders(probeNum).folder,sprintf('RecordingSites_%s.png',recName)),'png')
                end
            end
        end
    end
 