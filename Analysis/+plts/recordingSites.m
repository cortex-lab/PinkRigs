function plotRecordingSites(recFolderList,savePlt,recompute)
    %% Plots the recording sites (and save the plot if savePlt).
    %
    % Parameters:
    % -------------------
    % recFolderList: cell of str
    %   Contains the list of recordings
    % savePlt: bool
    %   Whether to save the plot
    % recompute: bool
    %   Whether to recompute 
    
    if ~exist('recompute','var')
        recompute = 0;
    end
    if ~iscell(recFolderList)
        recFolderList = {recFolderList};
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
                    [chanPos, elecPos, shank] = getRecordingSites(binFile(1).name,binFile(1).folder);
                    
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
 