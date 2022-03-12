function generateIMROProtocol(basePath,imroprop,days)
    %%% This function will generate the architecture and IMRO files for a
    %%% given protocol.
    
    if ~exist('days','var')
        days = cellstr(num2str([1:numel(imroprop)]'));
    end

    if ~exist(basePath,'dir')
        mkdir(basePath)
    end
    
    % Loop through days and protocols
    for d = 1:numel(imroprop)
                
        if exist(fullfile(basePath,days{d}),'dir')
            % just save it somewhere else
            today = datestr(now);
            movefile(fullfile(basePath,days{d}), regexprep([fullfile(basePath,days{d}) '_BACKUP' today],' |:','_'))
        end
        mkdir(fullfile(basePath,days{d}))
        
        for p = 1:numel(imroprop{d})
            % Get protocol
            protocol = imroprop{d}{p}.protocol;
            savePath = fullfile(basePath,days{d},protocol);
            mkdir(savePath)
            for probeNum = 1:numel(imroprop{d}{p}.probe)
                probeProp = imroprop{d}{p}.probe(probeNum);
                prefix = sprintf('Probe%d_%d_%s_',probeNum-1,d,protocol);
                fileName = generateIMRO_P24(probeProp.patternTag, probeProp.botRow, probeProp.shankChoice, probeProp.refElec, savePath);
                [path,file,ext] = fileparts(fileName);
                movefile(fileName,fullfile(path,[prefix file ext]))
            end
        end
    end
end