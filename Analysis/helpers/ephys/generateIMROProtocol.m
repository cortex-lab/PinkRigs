function generateIMROProtocol(basePath,imroprop)
    %%% This function will generate the architecture and IMRO files for a
    %%% given protocol.
    
    if exist(basePath,'dir')
        % just save it somewhere else
        today = datestr(now);
        movefile(basePath, regexprep([basePath '_' today]),' |:','_')
    end
    
    mkdir(basePath)
    
    % Loop through days and protocols
    for d = 1:numel(imroprop)
        mkdir(fullfile(basePath,sprintf(sprintf('Day%d',d))))
        for p = 1:numel(imroprop{d})
            % Get protocol
            protocol = imroprop{d}{p}.protocol;
            savePath = fullfile(basePath,sprintf('Day%d',d),protocol);
            mkdir(savePath)
            for probeNum = 1:numel(imroprop{d}{p}.probe)
                probeProp = imroprop{d}{p}.probe(probeNum);
                prefix = sprintf('Day%d_%s_probe%d_',d,protocol,probeNum-1);
                fileName = kilo.generateIMRO_P24(probeProp.patternTag, probeProp.botRow, probeProp.shankChoice, probeProp.refElec, savePath);
                [path,file,ext] = fileparts(fileName);
                movefile(fileName,fullfile(path,[prefix file ext]))
            end
        end
    end
end