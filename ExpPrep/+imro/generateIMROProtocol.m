function generateIMROProtocol(basePath,imroprop,days)
    %% Generate the architecture and IMRO files for a given protocol.
    %
    % Parameters:
    % -------------------
    % basePath: str
    %   Path for the IMRO protocol
    % imroprop: struct
    %   Structure for IMRO specs, as in 'exampleIMROProtocol', with fields:
    %     protocol: name of the protocol
    %     probe(1).patternTag: pattern of shanks 'hs2','hs4','ss'
    %     probe(1).botRow: bottom row to use
    %     probe(1).shankChoice: array of shanks to record from
    %     probe(1).refElec: Reference (0 is external, 1 internal)
    % days: cell str
    %   List of days ('YYYY-MM-DD')
    
    if ~exist('days','var')
        days = cellstr(num2str((1:numel(imroprop))'));
    end

    if ~exist(basePath,'dir')
        mkdir(basePath)
    end
    
    %% Find serial numbers
    [~,protocolName] = fileparts(basePath);
    
    % Get expected serial numbers
    csvData = csv.readTable(csv.getLocation('main'));
    csvData = csvData(strcmp(csvData.Subject,protocolName),:);
    if ~isempty(csvData)
        csvFields = fields(csvData);
        serialsFromCSV = cellfun(@(x) csvData.(x), csvFields(contains(csvFields, 'serial')), 'uni', 0)';
        serialsFromCSV = cell2mat(cellfun(@str2double, serialsFromCSV, 'uni', 0));
    else
        warning('Can''t recognize subject. Won''t put the serial numbers.')
        serialsFromCSV = [nan nan];
    end
    
    %% Loop through days and protocols
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
                prefix = sprintf('Probe%d_%d_%s_SN%d_',probeNum-1,d,protocol,serialsFromCSV(probeNum));
                fileName = imro.generateIMRO_P24(probeProp.patternTag, probeProp.botRow, probeProp.shankChoice, probeProp.refElec, savePath);
                [path,file,ext] = fileparts(fileName);
                movefile(fileName,fullfile(path,[prefix file ext]))
            end
        end
    end
end