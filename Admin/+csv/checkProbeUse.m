function probeInfo = checkProbeUse(queryData, implantSelect, mkPlt, csvData)
    %% Gets information about a specific probe or a specific subject's probe.
    %
    % Parameters:
    % -------------------
    % queryData: can be a cell array/vector
    %    Probe numbers or mice names
    %
    % Returns: 
    % -------------------
    % probeInfo: struct
    %   Structure with the info about each animal/probe.
    %   If animal:
    %    probeInfo.subject
    %    probeInfo.implantDate
    %    probeInfo.explantDate
    %    probeInfo.serialNumbers
    %    probeInfo.probeType
    %   If probe SN:
    %    probeInfo.implantedSubjects
    %    probeInfo.implantDates
    %    probeInfo.explantDates
    %    probeInfo.positionAP
    %    probeInfo.positionML
    %    probeInfo.serialNumber

    %% 
    
    if ~exist('csvData', 'var')
        csvData = csv.readTable(csv.getLocation('main'));
    end
    csvFields = fields(csvData);
    serialsFromCSV = cellfun(@(x) csvData.(x), csvFields(contains(csvFields, 'serial')), 'uni', 0)';
    serialsFromCSV = cell2mat(cellfun(@str2double, serialsFromCSV, 'uni', 0));
    
    probeTypeFromCSV = cellfun(@(x) csvData.(x), csvFields(contains(csvFields, 'type')), 'uni', 0)';
    probeTypeFromCSV = cat(2,probeTypeFromCSV{:});
    
    if ~exist('queryData','var') || isempty(queryData)
        queryData = unique(serialsFromCSV);
        queryData(isnan(queryData)) = [];
    end
    if ~exist('mkPlt','var');  mkPlt = 0; end
    if ~exist('implantSelect','var');  implantSelect = 'all'; end
    if isnumeric(queryData); queryData = reshape(queryData,[numel(queryData),1]); end
    if ~iscell(queryData); queryData = num2cell(queryData,2); end
    
    subjects = cell(1,numel(queryData));
    implantDate = cell(1,numel(queryData));
    explantDate = cell(1,numel(queryData));
    positionAP = cell(1,numel(queryData));
    positionML = cell(1,numel(queryData));
    probeInfo = struct;
    
    if isnumeric(queryData{1})
        % Case where the input is a probe SN
        for i = 1:numel(queryData)
            % get their orders and dates
            snIdx = find(serialsFromCSV == queryData{i});
            [subjectIdx,probeIdx] = ind2sub(size(serialsFromCSV),snIdx);
            for pp = 1:numel(probeIdx)
                subjects{i}{pp} = csvData.Subject{subjectIdx(pp)};
                implantDate{i}{pp} = csvData.(sprintf('P%d_implantDate',probeIdx(pp)-1)){subjectIdx(pp)};
                explantDate{i}{pp} = csvData.(sprintf('P%d_explantDate',probeIdx(pp)-1)){subjectIdx(pp)};
                positionAP{i}{pp} = csvData.(sprintf('P%d_AP',probeIdx(pp)-1)){subjectIdx(pp)};
                positionML{i}{pp} = csvData.(sprintf('P%d_ML',probeIdx(pp)-1)){subjectIdx(pp)};
            end
            undefIdx = cellfun(@isempty, explantDate{i}) | ...
                cellfun(@(x) strcmp(x,'Permanent'), explantDate{i});
            implantDate{i}(undefIdx) = {datestr(now+20,'yyyy-mm-dd')};
            explantDate{i}(undefIdx) = {datestr(now+20,'yyyy-mm-dd')};
    
            % resort it
            [~,sortIdx] = sort(datenum(explantDate{i},'yyyy-mm-dd'),'ascend');
            subjects{i} = subjects{i}(sortIdx);
            implantDate{i} = implantDate{i}(sortIdx);
            explantDate{i} = explantDate{i}(sortIdx);
            positionAP{i} = positionAP{i}(sortIdx);
            positionML{i} = positionML{i}(sortIdx);
    
            probeInfo.implantedSubjects{i,1} = subjects{i};
            probeInfo.implantDates{i,1} = implantDate{i};
            probeInfo.explantDates{i,1} = explantDate{i};
            probeInfo.positionAP{i,1} = positionAP{i};
            probeInfo.positionML{i,1} = positionML{i};
            probeInfo.serialNumber{i,1} = queryData{i};
        end
        if strcmpi(implantSelect, 'last')
            probeInfo.implantedSubjects = cellfun(@(x) x{end}, probeInfo.implantedSubjects, 'uni', 0);
            probeInfo.implantDates = cellfun(@(x) x{end}, probeInfo.implantDates, 'uni', 0);
            probeInfo.explantDates = cellfun(@(x) x{end}, probeInfo.explantDates, 'uni', 0);
            probeInfo.positionAP = cellfun(@(x) x{end}, probeInfo.positionAP, 'uni', 0);
            probeInfo.positionML = cellfun(@(x) x{end}, probeInfo.positionML, 'uni', 0);
        end
    else
        % case where the input is an animal's name
        P0_implantDates = csvData.P0_implantDate; %(This plays better with cmd calls)
        implantDate = cellfun(@(x) P0_implantDates{strcmp(csvData.Subject, x)}, queryData, 'uni', 0);
        validImplants = cellfun(@(x) ~isempty(regexp(x,'\d\d\d\d-\d\d-\d\d', 'once')), implantDate);
        implantDate(~validImplants) = deal({'none'});
    
        P0_explantDates = csvData.P0_explantDate; %(This plays better with cmd calls)
        explantDate = cellfun(@(x) P0_explantDates{strcmp(csvData.Subject, x)}, queryData, 'uni', 0);
        validExplants = cellfun(@(x) ~isempty(regexp(x,'\d\d\d\d-\d\d-\d\d', 'once')), explantDate);
        explantDate(~validExplants) = deal({'none'});
    
        serialNumbers = cellfun(@(x) serialsFromCSV(strcmp(csvData.Subject, x),:), queryData, 'uni', 0);
        serialNumbers = cellfun(@(x) x(~isnan(x)), serialNumbers, 'uni', 0);
    
        probeTypes = cellfun(@(x) probeTypeFromCSV(strcmp(csvData.Subject, x),:), queryData, 'uni', 0);
        probeTypes = cellfun(@(y) y(cell2mat(cellfun(@(x) ~isempty(x), y, 'uni', 0))), probeTypes, 'uni', 0);
    
        % resort it
        probeInfo.subject = queryData;
        probeInfo.implantDate = implantDate;
        probeInfo.explantDate = explantDate;
        probeInfo.serialNumbers = serialNumbers;
        probeInfo.probeType = probeTypes;
    end
    
    %% Plot figure
    if mkPlt
        figure;
        for i = 1:numel(queryData)
            subplot(ceil(sqrt(numel(queryData))),ceil(sqrt(numel(queryData))),i); hold all
            minDate = datenum(implantDate{i}{1},'yyyy-mm-dd');
            maxDate = datenum(explantDate{i}{end},'yyyy-mm-dd');
            miceMat = zeros(numel(subjects{i}),maxDate-minDate+1);
            for pp = 1:numel(subjects{i})
                miceMat(pp,datenum(implantDate{i}{pp},'yyyy-mm-dd')-minDate+1:datenum(explantDate{i}{pp},'yyyy-mm-dd')-minDate+1) = pp;
            end
            imagesc(miceMat);
            colormap(plts.general.redBlueMap)
            caxis([-max(cellfun(@numel, subjects)) max(cellfun(@numel, subjects))])
    
            % add animal names
            yticks(1:numel(subjects{i}))
            yticklabels(subjects{i})
    
            % put proper implant/explant dates along x axis
            dates = [datenum(implantDate{i},'yyyy-mm-dd')-minDate+1; ...
                datenum(explantDate{i},'yyyy-mm-dd')-minDate];
            dateLabels = [implantDate{i} explantDate{i}];
            [~,sortDatesIdx] = sort(dates);
            xticks(dates(sortDatesIdx))
            xticklabels(dateLabels(sortDatesIdx))
            xtickangle(45)
    
            % SN in title
            title(sprintf('Probe #%d',queryData{i}))
        end
    end
end