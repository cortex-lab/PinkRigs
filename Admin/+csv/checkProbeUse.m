function probeInfo = checkProbeUse(queryData, implantSelect, mkPlt, csvData)
%% Function to give implantation info about a probe, or a mouse
% "queryData" can be a cell array/vector or probe numbers or mice names

if ~exist('csvData', 'var')
    csvData = csv.readTable(csv.getLocation('main'));
end
csvFields = fields(csvData);
serialsFromCSV = cellfun(@(x) csvData.(x), csvFields(contains(csvFields, 'serial')), 'uni', 0)';
serialsFromCSV = cell2mat(cellfun(@str2double, serialsFromCSV, 'uni', 0));

if ~exist('queryData','var') || isempty(queryData)
    queryData = unique(serialsFromCSV);
    queryData(isnan(queryData)) = [];
end
if ~exist('mkPlt','var');  mkPlt = 0; end
if ~exist('implantSelect','var');  implantSelect = 'all'; end
if ~iscell(queryData); queryData = num2cell(queryData,2); end


subjects = cell(1,numel(queryData));
implantDate = cell(1,numel(queryData));
explantDate = cell(1,numel(queryData));
probeInfo = struct;

if isnumeric(queryData{1})
    for i = 1:numel(queryData)
        % get their orders and dates
        snIdx = find(serialsFromCSV == queryData{i});
        [subjectIdx,probeIdx] = ind2sub(size(serialsFromCSV),snIdx);
        for pp = 1:numel(probeIdx)
            subjects{i}{pp} = csvData.Subject{subjectIdx(pp)};
            implantDate{i}{pp} = csvData.(sprintf('P%d_implantDate',probeIdx(pp)-1)){subjectIdx(pp)};
            explantDate{i}{pp} = csvData.(sprintf('P%d_explantDate',probeIdx(pp)-1)){subjectIdx(pp)};
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


        probeInfo.implantedSubjects{i,1} = subjects{i};
        probeInfo.implantDates{i,1} = implantDate{i};
        probeInfo.explantDates{i,1} = explantDate{i};
        probeInfo.serialNumber{i,1} = queryData{i};
    end
    if strcmpi(implantSelect, 'last')
        probeInfo.implantedSubjects = cellfun(@(x) x{end}, probeInfo.implantedSubjects, 'uni', 0);
        probeInfo.implantDates = cellfun(@(x) x{end}, probeInfo.implantDates, 'uni', 0);
        probeInfo.explantDates = cellfun(@(x) x{end}, probeInfo.explantDates, 'uni', 0);
    end
else
    implantDate = cellfun(@(x) csvData.P0_implantDate{strcmp(csvData.Subject, x)}, queryData, 'uni', 0);
    validImplants = cellfun(@(x) ~isempty(regexp(x,'\d\d\d\d-\d\d-\d\d', 'once')), implantDate);
    implantDate(~validImplants) = deal({'none'});

    explantDate = cellfun(@(x) csvData.P0_explantDate{strcmp(csvData.Subject, x)}, queryData, 'uni', 0);
    validExplants = cellfun(@(x) ~isempty(regexp(x,'\d\d\d\d-\d\d-\d\d', 'once')), explantDate);
    explantDate(~validExplants) = deal({'none'});

    serialNumbers = cellfun(@(x) serialsFromCSV(strcmp(csvData.Subject, x),:), queryData, 'uni', 0);
    serialNumbers = cellfun(@(x) x(~isnan(x)), serialNumbers, 'uni', 0);

    probeType = cellfun(@(x) csvData.P0_type{strcmp(csvData.Subject, x)}, queryData, 'uni', 0);

    % resort it
    probeInfo.subject = queryData;
    probeInfo.implantDate = implantDate;
    probeInfo.explantDate = explantDate;
    probeInfo.serialNumbers = serialNumbers;
    probeInfo.probeType = probeType;
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
        colormap(plt.general.redBlueMap)
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