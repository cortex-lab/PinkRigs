function extracted = getTrainingData(varargin)
varargin = ['sepPlots', {nan}, varargin];
varargin = ['expDef', {'t'}, varargin];
varargin = ['plotType', {'res'}, varargin];
varargin = ['noPlot', {0}, varargin];
params = csv.inputValidation(varargin{:});

if length(params.subject) > 1 && isnan(params.sepPlots{1})
    fprintf('Multiple subjects, so will combine within subjects \n');
    params.sepPlots = repmat({0},length(params.subject),1);
elseif isnan(params.sepPlots{1})
    params.sepPlots = repmat({1},length(params.subject),1);
end

expList = csv.queryExp(params);
if isempty(expList); error('No subjects found to match criteria'); end
if params.sepPlots{1}
    params = csv.inputValidation(varargin{:}, 'sepPlots', 1,  expList);
else
    params.subject = unique(params.subject);
end

[extracted.subject, extracted.blkDates, extracted.rigNames, extracted.AVParams, extracted.nExp...
    ] = deal(repmat({'X'},length(params.subject),1));
extracted.data = cell(length(params.subject),1);

extracted.validSubjects = ones(length(params.subject),1);
for i = 1:length(params.subject)
    if params.sepPlots{1}
        currData = expList(i,:);
        extracted.blkDates{i} = currData.expDate;
        extracted.rigNames{i} = strrep(currData.rigName, 'zelda-stim', 'Z');
        fprintf('Getting training data for %s on %s... \n', currData.subject{1}, currData.expDate{1});
    else
        currData = expList(strcmp(expList.subject, params.subject{i}),:);
    end
    if isempty(currData)
        fprintf('No matching data for %s \n', params.subject{i});
        extracted.validSubjects(i) = 0;
        continue;
    end

    alignedBlock = cellfun(@(x) strcmp(x(1), '1'), currData.alignBlock);
    if any(~alignedBlock)
        fprintf('Missing block alignments. Will try and align...\n')
        preproc.align.main(varargin{:}, currData(~alignedBlock,:), 'process', 'block');
        currData = csv.queryExp(currData);
    end

    evExtracted = cellfun(@(x) strcmp(x(1), '1'), currData.extractEvents);
    if any(~evExtracted)
        fprintf('EV extractions. Will try to extract...\n')
        preproc.extractExpData(varargin{:}, currData(~evExtracted,:), 'recompute', 'events', 'process', 'events');
        currData = csv.queryExp(currData);
    end
    
    alignedBlock = cellfun(@(x) strcmp(x(1), '1'), currData.alignBlock);
    evExtracted = cellfun(@(x) strcmp(x(1), '1'), currData.extractEvents);

    failIdx = any(~[alignedBlock, evExtracted],2);
    if any(failIdx)
        failNames = currData.expFolder(failIdx);
        cellfun(@(x) fprintf('WARNING: Files missing for %s. Skipping...\n', x), failNames);
        currData = currData(~failIdx,:);
        extracted.validSubjects(i) = 0;
    end
    if isempty(currData)
        extracted.validSubjects(i) = 0;
        continue
    end

    if length(unique(currData.expDate)) ~= length(currData.expDate)
        expDurations = cellfun(@str2double, currData.expDuration);
        [~, ~, uniIdx] = unique(currData.expDate);
        keepIdx = arrayfun(@(x) find(expDurations == max(expDurations(x == uniIdx))), unique(uniIdx), 'uni', 0);
        currData = currData(cell2mat(keepIdx),:);
    end
    extracted.blkDates{i} = currData.expDate;
    extracted.rigNames{i} = strrep(currData.rigName, 'zelda-stim', 'Z');

    loadedEV = csv.loadData(currData, 'dataType', 'ev', 'verbose', 0);
    dataEvents = [loadedEV.dataEvents{:}];

    AVParams = cell(length(dataEvents),1);
    for j = 1:length(dataEvents)
        dataEvents(j).stim_visAzimuth(isnan(dataEvents(j).stim_visAzimuth)) = 0;
        dataEvents(j).stim_visDiff = dataEvents(j).stim_visContrast.*sign(dataEvents(j).stim_visAzimuth);
        dataEvents(j).stim_audDiff = dataEvents(j).stim_audAzimuth;
        AVParams{j,1} = unique([dataEvents(j).stim_audDiff dataEvents(j).stim_visDiff], 'rows');
    end

    [uniParams, ~, uniMode] = unique(cellfun(@(x) num2str(x(:)'), AVParams, 'uni', 0));
    modeIdx = uniMode == mode(uniMode);
    if numel(uniParams) ~= 1
        fprintf('Multiple param sets detected for %s, using mode \n', currData.subject{1});
    end
    names = fieldnames(dataEvents);
    cellData = cellfun(@(f) {vertcat(dataEvents(modeIdx).(f))}, names);

    for j = 1:length(cellData)
        if isa(cellData{j}, 'single')
            cellData{j} = double(cellData{j});
        end
    end

    extracted.subject{i} = currData.subject{1};
    extracted.data{i} = cell2struct(cellData, names);
    extracted.nExp{i} = sum(modeIdx);
    extracted.AVParams{i} = AVParams(find(modeIdx,1));    
    extracted.blkDates{i} = extracted.blkDates{i}(modeIdx);
    extracted.rigNames{i} = extracted.rigNames{i}(modeIdx);
end
if all(cellfun(@isempty, extracted.data))
    warning('No sessions match criteria, returning')
    return;
end
end