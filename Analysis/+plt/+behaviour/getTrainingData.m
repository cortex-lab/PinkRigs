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
    ] = deal(repmat({{'X'}},length(params.subject),1));
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

    alignedBlock = cellfun(@(x) strcmp(x(1), '1'), currData.alignBlkFrontSideEyeMicEphys);
    if any(~alignedBlock)
        fprintf('Missing block alignments. Will try and align...\n')
        preproc.align.main(varargin{:}, currData(~alignedBlock,:), 'process', 'block');
        currData = csv.queryExp(currData);
    end

    evExtracted = cellfun(@(x) strcmp(x(end), '1'), currData.preProcSpkEV);
    if any(~evExtracted)
        fprintf('EV extractions. Will try to extract...\n')
        preproc.extractExpData(varargin{:}, currData(~evExtracted,:), 'process', 'ev');
        currData = csv.queryExp(currData);
    end
    
    alignedBlock = cellfun(@(x) strcmp(x(1), '1'), currData.alignBlkFrontSideEyeMicEphys);
    evExtracted = cellfun(@(x) strcmp(x(end), '1'), currData.preProcSpkEV);

    failIdx = any(~[alignedBlock, evExtracted],2);
    if any(failIdx)
        failNames = currData.expFolder(failIdx);
        cellfun(@(x) fprintf('WARNING: Files mising for %s. Skipping...\n', x), failNames);
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

    loadedEV = csv.loadData(currData, 'loadTag', 'ev');
    evData = [loadedEV.evData{:}];

    AVParams = cell(length(evData),1);
    for j = 1:length(evData)
        evData(j).stim_visAzimuth(isnan(evData(j).stim_visAzimuth)) = 0;
        evData(j).stim_visDiff = evData(j).stim_visContrast.*sign(evData(j).stim_visAzimuth);
        evData(j).stim_audDiff = evData(j).stim_audAzimuth;
        AVParams{j,1} = unique([evData(j).stim_audDiff evData(j).stim_visDiff], 'rows');
    end

    [uniParams, ~, uniMode] = unique(cellfun(@(x) num2str(x(:)'), AVParams, 'uni', 0));
    modeIdx = uniMode == mode(uniMode);
    if numel(uniParams) ~= 1
        fprintf('Multiple param sets detected for %s, using mode \n', currData.subject{1});
    end
    names = fieldnames(evData);
    cellData = cellfun(@(f) {vertcat(evData(modeIdx).(f))}, names);

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