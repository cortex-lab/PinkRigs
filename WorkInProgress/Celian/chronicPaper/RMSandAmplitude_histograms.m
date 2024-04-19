%% Plot distribution of amplitudes and RMS for example days

% exSubj = 'Lignani001';
% ssEx = find(contains(subjects,exSubj));
% exSubjectIdx = contains(subjectsAll,subjects(ssEx));
% probes = unique(probeSNAll(exSubjectIdx));
% bankSelList = {sprintf('%s__0__0',probes{1})  sprintf('%s__0__0',probes{1})};
% day2pltList = {5 14};

exSubj = 'Lignani002';
ssEx = find(contains(subjects,exSubj));
exSubjectIdx = contains(subjectsAll,subjects(ssEx));
probes = unique(probeSNAll(exSubjectIdx));
bankSelList = {sprintf('%s__0__0',probes{1}) sprintf('%s__0__0',probes{1})};
day2pltList = {11 27};

% exSubj = 'Margrie004';
% ssEx = find(contains(subjects,exSubj));
% exSubjectIdx = contains(subjectsAll,subjects(ssEx));
% probes = unique(probeSNAll(exSubjectIdx));
% bankSelList = {sprintf('%s__0  1  2  3__0',probes{1}) sprintf('%s__0  1  2  3__0',probes{1})};
% day2pltList = {26 27};
% startTime = 10*60;

% exSubj = 'Wikenheiser001';
% ssEx = find(contains(subjects,exSubj));
% exSubjectIdx = contains(subjectsAll,subjects(ssEx));
% probes = unique(probeSNAll(exSubjectIdx));
% bankSelList = {sprintf('%s__0__160',probes{1}) sprintf('%s__0__160',probes{1})};
% day2pltList = {4 22};

rmsBins = 0:2.5:50;
ampBins = 0:25:600;
clear p_rms p_amp
for ll = 1:numel(bankSelList)
    bankSel = bankSelList{ll};
    day2plt = day2pltList{ll};

    recIdx = find(exSubjectIdx & cell2mat(expInfoAll.daysSinceImplant)' == day2plt & contains(recLocAll, bankSel));
    recIdx = recIdx(end); % in case there are two
    expInfo = expInfoAll(recIdx,:);

    % RMS
    probeName = fieldnames(expInfo.dataSpikes{1});
    rec = expInfo.(sprintf('ephysPathP%s',probeName{1}(2:end))){1};
    dateStr = rec(strfind(rec,'Subjects')+9+numel(subjects{ssEx})+(1:10));
    d = dir(fullfile(rec,'*ap.cbin'));
    dat.fileName = fullfile(d.folder,d.name);

    % Get save dir file name
    [binFolder,tag] = fileparts(dat.fileName);
    tag = [subjects{ssEx} '_' dateStr '_' regexprep(tag,'\.','-') '.mat'];
    saveDirAni = fullfile(saveDir,subjects{ssEx});

    % Get scaling factor
    metaFile = regexprep(dat.fileName, 'ap.cbin', 'ap.meta');
    [oldScalingFactor, ~, ~] = bc_readSpikeGLXMetaFile(metaFile, probeType);

    %%% PROBLEM WITH SCALING FACTOR FOR 2.0 AND SPIKEGADGETS -- HACK
    if contains(recLocAll{recIdx}, 'Margrie') && ~contains(recLocAll{recIdx}, 'Margrie002')
        scalingFactor = 1.2*1e6 / (2^12) / 100; % Neuropixels 2.0
    elseif contains(recLocAll{recIdx}, 'Wikenheiser')
        scalingFactor =  0.0183; %Vrange / (2^bits_encoding) / gain;
    else
        scalingFactor = oldScalingFactor;
    end

    load(fullfile(saveDirAni, tag), 'dat')
    p_rms(ll,:) = histcounts(dat.RMS*scalingFactor,rmsBins,'Normalization','probability');

    % Spike amplitude
    clusters = expInfo.dataSpikes{1}.(probeName{1}).clusters;
    clusters.bc_qualityMetrics.rawAmplitude = clusters.bc_qualityMetrics.rawAmplitude*scalingFactor / oldScalingFactor;
    unitType = bc_getQualityUnitType(paramBC, clusters.bc_qualityMetrics);
    idx2Use = ismember(unitType, [1 3]);
    p_amp(ll,:) = histcounts(clusters.bc_qualityMetrics.rawAmplitude(idx2Use),ampBins,'Normalization','probability');
end

figure; 
for ll = 1:numel(bankSelList)
    subplot(121); hold all
    stairs(rmsBins(1:end-1), p_rms(ll,:))

    subplot(122); hold all
    stairs(ampBins(1:end-1), p_amp(ll,:))
end
subplot(121); 
xlabel('RMS (uV)')
ylabel('% channels')
subplot(122); 
xlabel('Unit amplitude (uV)')
ylabel('% unit')

%% All animals

rmsBins = 0:2.5:50;
ampBins = 0:25:600;
p_rms  = nan(size(expInfoAll,1), numel(rmsBins)-1);
p_rms  = nan(size(expInfoAll,1), numel(rmsBins)-1);
for ff = 1:size(expInfoAll,1)
    expInfo = expInfoAll(ff,:);

    % RMS
    probeName = fieldnames(expInfo.dataSpikes{1});
    rec = expInfo.(sprintf('ephysPathP%s',probeName{1}(2:end))){1};
    dateStr = rec(strfind(rec,'Subjects')+9+numel(subjectsAll{ff})+(1:10));
    d = dir(fullfile(rec,'*ap.cbin'));
    dat.fileName = fullfile(d.folder,d.name);

    % Get save dir file name
    [binFolder,tag] = fileparts(dat.fileName);
    tag = [subjectsAll{ff} '_' dateStr '_' regexprep(tag,'\.','-') '.mat'];
    saveDirAni = fullfile(saveDir,subjectsAll{ff});

    % Get scaling factor
    metaFile = regexprep(dat.fileName, 'ap.cbin', 'ap.meta');
    [oldScalingFactor, ~, ~] = bc_readSpikeGLXMetaFile(metaFile, probeType);

    %%% PROBLEM WITH SCALING FACTOR FOR 2.0 AND SPIKEGADGETS -- HACK
    if contains(recLocAll{ff}, 'Margrie') && ~contains(recLocAll{ff}, 'Margrie002')
        scalingFactor = 1.2*1e6 / (2^12) / 100; % Neuropixels 2.0
    elseif contains(recLocAll{ff}, 'Wikenheiser')
        scalingFactor =  0.0183; %Vrange / (2^bits_encoding) / gain;
    else
        scalingFactor = oldScalingFactor;
    end

    load(fullfile(saveDirAni, tag), 'dat')
    p_rms(ff,:) = histcounts(dat.RMS*scalingFactor,rmsBins,'Normalization','probability');

    % Spike amplitude
    clusters = expInfo.dataSpikes{1}.(probeName{1}).clusters;
    clusters.bc_qualityMetrics.rawAmplitude = clusters.bc_qualityMetrics.rawAmplitude*scalingFactor / oldScalingFactor;
    unitType = bc_getQualityUnitType(paramBC, clusters.bc_qualityMetrics);
    idx2Use = ismember(unitType, [1 3]);
    p_amp(ff,:) = histcounts(clusters.bc_qualityMetrics.rawAmplitude(idx2Use),ampBins,'Normalization','probability');
end

%% 
subjFree = {'Lignani001', 'AV043'};
subjHolder = {'Lignani002', 'Mainen001', 'Duan001', 'Duan002'};
subjMini = subjects(contains(subjects,'Margrie'));
subjRats = subjects(contains(subjects,'Wikenheiser'));
subjHead = subjects(~contains(subjects, [subjFree, subjHolder, subjMini, subjRats]));

subjToPlot = subjHolder;
figure('Position', [902   765   338   213]);
subplot(121); hold all
stairs(rmsBins(1:end-1), nanmean(p_rms(contains(subjectsAll, subjHead),:)), 'k')
stairs(rmsBins(1:end-1), nanmean(p_rms(contains(subjectsAll, subjToPlot),:)), 'Color', [39 180 159]/256)
xlabel('RMS (uV)')
ylabel('% channels')

subplot(122); hold all
stairs(ampBins(1:end-1), nanmean(p_amp(contains(subjectsAll, subjHead),:)), 'k')
stairs(ampBins(1:end-1), nanmean(p_amp(contains(subjectsAll, subjToPlot),:)), 'Color', [39 180 159]/256)
xlabel('Unit amplitude (uV)')
ylabel('% unit')
