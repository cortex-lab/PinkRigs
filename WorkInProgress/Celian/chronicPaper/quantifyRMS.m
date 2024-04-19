%% Load data

load('\\znas.cortexlab.net\Lab\Share\Celian\dataForPaper_ChronicImplant_stability_withQM_2024_03_21');

saveDir = 'D:\RMS';

%% Loop through files to extract 10s and compute RMS

startTime = 1*60;
winSize = 30;
process = 1;

recInfo = cellfun(@(x) split(x,'__'),recLocAll,'uni',0);
subjectsAll = cellfun(@(x) x{1}, recInfo, 'UniformOutput', false);

err = cell(size(expInfoAll,1),1);
for ff = 1:size(expInfoAll,1)
    clear dat
    try
        expInfo = expInfoAll(ff,:);
        probeName = fieldnames(expInfo.dataSpikes{1});
        rec = expInfo.(sprintf('ephysPathP%s',probeName{1}(2:end))){1};
        dateStr = rec(strfind(rec,'Subjects')+9+numel(subjectsAll{ff})+(1:10));
        d = dir(fullfile(rec,'*ap.cbin'));
        dat.fileName = fullfile(d.folder,d.name);

        % Get save dir file name
        [binFolder,tag] = fileparts(dat.fileName);
        tag = [subjectsAll{ff} '_' dateStr '_' regexprep(tag,'\.','-') '.mat'];
        saveDirAni = fullfile(saveDir,subjectsAll{ff});
        if ~exist(saveDirAni,'dir')
            mkdir(saveDirAni)
        end

        if ~exist(fullfile(saveDirAni, tag))
            % Get raw data snippet
            fprintf('Extracting from %s...\n', tag)
            data = readRawDataChunk(dat.fileName,startTime,winSize,process);
            dat.RMS = rms(data,1);

            % save
            save(fullfile(saveDirAni, tag), 'dat');
        else
            if exist(fullfile(saveDirAni, tagOld))
                movefile(fullfile(saveDirAni, tagOld), fullfile(saveDirAni, tag))
            end

            fprintf('%s done.\n', tag)
        end
    catch me
        warning('error on %s', tag)
        err{ff} = me;
    end
end

%%

% Get mice info
mice = csv.readTable(csv.getLocation('main'));

recInfo = cellfun(@(x) split(x,'__'),recLocAll,'uni',0);
subjectsAll = cellfun(@(x) x{1}, recInfo, 'UniformOutput', false);
probeSNAll = cellfun(@(x) x{2}, recInfo, 'UniformOutput', false);

subjects = unique(subjectsAll);
colAni = brewermap(numel(subjects),'paired');
colAni(:) = 0.5;

% Get probe info
probeSNUni = unique(probeSNAll);
probeInfo = csv.checkProbeUse(str2double(probeSNUni));


%% fetch RMS

yRng = [0 30];
probeType = 'NaN';

rmsq = nan(1,size(expInfoAll,1));
for ff = 1:size(expInfoAll,1)
    expInfo = expInfoAll(ff,:);
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
    [scalingFactor, ~, ~] = bc_readSpikeGLXMetaFile(metaFile, probeType);

    %%% PROBLEM WITH SCALING FACTOR FOR 2.0 AND SPIKEGADGETS -- HACK
    if contains(recLocAll{ff}, 'Margrie') && ~contains(recLocAll{ff}, 'Margrie002')
        scalingFactor = 1.2*1e6 / (2^12) / 100; % Neuropixels 2.0
    elseif contains(recLocAll{ff}, 'Wikenheiser')
        scalingFactor =  0.0183; %Vrange / (2^bits_encoding) / gain;
    end
    
    if exist(fullfile(saveDirAni, tag))
        load(fullfile(saveDirAni, tag), 'dat')
        rmsq(ff) = nanmedian(dat.RMS)*scalingFactor;
    else
        rmsq(ff) = nan;
    end
end

%% get mean RMS per animal

meanRMS = nan(numel(subjectsToInspect),2);
for ss = 1:numel(subjectsToInspect)
    subjectIdx = contains(subjectsAll,subjectsToInspect(ss));
    probes = unique(probeSNAll(subjectIdx));
    colAniToInspect = colAni(ismember(subjects,subjectsToInspect(ss)),:);
    for pp = 1:numel(probes)

        % Check number of uses for this probe
        [~,useNum(ss,pp)] = find(contains(probeInfo.implantedSubjects{strcmp(probeSNUni,probes(pp))},subjectsToInspect{ss}));

        probeIdx = contains(probeSNAll,probes(pp));
        subAndProbeIdx = find(subjectIdx & probeIdx);
        recLocGood = recLocAll(subAndProbeIdx);

        meanRMS(ss,pp) = nanmedian(qm(ismember(recLocAll, recLocGood)));
    end
end

%% Then use plotting function from main? 
% will need to fetch some variables in the main script

plotQuantifSummary(meanRMS, subj, useNum, probeInfo, exSubj, 'rms', colAni(ssEx,:))

%% Example distributions for different probe uses?



%% Quantify freely moving vs. headfixed?