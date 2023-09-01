queuePath = '\\zinu.cortexlab.net\Subjects\PinkRigs\Helpers';

%% Get recordings

subjectList = {'AL032'};

serverLocations = getServersList;
sorted = [];
iblformatted = [];
D = {};
for subject = subjectList
    for server = serverLocations'
        d = dir(fullfile(server{1}, subject{1},'**','*ap.*bin'));
        d(contains(lower({d.folder}),'stitched')) = [];
        d(contains(lower({d.folder}),'catgt')) = [];
        d(contains(lower({d.folder}),'everything')) = [];
        d(contains(lower({d.folder}),'everything')) = [];
        [c,ia,ic] = unique({d.folder});
        dupidx = find(diff(ic)==0);
        for dd = 1:numel(dupidx)
            dupindices = find(strcmp({d.folder},d(dupidx(dd)).folder));
            [~,i] = max([d(dupindices).bytes]);
            d(setdiff(dupindices,dupindices(i))) = [];
        end

        dupidx = [dupidx(dd) dupidx(dd)+1];

        sorted_tmp = false(1,numel(d));
        iblformatted_tmp = false(1,numel(d));
        for dd = 1:numel(d)
            if exist(fullfile(d(dd).folder,'pyKS\output'))
                sorted_tmp(dd) = true;
            end
            if exist(fullfile(d(dd).folder,'pyKS\output\ibl_format'))
                iblformatted_tmp(dd) = true;
            end
        end

        sorted = [sorted sorted_tmp];
        iblformatted = [iblformatted iblformatted_tmp];
        D = [D d];
    end
end
D = cat(1,D{:});
dtosort = D(~sorted);
dtoformat = D(sorted & ~iblformatted);
dtoQM = D(sorted & iblformatted);

%% Update pyKS queue

M = csv.readTable(fullfile(queuePath, 'pykilosort_queue.csv'));
s = size(M,1);
for dd = 1:numel(dtosort)
    M(s+dd,1) = {fullfile(dtosort(dd).folder, dtosort(dd).name)};
    M(s+dd,2) = {'0.0'};
end
csv.writeTable(M,fullfile(queuePath, 'pykilosort_queue.csv'));

%% Change the param.py...

for dd = 1:numel(dtoformat)
    paramPath = fullfile(dtoformat(dd).folder, 'pyKS','output','params.py');
    fid = fopen(paramPath);
    C = textscan(fid,'%s','delimiter','\n');
    fclose(fid);
    C{1}{1} = regexprep(C{1}{1},'"../','r"');

    fid = fopen(paramPath,'w');
    for kk = 1:numel(C{1})
        fprintf(fid,'%s\n',C{1}{kk});
    end
    fclose(fid);
end

%% Update IBL format queue

M = csv.readTable(fullfile(queuePath, 'ibl_formatting_queue.csv'));
s = size(M,1);
for dd = 1:numel(dtoformat)
    pyKSpath = fullfile(dtoformat(dd).folder, 'pyKS');
    M(s+dd,1) = {pyKSpath};
    M(s+dd,2) = {'0.0'};
end
csv.writeTable(M,fullfile(queuePath, 'ibl_formatting_queue.csv'));

%% Run bombcell

decompressDataLocal = 'C:\Users\Experiment\Documents\KSworkfolder';
if ~exist(decompressDataLocal, 'dir')
    mkdir(decompressDataLocal)
end

recompute = 0;
for dd = 1:numel(dtoQM)
    % Set paths
    ephysKilosortPath = fullfile(dtoQM(dd).folder,'PyKS','output');
    ephysDirPath = dtoQM(dd).folder;
    ephysRawDir = dtoQM(dd);
%     ephysRawDir = dir(fullfile(ephysDirPath,'*.*bin'));
%     if numel(ephysRawDir)>1
%         idx = find(contains({ephysRawDir.name},'.cbin'));
%         if ~isempty(idx) && numel(idx)==1
%             ephysRawDir = ephysRawDir(idx);
%         end
%     end
    ephysMetaDir = dir(fullfile(ephysDirPath,'*ap.meta')); % used in bc_qualityParamValues
    savePath = fullfile(ephysKilosortPath,'qMetrics');

    qMetricsExist = ~isempty(dir(fullfile(savePath, 'templates._bc_qMetrics.parquet')));
    sortingExist = ~isempty(dir(fullfile(ephysKilosortPath,'spike_templates.npy')));

    if sortingExist && (qMetricsExist == 0 || recompute)
        % Load data
        [spikeTimes_samples, spikeTemplates, ...
            templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

        % Detect whether data is compressed, decompress locally if necessary
        rawFile = bc_manageDataCompression(ephysRawDir, decompressDataLocal);

        % Which quality metric parameters to extract and thresholds
        param = bc_qualityParamValuesForUnitMatch(ephysMetaDir, rawFile);

        % Compute quality metrics
        param.plotGlobal = 0;
        bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
            templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);

        % Delete local file if ran fine
        delete(rawFile);
    end
end