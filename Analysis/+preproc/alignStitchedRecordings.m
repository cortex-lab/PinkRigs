% align recordings that have been stitched by pyKilosort
clc; clear;
clear params
params.mice2Check = 'AV008';
%params.days2Check = 1;
params.days2Check = {'2022-03-11'};
params.expDef2Check = 'AVPassive_ckeckerboard_postactive';
% params.timeline2Check = 1;
% params.align2Check = '*,*,*,*,*,~1'; % "any 0"
% params.preproc2Check = '*,2';
exp2checkList = csv.queryExp(params);
%%
expInfo = exp2checkList(1,:);
paramsKS.KSdir = 'D:\output';


% get quality metrics for this recording - actually, I don't think this
% works either because of course the dataset itself is not stitched. 

% if ~exist(fullfile(paramsKS.KSdir,'qualityMetrics.mat'))
%     kilo.getQualityMetrics(paramsKS.KSdir,ephysFolder)
% end 
% load the params file which should tell which index is the queried
% recording...
% I don't know why the loadparamsPy does not parse the dat_path properly 
%
paramsPyfile = fullfile(paramsKS.KSdir,'params.py');
fid = fopen(paramsPyfile, 'r');
mcFileMeta = textscan(fid, '%s%s', 'Delimiter', '=',  'ReturnOnError', false);
fclose(fid);
filenames = mcFileMeta{2}{1};

% parse the filenames to be able to get the dataset_idx of the recording in
% question
rec_separation_idx = regexp(filenames,'.ap.bin');
datecheck = regexp(filenames,params.days2Check{1});
dataset_idx = numel(find(rec_separation_idx-datecheck(1)<0)); % note that this indexing starts from 0 since pyKS
%
alignmentFile = dir(fullfile(expInfo.expFolder{1},'*alignment.mat'));
alignment = load(fullfile(alignmentFile.folder,alignmentFile.name),'ephys','block');

expPath = expInfo.expFolder{1};
block = getBlock(expPath);

spk = cell(1,numel(alignment.ephys));
sp = cell(1,numel(alignment.ephys));

for probeNum = 1:numel(alignment.ephys)
    % Get spikes times & cluster info
    % the KSdir might be different down the line...
    [spk{probeNum},sp{probeNum}] = preproc.getSpikeData(alignment.ephys(probeNum).ephysPath,paramsKS);
    
    spk{probeNum}.spikes.time = preproc.align.event2Timeline(spk{probeNum}.spikes.time, ...
        alignment.ephys(probeNum).originTimes,alignment.ephys(probeNum).timelineTimes);

    % Subselect the ones that are within this experiment
    expLength = block.duration;
    spk2keep = (spk{probeNum}.spikes.time>0) & (spk{probeNum}.spikes.time<expLength) & (sp{probeNum}.spk_dataset_idx==dataset_idx);
    spk{probeNum}.spikes.time = spk{probeNum}.spikes.time(spk2keep);
    spk{probeNum}.spikes.cluster = spk{probeNum}.spikes.cluster(spk2keep);
    spk{probeNum}.spikes.xpos = spk{probeNum}.spikes.xpos(spk2keep);
    spk{probeNum}.spikes.depth = spk{probeNum}.spikes.depth(spk2keep);
    spk{probeNum}.spikes.tempScalingAmp = spk{probeNum}.spikes.tempScalingAmp(spk2keep);
end

% save the spike data 
expPath = expInfo.expFolder{1};
        
% Define savepath for the preproc results
[subject, expDate, expNum] = parseExpPath(expPath);
savePath = fullfile(expPath,[expDate '_' expNum '_' subject '_spkData_stitch.mat']);
save(savePath,'spk')      
