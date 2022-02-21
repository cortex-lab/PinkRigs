function spk = getSpikeData(ephysPath,varargin)
    %%% This function will load the spike data (spike time, clusters,
    %%% but also cluster info, etc.). "spk" contains a condensed summary
    %%% with spikes times & cluster info.
    %%%
    %%% This code is inspired by the code from kilotrode
    %%% (https://github.com/cortex-lab/kilotrodeRig).

    %% Parameters
    params.loadPCs = true; % to be able to compute the depth
    params.excludeNoise = true;
    
    if ~isempty(varargin)
        paramsIn = varargin{1};
        params = parseInputParams(params,paramsIn);
    end
    
    KSFolder = fullfile(ephysPath,'kilosort2');
    
    %% Get spike data
    
    sp = loadKSdir(KSFolder,params);
    
    %% Compute depths

    if isfield(sp,'pcFeat')
        pcFeat = sp.pcFeat;
        pcFeat = squeeze(pcFeat(:,1,:)); % take first PC only
        pcFeat(pcFeat<0) = 0; % some entries are negative, but we don't really want to push the CoM away from there.
        
        % which channels for each spike?
        spikeFeatInd = sp.pcFeatInd(sp.spikeTemplates+1,:);
        % ycoords of those channels?
        spikeFeatYcoords = sp.ycoords(spikeFeatInd+1); % 2D matrix of size #spikes x 12
        % center of mass is sum(coords.*features)/sum(features)
        spikeDepths = sum(spikeFeatYcoords.*pcFeat.^2,2)./sum(pcFeat.^2,2);
        
        spikeFeatXcoords = sp.xcoords(spikeFeatInd+1); % 2D matrix of size #spikes x 12
        spikeXPos = sum(spikeFeatXcoords.*pcFeat.^2,2)./sum(pcFeat.^2,2);
    end
        
    %% Group it by clusters
    %%% clusterList not exactly the same as sp.cids..???
    
    clusterList = unique(sp.clu);
    for ii = 1:numel(clusterList)
        cl = clusterList(ii);
        spkIdx = sp.clu == cl;
        spk(ii).spikeTimes = sp.st(spkIdx);
        spk(ii).ID = sp.cids(ii)';
        spk(ii).KSLab = sp.cgs(ii);
        spk(ii).Spknum = numel(spk(ii).spikeTimes);
        if exist('spikeDepths','var')
            spk(ii).XPos = nanmean(spikeXPos(spkIdx)); % not sure why there can be nans here
            spk(ii).Depth = nanmean(spikeDepths(spkIdx));
        else 
            spk(ii).XPos = nan;
            spk(ii).Depth = nan;
        end
    end
    
    %% Load and match quality metrics
    qMetricFile = fullfile(KSFolder,'qualityMetrics.mat');
    if exist(qMetricFile,'file')
        load(qMetricFile,'qMetric');
        fieldsQM = fieldnames(qMetric); fieldsQM(contains(fieldsQM,'clusterID')) = [];
        
        % /!\ qMetrics is in indexing 1, KS output is in indexing 0
        % match cluster IDs
        for ii = 1:numel(clusterList)
            cl = double(clusterList(ii));
            clQMidx = qMetric.clusterID == cl+1;
            for f = 1:numel(fieldsQM)
                fieldname = fieldsQM{f};
                spk(ii).(fieldname) = qMetric.(fieldname)(clQMidx);
            end
        end
    end