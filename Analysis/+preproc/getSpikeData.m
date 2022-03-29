function [spk,sp] = getSpikeData(ephysPath,varargin)
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
    
    spk.spikes.time = sp.st;
    spk.spikes.cluster = sp.clu;
    spk.spikes.tempScalingAmp = sp.tempScalingAmps;
    
    %% Compute depths
    %%% SHOULD FIND A BETTER WAY TO DO IT?
    
    if isfield(sp,'pcFeat')
        pcFeat = sp.pcFeat;
        pcFeat = squeeze(pcFeat(:,1,:)); % take first PC only
        pcFeat(pcFeat<0) = 0; % some entries are negative, but we don't really want to push the CoM away from there.
        
        % which channels for each spike?
        spikeFeatInd = sp.pcFeatInd(sp.spikeTemplates+1,:);
        % ycoords of those channels?
        spikeFeatYcoords = sp.ycoords(spikeFeatInd+1); % 2D matrix of size #spikes x 12
        % center of mass is sum(coords.*features)/sum(features)
        sp.spikeDepths = sum(spikeFeatYcoords.*pcFeat.^2,2)./sum(pcFeat.^2,2);
        
        spikeFeatXcoords = sp.xcoords(spikeFeatInd+1); % 2D matrix of size #spikes x 12
        sp.spikeXPos = sum(spikeFeatXcoords.*pcFeat.^2,2)./sum(pcFeat.^2,2);
        
        spk.spikes.xpos = sp.spikeXPos;
        spk.spikes.depth = sp.spikeDepths;
    end
        
    %% Group it by clusters
    %%% clusterList not exactly the same as sp.cids..???
    
    clusterList = unique(sp.clu);
    
    spk.clusters = struct([]);
    for ii = 1:numel(clusterList)
        cl = clusterList(ii);
        spkIdx = sp.clu == cl;
        
        spk.clusters(ii).ID = cl;
        spk.clusters(ii).KSLab = sp.cgs(sp.cids == cl);
        spk.clusters(ii).Spknum = sum(spkIdx);
        if isfield(sp,'spikeDepths')
            spk.clusters(ii).XPos = nanmean(sp.spikeXPos(spkIdx)); % not sure why there can be nans here
            spk.clusters(ii).Depth = nanmean(sp.spikeDepths(spkIdx));
        else 
            spk.clusters(ii).XPos = nan;
            spk.clusters(ii).Depth = nan;
        end
    end
    
    %% Load and match quality metrics
    qMetricFile = fullfile(KSFolder,'qualityMetrics.mat');
    if exist(qMetricFile,'file')
        load(qMetricFile,'qMetric');
        fieldsQM = fieldnames(qMetric); fieldsQM(contains(fieldsQM,'clusterID')) = [];
        
        % /!\ qMetrics is in indexing 1, KS output is in indexing 0
        for ii = 1:numel(clusterList)
            cl = double(clusterList(ii));
            clQMidx = qMetric.clusterID == cl+1;
            for f = 1:numel(fieldsQM)
                fieldname = fieldsQM{f};
                spk.clusters(ii).(fieldname) = qMetric.(fieldname)(clQMidx);
            end
        end
    end