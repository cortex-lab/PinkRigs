function sigResults = findResponsiveCells(spk,eventTimes,tWin)
%% Returns p-values for cell responses to a set of event times
%
% NOTE: This code uses some tricks so that it can test a lot of clusters
% with relatively little coding time
%
% Parameters:
% ------------
% spk (required): stuct/cell array of structs
%   This is the structure that is output from the PinkRigs pipeline. For
%   this function spk.spikes.times (spike times) and spk.spikes.clusters
%   (the cluster ID for each spk), and spk.clusters.IDs (the cluster ID for
%   each cluster) are used.
%
% eventTimes (requied): cell or cell array
%   Each cell of eventTimes should be the eventTimes relative to which the
%   significance of each cluster (in the corresponding stuct) will be
%   evaluated. Note, if you want to test multiple sets of event times with
%   the same cluster, you can just replicated the "spk" input.
%  
% tWin (default=[-0.3 -0.01 0.01 0.3]): nx4 vector
%   These 4 values represent two time windows relative to the eventTimes
%   (before and after). In these windows, the mean spike rate will be
%   calculated and a t-test for before vs after will be performed across
%   trials. Time 0 is the eventTime, and tWin(1:2) is the pre-event window
%   and tWin(3:4) is the post-event window.
%
%
% Returns: 
% -----------
% sigResults: structure array
%   .clusterID = clusterIDs for each cluster
%   .pVal = the p value for the t-test for each cluster
%   .spikeCountsPre = The spike count during each pre-event window;
%   .spikeCountsPost = The spike count during each post-event window;

if ~iscell(spk); spk  = {spk}; end
if iscell(eventTimes); eventTimes = eventTimes{1}; end
eventTimes(isnan(eventTimes)) = [];

if ~exist('tWin', 'var') || isempty(tWin); tWin = [-0.3 -0.01 0.01 0.3];
elseif numel(tWin)~=4; error('tWin should be 1x4 vector');
elseif ~all([(tWin(1:2)<=0) (tWin(3:4)>=0)]); error('pre/post windows should be negative/positive (or zero)');
end

% Note: this looping is faster because it means smaller subsets are indexed when searching.
sigResults = struct;
for i = 1:length(spk)
    spkTimes = spk{i}.spikes.times;
    spkCluster = spk{i}.spikes.clusters;
    clusterIDs = [spk{i}.clusters.IDs]';

    matlabMod = 0;
    if min(clusterIDs) == 0
        matlabMod = 1;
        spkCluster = spkCluster+1;
        clusterIDs = clusterIDs+1;
    end

    eventWindows = eventTimes+tWin;
    spikeCounts = histcounts2(spkTimes, spkCluster, sort(eventWindows(:)), 1:(max(clusterIDs)+1));
    spikeCountsPre = spikeCounts(1:4:end,clusterIDs)./range(tWin(1:2));
    spikeCountsPost = spikeCounts(3:4:end,clusterIDs)./range(tWin(3:4));
    ttestPrePost = spikeCountsPost - spikeCountsPre;

    pVal = zeros(size(ttestPrePost,2),1);
    for k = 1:size(ttestPrePost,2) 
        pVal(k) = signrank(ttestPrePost(:,k));
    end

    sigResults(i).clusterID = clusterIDs - matlabMod;
    sigResults(i).pVal = pVal;
    sigResults(i).spikeCountsPre = spikeCountsPre';
    sigResults(i).spikeCountsPost = spikeCountsPost';
end



