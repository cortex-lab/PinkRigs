function sigResults = findResponsiveCells(spk,eventTimes,tWin)

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



