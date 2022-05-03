function cellRaster(spk, eventTimes, trialGroups, opt)
%% Cell raster "browser"
% NOTE: Designed to operate on the output of the PinkAV Rigs pipeline.
% Letters below are used as follows:
%    "p": number of probes
%    "s": number of spikes
%    "c": number of clusters
%    "n": number of sets of eventTimes
%    "m": number of eventTimes within a set
%    "q": number of distinct trial groupings for each set of event times


% INPUTS(default values)
% spk(Required)-----------[px1] cell of structs: spike info for p probes in the same recording
%   spk{1}.spikes.time-------------[sx1] matrix: time of each spike
%   spk{1}.spikes.cluster----------[sx1] matrix: cluster ID for each spike
%   spk{1}.spikes.tempScalingAmp---[sx1] matrix: scaled amplitude for each spike
%   spk{1}.clusters.ID-------------[cx1] matrix: ID for each cluster
%   spk{1}.clusters.Depth----------[cx1] matrix: Distance form probe tip for each cluster
%   spk{1}.clusters.XPos-----------[cx1] matrix: x position for each cluster (0 = shank 0)

% eventTimes(Required)-------[nx1] cell array of [mx1] matrices: "n" sets event times
%   ALTERNATIVELY can be "ev" from the preproc file. See "opt.customPipe" below
%   NOTE: "nans" will be automatically removed

% trialGroups(ones(m,1))-----[nx1] cell array of [mxq] matrices: "q" sets of trial labels for events
%   NOTE: A uniform label (ones(p,1)) will be added to the first column if absent

% opt-------Struct with optional inputs. Each of these will have default values
%
%    .paramTag('default')-----string: identifier for param set in combination with "ev" input
%       NOTE: custom paramSets can be stored in +rasterParams folder. paramTag is function name
%
%    .groupNames('Unsorted')------[nx1] cell array of [1xq] cell arrays: names for each trial group
%       NOTE: each cell array should contain a name for each unique value within that trial group
%
%    .sortClusters('sig')------string OR [nx1] cell array of [cx1] matrices: how to sort clusters
%       NOTE: This only affects the order of cycling through clusters (with up/down arrow)
%
%    .sortTrials(eventTimes)------[nx1] cell array of [mx1] matrices: indicates how to sort trials
%       NOTE: Will sort trials within each trial group according to increasing values in .sortTrials
%
%    .trialTickTimes([])--[nx1] cell array of [mx1] matrices: additional raster ticks (e.g. movements)
%
%    .highlight([])-------[nx1] cell array of [cx1] logical matrices: visually highligts indicated clusters


%% Controls:
% Up/Down arrow keys:             switch between different clusters
% Left/Right arrow keys:          switch between different groups of trial labels
% Shift+Left/Right arrow keys:    switch between different sets of event times
% "c" key:                        manually select a cluster

%% Input validation and contruction
% Validate eventTimes input
if isstruct(spk); spk  = {spk}; end
if ~exist('eventTimes','var'); error('No event times'); end
if ~exist('opt','var'); opt = struct; end
if ~iscell(eventTimes) && ~isstruct(eventTimes); eventTimes = {eventTimes}; end

% If "ev" has been input instead of event times, load a parameter set with corresponding function
if ~isfield('opt','paramTag'); opt.paramTag = 'default'; end
if isstruct(eventTimes)
    ev = eventTimes;
    fprintf('eventTimes appears to be "ev" so will load preset parameters \n');
    paramFunc = str2func(['plt.spk.rasterParams.' opt.paramTag]);
    [eventTimes, trialGroups, opt] = paramFunc(ev);
end

% Validate trialGroups input and assign default (uniform grouping)
if ~exist('trialGroups','var') || isempty(trialGroups)
    trialGroups = cellfun(@(x) x*0+1, eventTimes, 'uni' ,0); 
    opt.groupNames = repmat({'Unsorted'}, length(trialGroups),1);
end
trialGroups = inputLengthCheck(eventTimes, trialGroups);
noDefault = cellfun(@(x) ~all(x(:,1)==1 | isnan(x(:,1))), trialGroups);
trialGroups(noDefault) = cellfun(@(x) [x(:,1)*0+1, x],trialGroups(noDefault),'uni',0);

% Validate "opt" fields
tEvts = cellfun(@(x) x*0+1, eventTimes, 'uni', 0);
tClust = cellfun(@(x) [x.clusters.ID]'*0>0, spk, 'uni', 0);

if ~isfield(opt, 'highlight'); opt.highlight = tClust; 
else, opt.highlight = inputLengthCheck(tClust, opt.highlight, 'opt.highlight');
end

if ~isfield(opt, 'sortClusters'); opt.sortClusters = 'sig';
elseif ischar(opt.sortClusters) && ~contains(lower(opt.sortClusters(1:3)), {'dep';'sig'})
    error('Unrecognized tag for opt.sortClusters');
elseif ~ischar(opt.sortClusters)
    opt.sortClusters = inputLengthCheck(tClust, opt.sortClusters, 'opt.sortClusters');
end

if ~isfield(opt, 'sortTrials'); opt.sortTrials = tEvts; 
else, opt.sortTrials = inputLengthCheck(tEvts, opt.sortTrials, 'opt.sortTrials');
end

if ~isfield(opt, 'trialTickTimes'); opt.trialTickTimes = tEvts; 
else, opt.trialTickTimes = inputLengthCheck(tEvts, opt.trialTickTimes, 'opt.trialTickTimes');
end

if isfield(opt, 'groupNames')
    if all(cellfun(@(x) size(x,2), trialGroups(:))==(cellfun(@length, opt.groupNames(:)))+1)
        opt.groupNames = cellfun(@(x) ['Unsorted'; x(:)]', opt.groupNames, 'uni', 0);
    elseif all(cellfun(@(x) size(x,2), trialGroups(:))==(cellfun(@length, opt.groupNames(:))))
        error('opt.groupNames does not match column numbers in trialGroups')
    end
end

if isfield(opt, 'eventNames')
    if ~iscell(opt.eventNames) || length(opt.eventNames)~=length(eventTimes)
        error('opt.eventNames does not match length of eventTimes input')
    end
else, opt.eventNames = repmat({'Not Provided'}, length(eventTimes), 1);
end

% Create anon function to remove any NaN values from the eventTimes and corresponding cells
nanIdx = cellfun(@(x) ~isnan(x(:,1)), eventTimes, 'uni', 0);
remNans = @(x) cellfun(@(y,z) y(z,:), x, nanIdx, 'uni', 0);

%% Package gui data
cellrasterGui = figure('color','w');
guiData = struct;
guiData.title = annotation('textbox', [0.25, 0.98, 0.5, 0], 'string', 'My Text', 'EdgeColor', 'none',...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

% Generate axes (with dummy data) and figure
nCol = 4;
nRow = 5;
nAxes = nCol*nRow;

guiData.axes.clusters = subplot(nRow,nCol,1:nCol:(nAxes-nRow),'YDir','normal'); hold on;
xlim([-0.1,1]);
ylabel('Distance from tip (\mum)')
xlabel('xPosition (\mum)')
disableDefaultInteractivity(gca)

tRef = nCol-2:nCol;
guiData.axes.psth = subplot(nRow,nCol,tRef,'YAxisLocation','right'); hold on;
xlabel('Time from event (s)');
ylabel('spks/s/trial');
disableDefaultInteractivity(gca)

guiData.axes.raster = subplot(nRow,nCol,[tRef+nCol,tRef+nCol*2,tRef+nCol*3], ...
    'YDir','reverse','YAxisLocation','right'); hold on;
xlabel('Time from event (s)');
ylabel('Trial');
disableDefaultInteractivity(gca)

guiData.axes.amplitude = subplot(nRow,nCol,nAxes-nCol+1:nAxes); hold on;
xlabel('Time (s)');
ylabel('Amplitude');
axis tight
disableDefaultInteractivity(gca)

guiData.plot.clusterDots = [];

guiData.spk = spk;
guiData.curr.probe = 1;
guiData.curr.eventTimeRef = 1;
guiData.curr.trialGroupRef = 1;

guiData.eventTimes = remNans(eventTimes);
guiData.eventNames = opt.eventNames;
guiData.trialGroups = remNans(trialGroups);
guiData.groupNames = opt.groupNames;
guiData.sortClusters = opt.sortClusters;
guiData.sortTrials = remNans(opt.sortTrials);
guiData.trialTickTimes = remNans(opt.trialTickTimes);
guiData.highlight = opt.highlight;

guidata(cellrasterGui, guiData);
cycleProbe(cellrasterGui);
updatePlot(cellrasterGui);
end

%% Little function to validate length of inputs that should match eventTimes
function input2Check = inputLengthCheck(eventTimes, input2Check, inputTag)
if ~exist('inputTag', 'var'); inputTag = inputname(2); end

nEventSets = length(eventTimes);
if ~iscell(input2Check); input2Check = {input2Check}; end
if length(input2Check) ~= nEventSets
    if length(input2Check) == 1
        fprintf('Replicating %s to match number of event sets... \n', inputTag)
        input2Check = repmat(input2Check, nEventSets, 1);
    else
        error('Mismatching number of cells for eventTimes/highlights');
    end
end

% Check corresponding eventTimes and trialGroup cells have same number of rows
sizeCheck = cellfun(@(x,y) size(x,1) == size(y,1), eventTimes, input2Check, 'uni', 0);
if ~all(cell2mat(sizeCheck))
    error(['each cell for eventTimes & ' inputTag ' should have same number of rows'])
end
end

%%
function cycleProbe(cellrasterGui)
guiData = guidata(cellrasterGui);

% Populate guiData with current spike/cluster info from current probe
spk = guiData.spk{guiData.curr.probe};

guiData.curr.spkTimes = spk.spikes.time;
guiData.curr.spkCluster = spk.spikes.cluster;
guiData.curr.spkAmps = spk.spikes.tempScalingAmp;

guiData.curr.clustDepths = [spk.clusters.Depth]';
guiData.curr.clustXPos = [spk.clusters.XPos]';
guiData.curr.clustIDs = [spk.clusters.ID]';

% Remove the zero idx (if present) from clusters;
guiData.pythonModIdx = 0;
if min(guiData.curr.spkCluster) == 0
    guiData.pythonModIdx = 1;
    guiData.curr.spkCluster = guiData.curr.spkCluster+1; 
    guiData.curr.clustIDs = guiData.curr.clustIDs+1; 
end

%Populate current eventTimes and trialGroups
guiData.curr.eventTimes = guiData.eventTimes{guiData.curr.eventTimeRef};
guiData.curr.trialGroups = guiData.trialGroups{guiData.curr.eventTimeRef};

%Sort clusters (order of cycling with arrow keys) according to optional inputs
if ischar(guiData.sortClusters)
    switch guiData.sortClusters(1:3)
        case 'dep'
            [~, clusterSortIdx] = sort(guiData.curr.clustDepths);
        case 'sig'
            guiData.sigRes = neural.findResponsiveCells(spk,guiData.curr.eventTimes);
            [~, clusterSortIdx] = sort(guiData.sigRes.pVal);
            if isempty(guiData.highlight{guiData.curr.probe}) || ~any(guiData.highlight{guiData.curr.probe})
                guiData.highlight{guiData.curr.probe} = guiData.sigRes.pVal<0.01;
            end
    end
else
    [~, clusterSortIdx] = sort(guiData.sortClusters{guiData.curr.eventTimeRef});
end
guiData.curr.sortedClustIDs = guiData.curr.clustIDs(clusterSortIdx);
guiData.curr.clust = guiData.curr.sortedClustIDs(1);
guiData.curr.clustIdx = find(guiData.curr.clust == guiData.curr.clustIDs);

% Initialize figure and axes
axes(guiData.axes.clusters); cla;
ylim([min(guiData.curr.clustDepths)-50 max(guiData.curr.clustDepths)+50]);
xlim([min(guiData.curr.clustXPos)-25 max(guiData.curr.clustXPos)+25]);

% (plot cluster depths by depth and xPos)
highlight = guiData.highlight{guiData.curr.probe};
plot(guiData.curr.clustXPos(highlight),guiData.curr.clustDepths(highlight),'.b','MarkerSize',25);
clusterDots = plot(guiData.curr.clustXPos,guiData.curr.clustDepths,'.k','MarkerSize',15,'ButtonDownFcn',@clusterClick);

currClusterDot = plot(0,0,'.r','MarkerSize',20);

% (smoothed psth)
axes(guiData.axes.psth); cla;
maxNumGroups = 10; %HARDCODED for now... unlikely to be more than 10 trial labels...
psthLines = arrayfun(@(x) plot(NaN,NaN,'linewidth',2,'color','k'),1:maxNumGroups);
psthOnset = xline(0, ':c', 'linewidth', 3);

% (raster)
axes(guiData.axes.raster); cla;
rasterDots = scatter(NaN,NaN,5,'k','filled');
rasterOnset = xline(0, ':c', 'linewidth', 3);
addRasterTicks = scatter(NaN,NaN,5,'c','filled');

% (spk amplitude across the recording)
axes(guiData.axes.amplitude); cla;
amplitudePlot = plot(NaN,NaN,'.k');

% Set default raster times
rasterWindow = [-0.5,1];
psthBinSize = 0.001;
tBins = rasterWindow(1):psthBinSize:rasterWindow(2);
t = tBins(1:end-1) + diff(tBins)./2;

% Set functions for key presses
set(cellrasterGui,'WindowKeyPressFcn',@keyPress);

% (plots)
guiData.plot.clusterDots = clusterDots;
guiData.plot.currClusterDot = currClusterDot;
guiData.plot.psthLines = psthLines;
guiData.plot.psthOnset = psthOnset;
guiData.plot.rasterDots = rasterDots;
guiData.plot.rasterOnset = rasterOnset;
guiData.plot.addRasterTicks = addRasterTicks;
guiData.plot.amplitudes = amplitudePlot;

% (raster times)
guiData.plot.rasterTime = t;
guiData.plot.rasterBins = tBins;

% Upload gui data and draw
guidata(cellrasterGui, guiData);
end

function updatePlot(cellrasterGui)
% Get guidata
guiData = guidata(cellrasterGui);
guiData.curr.eventTimes = guiData.eventTimes{guiData.curr.eventTimeRef};
guiData.curr.trialGroups = guiData.trialGroups{guiData.curr.eventTimeRef};
if guiData.curr.trialGroupRef > size(guiData.curr.trialGroups,2)
    guiData.curr.trialGroupRef = size(guiData.curr.trialGroups,2);
end
guiData.curr.clustIdx = find(guiData.curr.clust == guiData.curr.clustIDs);

% Turn on/off the appropriate graphics
set(guiData.plot.rasterDots,'visible','on');

% Plot depth location on probe
clusterX = get(guiData.plot.clusterDots,'XData');
clusterY = get(guiData.plot.clusterDots,'YData');
set(guiData.plot.currClusterDot,'XData',clusterX(guiData.curr.clustIdx), 'YData',clusterY(guiData.curr.clustIdx));

% Bin spks (use only spks within time range, big speed-up)
currSpkIdx = ismember(guiData.curr.spkCluster,guiData.curr.clust);
currRasterSpkTimes = guiData.curr.spkTimes(currSpkIdx);

tPeriEvent = guiData.curr.eventTimes + guiData.plot.rasterBins;
% (handle NaNs by setting rows with NaN times to 0)
tPeriEvent(any(isnan(tPeriEvent),2),:) = 0;

currRasterSpkTimes(currRasterSpkTimes < min(tPeriEvent(:)) | ...
    currRasterSpkTimes > max(tPeriEvent(:))) = [];

tDat = tPeriEvent';
if ~any(diff(tDat(:))<0)
    currRaster = [histcounts(currRasterSpkTimes,tDat(:)),0];
    currRaster = reshape(currRaster, size(tPeriEvent'))';
    currRaster(:,end) = [];
else
    currRaster = cell2mat(arrayfun(@(x) ...
        histcounts(currRasterSpkTimes,tPeriEvent(x,:)), ...
        [1:size(tPeriEvent,1)]','uni',false));
end

% Set color scheme
currGroup = guiData.curr.trialGroups(:,guiData.curr.trialGroupRef);
currAddTicks = guiData.trialTickTimes{guiData.curr.eventTimeRef};
if any(currAddTicks)
    if max(currAddTicks) > 100
        currAddTicks = currAddTicks - guiData.eventTimes{guiData.curr.eventTimeRef};
    end
end
if length(unique(currGroup)) == 1
    % Black if one group
    groupColors = [0,0,0];
elseif length(unique(sign(currGroup(currGroup ~= 0)))) == 1
    % Black-to-red single-signed groups
    nGroups = length(unique(currGroup));
    groupColors = [linspace(0,0.8,nGroups)',zeros(nGroups,1),zeros(nGroups,1)];
elseif length(unique(sign(currGroup(currGroup ~= 0)))) == 2
    % Symmetrical blue-black-red if negative and positive groups
    nGroupsPos = length(unique(currGroup(currGroup > 0)));
    groupColorsPos = [linspace(0.3,1,nGroupsPos)',zeros(nGroupsPos,1),zeros(nGroupsPos,1)];
    
    nGroupsNeg = length(unique(currGroup(currGroup < 0)));
    groupColorsNeg = [zeros(nGroupsNeg,1),zeros(nGroupsNeg,1),linspace(0.3,1,nGroupsNeg)'];
    
    nGroupsZero = length(unique(currGroup(currGroup == 0)));
    groupColorsZero = [zeros(nGroupsZero,1),zeros(nGroupsZero,1),zeros(nGroupsZero,1)];
    
    groupColors = [flipud(groupColorsNeg);groupColorsZero;groupColorsPos];    
end

% Plot smoothed PSTH
gw = gausswin(31,3)';
smWin = gw./sum(gw);
currPsth = grpstats(currRaster,currGroup,@(x) mean(x,1));
padStart = repmat(mean(currPsth(:,1:10),2), 1, floor(length(smWin)/2));
padEnd = repmat(mean(currPsth(:,end-9:end),2), 1, floor(length(smWin)/2));
currSmoothedPsth = conv2([padStart,currPsth,padEnd], smWin,'valid')./mean(diff(guiData.plot.rasterTime));

% (set the first n lines by group, set all others to NaN)
arrayfun(@(x) set(guiData.plot.psthLines(x), ...
    'XData',guiData.plot.rasterTime,'YData',currSmoothedPsth(x,:), ...
    'Color',groupColors(x,:)),1:size(currPsth,1));
arrayfun(@(align_group) set(guiData.plot.psthLines(align_group), ...
    'XData',NaN,'YData',NaN), ...
    size(currPsth,1)+1:length(guiData.plot.psthLines));

ylim(get(guiData.plot.psthLines(1),'Parent'),[min(currSmoothedPsth(:)), ...
    max(max(currSmoothedPsth(:),min(currSmoothedPsth(:))+1))]);

% Plot raster
% (single cluster mode)
[~,~,rowGroup] = unique(currGroup,'sorted');
[~, rasterSortIdx] = sortrows([currGroup, currAddTicks], [1,2]);
currRaster = currRaster(rasterSortIdx,:);
rowGroup = rowGroup(rasterSortIdx);
currAddTicks = currAddTicks(rasterSortIdx);

[rasterY,rasterX] = find(currRaster);
set(guiData.plot.rasterDots,'XData',guiData.plot.rasterTime(rasterX),'YData',rasterY);
xlim(get(guiData.plot.rasterDots,'Parent'),[guiData.plot.rasterBins(1),guiData.plot.rasterBins(end)]);
ylim(get(guiData.plot.rasterDots,'Parent'),[0,size(tPeriEvent,1)]);
if any(currAddTicks)
    set(guiData.plot.addRasterTicks,'XData',currAddTicks,'YData',1:length(currAddTicks));
end

% (set dot color by group)
psthColors = get(guiData.plot.psthLines,'color');
if iscell(psthColors); psthColors = cell2mat(psthColors); end
rasterDotColor = psthColors(rowGroup(rasterY),:);
set(guiData.plot.rasterDots,'CData',rasterDotColor);


% Plot cluster amplitude over whole experiment
set(guiData.plot.amplitudes,'XData', ...
    guiData.curr.spkTimes(currSpkIdx), ...
    'YData',guiData.curr.spkAmps(currSpkIdx),'linestyle','none');

assignin('base','guiData',guiData)
set(guiData.title, 'String', sprintf('ClusterID--%d   Alignment--%s   Grouping--%s', ...
    guiData.curr.clust-guiData.pythonModIdx, guiData.eventNames{guiData.curr.eventTimeRef}, ...
    guiData.groupNames{guiData.curr.eventTimeRef}{guiData.curr.trialGroupRef}));
end


function keyPress(cellrasterGui,eventdata)
% Get guidata
guiData = guidata(cellrasterGui);

switch eventdata.Key
    case 'p' %Switch probe
        guiData.curr.probe = guiData.curr.probe + 1;
        if guiData.curr.probe > length(guiData.spk)
            guiData.curr.probe = 1;
        end
        guidata(cellrasterGui,guiData);
        cycleProbe(cellrasterGui);
        guiData = guidata(cellrasterGui);
        
    case 'downarrow' % Next cluster
        currClusterPosition = guiData.curr.clust == guiData.curr.sortedClustIDs;
        newCluster = guiData.curr.sortedClustIDs(circshift(currClusterPosition,1));
        guiData.curr.clust = newCluster;
        
    case 'uparrow' % Previous cluster
        currClusterPosition = guiData.curr.clust(end) == guiData.curr.sortedClustIDs;
        newCluster = guiData.curr.sortedClustIDs(circshift(currClusterPosition,-1));
        guiData.curr.clust = newCluster;

    case 'rightarrow' % Next trial group or event times (if shift pressed)
        if contains(eventdata.Modifier, 'shift')
            newEventTimes = guiData.curr.eventTimeRef + 1;
            if newEventTimes > length(guiData.eventTimes)
                newEventTimes = 1;
            end
            guiData.curr.eventTimeRef = newEventTimes;
        else
            newTrialGroup = guiData.curr.trialGroupRef + 1;
            if newTrialGroup > size(guiData.trialGroups{guiData.curr.eventTimeRef},2)
                newTrialGroup = 1;
            end
            guiData.curr.trialGroupRef = newTrialGroup;
        end

    case 'leftarrow' % Previous trial group or event times (if shift pressed)
         if contains(eventdata.Modifier, 'shift')
             newEventTimes = guiData.curr.eventTimeRef - 1;
             if newEventTimes <= 0 
                 newEventTimes = length(guiData.eventTimes);
             end
             guiData.curr.eventTimeRef = newEventTimes;
         else
             newTrialGroup = guiData.curr.trialGroupRef - 1;
             if newTrialGroup < 1
                 newTrialGroup = size(guiData.trialGroups{guiData.curr.eventTimeRef},2);
             end
             guiData.curr.trialGroupRef = newTrialGroup;
         end

    case 'c'
        % Enter and go to cluster
        newCluster = str2double(cell2mat(inputdlg('Go to cluster:')));
        if ~ismember(newCluster,unique(guiData.curr.sortedClustIDs))
            error(['Cluster ' num2str(newCluster) ' not present'])
        end
        guiData.curr.clust = newCluster;
end

% Upload gui data and draw
guidata(cellrasterGui,guiData);
updatePlot(cellrasterGui);
end

function clusterClick(cellrasterGui,eventdata)
% Get guidata
guiData = guidata(cellrasterGui);

% Get the clicked cluster, update current cluster
clusterX = get(guiData.plot.clusterDots,'XData');
clusterY = get(guiData.plot.clusterDots,'YData');

[~,clickedCluster] = min(sqrt(sum(([clusterX;clusterY] - ...
    eventdata.IntersectionPoint(1:2)').^2,1)));
guiData.curr.clust = guiData.curr.clustIDs(clickedCluster);

% Upload gui data and draw
guidata(cellrasterGui,guiData);
updatePlot(cellrasterGui);
end
